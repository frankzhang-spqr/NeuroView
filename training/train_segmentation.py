import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import build_model
from training.dataset_index import build_or_load_index, summarize_entries


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, slice_ids):
        self.data_dir = data_dir
        self.slice_ids = slice_ids

    def __len__(self):
        return len(self.slice_ids)

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        slice_img = np.load(os.path.join(self.data_dir, f"slice_{slice_id}.npy"))
        label_img = np.load(os.path.join(self.data_dir, f"label_{slice_id}.npy"))

        slice_tensor = torch.from_numpy(slice_img).float().permute(2, 0, 1)
        label_tensor = torch.from_numpy(label_img).float().unsqueeze(0)
        has_tumor = float(label_img.sum() > 0)
        return slice_tensor, label_tensor, has_tumor


def dice_loss(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.shape[0], -1)
    target = target.view(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def combined_segmentation_loss(logits, target, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )
    dice = dice_loss(logits, target)
    return 0.4 * bce + 0.6 * dice


def compute_metrics(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice = ((2 * intersection + 1.0) / (pred_sum + target_sum + 1.0)).mean().item()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()

    tp = intersection.sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def build_sampler(entries):
    weights = []
    for entry in tqdm(entries, desc="Building train sampler", leave=False):
        weights.append(6.0 if entry["has_tumor"] else 1.0)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def evaluate(model, data_loader, device, pos_weight):
    model.eval()
    losses = []
    metrics = {"dice": [], "iou": [], "precision": [], "recall": []}

    with torch.no_grad():
        for slices, labels, _ in tqdm(data_loader, desc="Validation", leave=False):
            slices = slices.to(device)
            labels = labels.to(device)
            logits = model(slices)
            loss = combined_segmentation_loss(logits, labels, pos_weight)
            losses.append(loss.item())

            batch_metrics = compute_metrics(logits, labels)
            for key, value in batch_metrics.items():
                metrics[key].append(value)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "dice": float(np.mean(metrics["dice"])) if metrics["dice"] else 0.0,
        "iou": float(np.mean(metrics["iou"])) if metrics["iou"] else 0.0,
        "precision": float(np.mean(metrics["precision"])) if metrics["precision"] else 0.0,
        "recall": float(np.mean(metrics["recall"])) if metrics["recall"] else 0.0,
    }


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_val_dice, history, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "history": history,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, *, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def train_segmentation(data_dir="data/processed_slices", epochs=25, batch_size=16, learning_rate=1e-4, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume:
        run_dir = os.path.dirname(os.path.abspath(resume))
    else:
        run_dir = os.path.join("training_runs", "seg_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[seg] Using device: {device}", flush=True)
    print(f"[seg] Data directory: {os.path.abspath(data_dir)}", flush=True)
    print(f"[seg] Run directory: {os.path.abspath(run_dir)}", flush=True)

    print("[seg] Building or loading training index...", flush=True)
    index = build_or_load_index(data_dir)
    all_entries = index["entries"]
    slice_ids = [entry["slice_id"] for entry in all_entries]
    print(f"[seg] Found {len(slice_ids)} total slices", flush=True)
    groups = np.array([entry["group"] for entry in all_entries])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(slice_ids, groups=groups))
    train_slice_ids = [slice_ids[i] for i in train_idx]
    val_slice_ids = [slice_ids[i] for i in val_idx]
    train_entries = [all_entries[i] for i in train_idx]
    val_entries = [all_entries[i] for i in val_idx]
    print(f"[seg] Train slices: {len(train_slice_ids)} | Val slices: {len(val_slice_ids)}", flush=True)

    train_dataset = SegmentationDataset(data_dir, train_slice_ids)
    val_dataset = SegmentationDataset(data_dir, val_slice_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=build_sampler(train_entries),
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("[seg] Summarizing train split...", flush=True)
    summary = summarize_entries(train_entries)
    tumor_pixels = summary["tumor_pixels"]
    background_pixels = summary["background_pixels"]
    pos_weight_value = max(background_pixels / max(tumor_pixels, 1), 1.0)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    positive_train_slices = summary["positive_slices"]
    print(
        f"[seg] Positive train slices: {positive_train_slices} | pos_weight: {pos_weight_value:.2f}",
        flush=True,
    )

    model = build_model(in_channels=4, n_class=1, variant="enhanced").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    print(f"[seg] Starting training for {epochs} epochs", flush=True)

    best_val_dice = -1.0
    history = []
    start_epoch = 0
    config = {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    if resume:
        print(f"[seg] Resuming from checkpoint: {os.path.abspath(resume)}", flush=True)
        checkpoint = load_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        history = list(checkpoint.get("history", []))
        saved_config = checkpoint.get("config", {})
        print(
            f"[seg] Resumed at epoch {start_epoch + 1} with best val dice {best_val_dice:.4f}",
            flush=True,
        )
        if saved_config:
            print(f"[seg] Saved config: {saved_config}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []

        for slices, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False):
            slices = slices.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(slices)
            loss = combined_segmentation_loss(logits, labels, pos_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            **val_metrics,
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={epoch_summary['train_loss']:.4f} "
            f"val_dice={epoch_summary['dice']:.4f} "
            f"val_iou={epoch_summary['iou']:.4f}",
            flush=True,
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            torch.save(model.state_dict(), "best_model.pth")
            save_checkpoint(
                os.path.join(run_dir, "best_checkpoint.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_dice=best_val_dice,
                history=history,
                config=config,
            )

        save_checkpoint(
            os.path.join(run_dir, "last_checkpoint.pth"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_dice=best_val_dice,
            history=history,
            config=config,
        )

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "segmentation_metrics.csv"), index=False)
    print(f"[seg] Saved best segmentation model with val dice {best_val_dice:.4f}", flush=True)
    print(f"[seg] Run artifacts saved to {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_slices")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train_segmentation(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )
