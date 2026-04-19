import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.classifier import build_classifier
from training.dataset_index import build_or_load_index, summarize_entries


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, slice_ids):
        self.data_dir = data_dir
        self.slice_ids = slice_ids

    def __len__(self):
        return len(self.slice_ids)

    def __getitem__(self, idx):
        slice_id = self.slice_ids[idx]
        image = np.load(os.path.join(self.data_dir, f"slice_{slice_id}.npy"))
        label = np.load(os.path.join(self.data_dir, f"type_{slice_id}.npy"))[0]
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.tensor(int(label), dtype=torch.long)
        return image, label


def build_sampler(entries):
    counts = {0: 0, 1: 0}
    labels = []
    for entry in tqdm(entries, desc="Building train sampler", leave=False):
        label = int(entry["type_label"])
        labels.append(label)
        counts[label] += 1

    weights = [1.0 / max(counts[label], 1) for label in labels]
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    losses = []
    labels_all = []
    preds_all = []
    probs_all = []
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="binary", zero_division=0)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(labels_all, preds_all) if labels_all else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_val_auc, history, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_auc": best_val_auc,
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


def train_classifier(data_dir="data/processed_slices", epochs=18, batch_size=24, learning_rate=3e-4, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume:
        run_dir = os.path.dirname(os.path.abspath(resume))
    else:
        run_dir = os.path.join("training_runs", "cls_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[cls] Using device: {device}", flush=True)
    print(f"[cls] Data directory: {os.path.abspath(data_dir)}", flush=True)
    print(f"[cls] Run directory: {os.path.abspath(run_dir)}", flush=True)

    print("[cls] Building or loading training index...", flush=True)
    index = build_or_load_index(data_dir)
    all_entries = index["entries"]
    slice_ids = [entry["slice_id"] for entry in all_entries]
    print(f"[cls] Found {len(slice_ids)} total slices", flush=True)
    groups = np.array([entry["group"] for entry in all_entries])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(slice_ids, groups=groups))
    train_ids = [slice_ids[i] for i in train_idx]
    val_ids = [slice_ids[i] for i in val_idx]
    train_entries = [all_entries[i] for i in train_idx]
    val_entries = [all_entries[i] for i in val_idx]
    print(f"[cls] Train slices: {len(train_ids)} | Val slices: {len(val_ids)}", flush=True)
    class_counts = summarize_entries(train_entries)["class_counts"]
    print(f"[cls] Train class counts: GLI={class_counts.get(0, 0)} | MEN={class_counts.get(1, 0)}", flush=True)

    train_loader = DataLoader(
        ClassificationDataset(data_dir, train_ids),
        batch_size=batch_size,
        sampler=build_sampler(train_entries),
        num_workers=0,
    )
    val_loader = DataLoader(ClassificationDataset(data_dir, val_ids), batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_classifier(in_channels=4, num_classes=2, variant="enhanced").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    print(f"[cls] Starting training for {epochs} epochs", flush=True)

    best_val_auc = -1.0
    history = []
    start_epoch = 0
    config = {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    if resume:
        print(f"[cls] Resuming from checkpoint: {os.path.abspath(resume)}", flush=True)
        checkpoint = load_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_auc = float(checkpoint.get("best_val_auc", -1.0))
        history = list(checkpoint.get("history", []))
        saved_config = checkpoint.get("config", {})
        print(
            f"[cls] Resumed at epoch {start_epoch + 1} with best val auc {best_val_auc:.4f}",
            flush=True,
        )
        if saved_config:
            print(f"[cls] Saved config: {saved_config}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_acc={row['accuracy']:.4f} "
            f"val_auc={row['auc']:.4f}",
            flush=True,
        )

        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best_classifier.pth"))
            torch.save(model.state_dict(), "best_classifier.pth")
            save_checkpoint(
                os.path.join(run_dir, "best_checkpoint.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_auc=best_val_auc,
                history=history,
                config=config,
            )

        save_checkpoint(
            os.path.join(run_dir, "last_checkpoint.pth"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_auc=best_val_auc,
            history=history,
            config=config,
        )

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "classification_metrics.csv"), index=False)
    print(f"[cls] Saved best classifier with val AUC {best_val_auc:.4f}", flush=True)
    print(f"[cls] Run artifacts saved to {run_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_slices")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train_classifier(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )
