import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import build_model
from models.classifier import build_classifier

class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, slice_ids=None):
        self.data_dir = data_dir
        all_files = sorted([f for f in os.listdir(data_dir) if f.startswith('slice_')])
        if slice_ids is None:
            self.slice_files = all_files
        else:
            self.slice_files = [f"slice_{slice_id}.npy" for slice_id in slice_ids]

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        slice_file = self.slice_files[idx]
        slice_num = slice_file.split('_')[1].split('.')[0]
        
        slice_path = os.path.join(self.data_dir, slice_file)
        label_path = os.path.join(self.data_dir, f'label_{slice_num}.npy')
        type_path = os.path.join(self.data_dir, f'type_{slice_num}.npy')

        slice_img = np.load(slice_path)
        label_img = np.load(label_path)
        tumor_type = np.load(type_path)

        slice_tensor = torch.from_numpy(slice_img).float().permute(2, 0, 1)
        label_tensor = torch.from_numpy(label_img).float().unsqueeze(0)
        type_tensor = torch.from_numpy(tumor_type).long()

        return slice_tensor, label_tensor, type_tensor


def load_groups(data_dir, slice_ids):
    groups = []
    for slice_id in slice_ids:
        group_path = os.path.join(data_dir, f"group_{slice_id}.npy")
        if os.path.exists(group_path):
            groups.append(str(np.load(group_path, allow_pickle=True)[0]))
        else:
            groups.append(f"slice-{slice_id}")
    return np.array(groups)

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def combined_seg_loss(pred, target, pos_weight):
    dice = dice_loss(pred, target)
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    return 0.6 * dice + 0.4 * bce

def calculate_seg_metrics(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.cpu().numpy().astype(int).flatten()
    target = target.cpu().numpy().astype(int).flatten()

    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    fn = np.sum((1 - pred) * target)
    tn = np.sum((1 - pred) * (1 - target))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {'dice': f1, 'precision': precision, 'recall': recall, 'specificity': specificity}

def plot_metrics(metrics_df, run_dir):
    for metric in metrics_df.columns:
        if metric == 'epoch': continue
        plt.figure()
        plt.plot(metrics_df['epoch'], metrics_df[metric], label=metric)
        plt.title(f'{metric.replace("_", " ").title()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.savefig(os.path.join(run_dir, f'{metric}_plot.png'))
        plt.close()

def train_multitask(data_dir, epochs=5, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dir = os.path.join('training_runs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Training run results will be saved in: {run_dir}")

    seg_model = build_model(in_channels=4, n_class=1, variant="enhanced").to(device)
    cls_model = build_classifier(in_channels=4, num_classes=2, variant="enhanced").to(device)

    slice_ids = sorted([
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(data_dir)
        if f.startswith('slice_') and f.endswith('.npy')
    ])
    groups = load_groups(data_dir, slice_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(slice_ids, groups=groups))
    train_slice_ids = [slice_ids[i] for i in train_idx]
    val_slice_ids = [slice_ids[i] for i in val_idx]

    dataset = TumorDataset(data_dir)
    train_dataset = TumorDataset(data_dir, train_slice_ids)
    val_dataset = TumorDataset(data_dir, val_slice_ids)

    sampler_weights = []
    tumor_pixels = 0
    background_pixels = 0
    for slice_id in train_slice_ids:
        label = np.load(os.path.join(data_dir, f"label_{slice_id}.npy"))
        tumor_pixels += int(label.sum())
        background_pixels += int(label.size - label.sum())
        slice_type = int(np.load(os.path.join(data_dir, f"type_{slice_id}.npy"))[0])
        has_tumor = float(label.sum() > 0)
        sampler_weights.append((6.0 if has_tumor else 1.0) * (2.0 if slice_type == 1 else 1.0))

    pos_weight = torch.tensor([max(background_pixels / max(tumor_pixels, 1), 1.0)], device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.DoubleTensor(sampler_weights), num_samples=len(sampler_weights), replacement=True),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    params = list(seg_model.parameters()) + list(cls_model.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    
    seg_criterion = lambda pred, target: combined_seg_loss(pred, target, pos_weight)
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_loss = float('inf')
    
    seg_metrics_history = []
    cls_metrics_history = []

    for epoch in range(epochs):
        seg_model.train()
        cls_model.train()
        
        train_loss = 0.0
        for slices, labels, types in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            slices, labels, types = slices.to(device), labels.to(device), types.to(device).squeeze()
            optimizer.zero_grad()
            seg_outputs = seg_model(slices)
            cls_outputs = cls_model(slices)
            loss_seg = seg_criterion(seg_outputs, labels)
            loss_cls = cls_criterion(cls_outputs, types)
            total_loss = loss_seg + loss_cls
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        scheduler.step()

        # Validation
        seg_model.eval()
        cls_model.eval()
        val_loss = 0.0
        
        epoch_seg_metrics = {'dice': [], 'precision': [], 'recall': [], 'specificity': []}
        all_cls_preds, all_cls_probs, all_cls_labels = [], [], []

        with torch.no_grad():
            for slices, labels, types in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                slices, labels, types = slices.to(device), labels.to(device), types.to(device).squeeze()
                
                seg_outputs = seg_model(slices)
                cls_outputs = cls_model(slices)
                
                loss_seg = seg_criterion(seg_outputs, labels)
                loss_cls = cls_criterion(cls_outputs, types)
                val_loss += (loss_seg + loss_cls).item()

                # Seg metrics
                seg_metrics = calculate_seg_metrics(seg_outputs, labels)
                for k, v in seg_metrics.items():
                    epoch_seg_metrics[k].append(v)

                # Cls metrics
                cls_probs = torch.softmax(cls_outputs, dim=1)
                cls_preds = torch.argmax(cls_probs, dim=1)
                all_cls_preds.extend(cls_preds.cpu().numpy())
                all_cls_probs.extend(cls_probs.cpu().numpy())
                all_cls_labels.extend(types.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Aggregate and log metrics
        avg_seg_metrics = {k: np.mean(v) for k, v in epoch_seg_metrics.items()}
        avg_seg_metrics['epoch'] = epoch + 1
        avg_seg_metrics['loss'] = np.mean([m['dice'] for m in seg_metrics_history]) if seg_metrics_history else 0
        seg_metrics_history.append(avg_seg_metrics)

        prec, rec, f1, _ = precision_recall_fscore_support(all_cls_labels, all_cls_preds, average='binary', zero_division=0)
        cls_acc = accuracy_score(all_cls_labels, all_cls_preds)
        cls_auc = roc_auc_score(all_cls_labels, np.array(all_cls_probs)[:, 1])
        
        avg_cls_metrics = {'epoch': epoch + 1, 'accuracy': cls_acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'auc': cls_auc, 'loss': avg_val_loss}
        cls_metrics_history.append(avg_cls_metrics)

        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Dice: {avg_seg_metrics['dice']:.4f}, Cls Acc: {cls_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(seg_model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            torch.save(cls_model.state_dict(), os.path.join(run_dir, 'best_classifier.pth'))
            print(f"Saved new best models to {run_dir}")

    # Save metrics to CSV and plot
    seg_df = pd.DataFrame(seg_metrics_history)
    cls_df = pd.DataFrame(cls_metrics_history)
    seg_df.to_csv(os.path.join(run_dir, 'segmentation_metrics.csv'), index=False)
    cls_df.to_csv(os.path.join(run_dir, 'classification_metrics.csv'), index=False)
    
    plot_metrics(seg_df, run_dir)
    plot_metrics(cls_df, run_dir)
    print(f"Metrics and plots saved to {run_dir}")

if __name__ == '__main__':
    data_dir = 'data/processed_slices'
    train_multitask(data_dir)
