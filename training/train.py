"""
Contains functions for training the PyTorch model, with advanced metrics tracking.
"""
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

class TumorDataset(Dataset):
    def __init__(self, file_list, processed_path):
        self.file_list = file_list
        self.processed_path = processed_path

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        slice_path = os.path.join(self.processed_path, filename)
        label_path = os.path.join(self.processed_path, filename.replace('slice_', 'label_'))
        
        # Load image in (H, W, C) format
        slice_data = np.load(slice_path)
        label_data = np.load(label_path)
        
        # Convert to tensor and permute dimensions to (C, H, W) for PyTorch
        slice_tensor = torch.from_numpy(slice_data).float().permute(2, 0, 1)
        # Add channel dimension for the label mask
        label_tensor = torch.from_numpy(label_data).float().unsqueeze(0)
        
        return slice_tensor, label_tensor

def calculate_metrics(labels, preds):
    """Calculates a comprehensive set of metrics."""
    # Handle cases where a batch contains only one class
    if len(np.unique(labels)) < 2:
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': 0, 'recall': 0, 'f1_score': 0, 'specificity': 0, 'auc': 0
        }

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5 # Default value if only one class is present

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc': auc
    }

def plot_metrics(history, run_dir):
    """Plots and saves graphs for all metrics."""
    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, f'{metric}_plot.png'))
        plt.close()

def train_model(model, data_path, run_dir, epochs=5, batch_size=32):
    """
    Train the PyTorch U-Net model with an 80/20 train-validation split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_files = [f for f in os.listdir(data_path) if f.startswith('slice_')]
    
    # Perform an 80/20 split of the data
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = TumorDataset(train_files, data_path)
    val_dataset = TumorDataset(val_files, data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1_score': [], 'val_f1_score': [],
        'train_specificity': [], 'val_specificity': [],
        'train_auc': [], 'val_auc': []
    }

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_metrics_agg = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc']}
        num_train_batches = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Calculate metrics for this batch
            batch_metrics = calculate_metrics(labels.cpu().numpy().flatten(), (outputs > 0.5).cpu().numpy().flatten())
            for k in train_metrics_agg:
                train_metrics_agg[k] += batch_metrics[k]
            num_train_batches += 1

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_metrics = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc']}
        
        if len(val_loader) > 0:
            val_metrics_agg = {k: 0.0 for k in val_metrics}
            num_val_batches = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    batch_metrics = calculate_metrics(labels.cpu().numpy().flatten(), (outputs > 0.5).cpu().numpy().flatten())
                    for k in val_metrics_agg:
                        val_metrics_agg[k] += batch_metrics[k]
                    num_val_batches += 1
            
            val_loss /= len(val_loader.dataset)
            val_metrics = {k: v / num_val_batches for k, v in val_metrics_agg.items()}
        else:
            print(f"Warning: Validation set is empty. Skipping validation for epoch {epoch+1}.")
            val_loss = 0.0 # Assign a default value

        # --- Metrics Calculation & History Update ---
        train_metrics = {k: v / num_train_batches for k, v in train_metrics_agg.items()}

        history['train_loss'].append(train_loss / len(train_loader.dataset))
        history['val_loss'].append(val_loss)
        for metric in train_metrics:
            history[f'train_{metric}'].append(train_metrics[metric])
            history[f'val_{metric}'].append(val_metrics[metric])

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train -> Loss: {history['train_loss'][-1]:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"Val   -> Loss: {history['val_loss'][-1]:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")
        
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))

    # --- Save Metrics and Plots ---
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
    plot_metrics(history, run_dir)
    print(f"Metrics and plots saved to {run_dir}")