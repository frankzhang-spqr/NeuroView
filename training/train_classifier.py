"""
Contains functions for training the tumor type classifier, with advanced metrics.
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

class ClassifierDataset(Dataset):
    def __init__(self, file_list, processed_path):
        self.file_list = file_list
        self.processed_path = processed_path

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        slice_path = os.path.join(self.processed_path, filename)
        type_path = os.path.join(self.processed_path, filename.replace('slice_', 'type_'))
        
        # Load image in (H, W, C) format
        slice_data = np.load(slice_path)
        type_data = np.load(type_path)
        
        # Convert to tensor and permute dimensions to (C, H, W) for PyTorch
        slice_tensor = torch.from_numpy(slice_data).float().permute(2, 0, 1)
        type_tensor = torch.from_numpy(type_data).long().squeeze()
        
        return slice_tensor, type_tensor

def calculate_classification_metrics(labels, preds, probs):
    """Calculates a comprehensive set of classification metrics."""
    # Handle cases where a batch contains only one class
    if len(np.unique(labels)) < 2:
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': 0, 'recall': 0, 'f1_score': 0, 'auc': 0
        }
        
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    try:
        # For multi-class, use one-vs-rest and average
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc = 0.5

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'auc': auc}

def plot_classifier_metrics(history, run_dir):
    """Plots and saves graphs for all classifier metrics."""
    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.title(f'Classifier {metric.capitalize()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, f'classifier_{metric}_plot.png'))
        plt.close()

def train_classifier(model, data_path, run_dir, epochs=5, batch_size=32):
    """
    Train the PyTorch tumor type classifier with an 80/20 train-validation split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # We only train the classifier on slices that actually contain a tumor
    all_files = [f for f in os.listdir(data_path) if f.startswith('slice_')]
    tumor_slices = [f for f in all_files if np.sum(np.load(os.path.join(data_path, f.replace('slice_', 'label_')))) > 0]

    # Perform an 80/20 split of the tumor-containing slices
    train_files, val_files = train_test_split(tumor_slices, test_size=0.2, random_state=42)

    train_dataset = ClassifierDataset(train_files, data_path)
    val_dataset = ClassifierDataset(val_files, data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'train_precision', 'val_precision', 'train_recall', 'val_recall', 'train_f1_score', 'val_f1_score', 'train_auc', 'val_auc']}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_labels, train_preds, train_probs = 0.0, [], [], []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Classifier Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            train_probs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_metrics = {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']}

        if len(val_loader) > 0:
            val_labels, val_preds, val_probs = [], [], []
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Classifier Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                    val_probs.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())
            
            val_loss /= len(val_loader.dataset)
            val_metrics = calculate_classification_metrics(val_labels, val_preds, val_probs)
        else:
            print(f"Warning: Classifier validation set is empty. Skipping validation for epoch {epoch+1}.")
            val_loss = 0.0 # Assign a default value

        # --- Metrics Calculation & History Update ---
        train_metrics = calculate_classification_metrics(train_labels, train_preds, train_probs)

        history['train_loss'].append(train_loss / len(train_loader.dataset))
        history['val_loss'].append(val_loss)
        for metric in train_metrics:
            history[f'train_{metric}'].append(train_metrics[metric])
            history[f'val_{metric}'].append(val_metrics[metric])

        print(f"\nEpoch {epoch+1}/{epochs} [Classifier]")
        print(f"Train -> Loss: {history['train_loss'][-1]:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"Val   -> Loss: {history['val_loss'][-1]:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")
        
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_classifier.pth'))

    # --- Save Metrics and Plots ---
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(run_dir, 'classifier_metrics.csv'), index=False)
    plot_classifier_metrics(history, run_dir)
    print(f"Classifier metrics and plots saved to {run_dir}")