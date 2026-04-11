"""
Main script to run the tumor detection pipeline using PyTorch.
"""
import os
import torch
from datetime import datetime
from preprocessing.preprocess import get_patient_folders, process_and_save_slices
from models.model import build_model
from models.classifier import build_classifier
from training.train import train_model
from training.train_classifier import train_classifier

def main():
    """
    Main function to run the preprocessing, model building, and training pipeline.
    """
    # Check for GPU
    if torch.cuda.is_available():
        print("GPU found, training will run on GPU.")
    else:
        print("No GPU found, training will run on CPU.")

    # Step 1: Data Preprocessing
    print("Starting data preprocessing...")
    # Define paths for all available labeled data
    train_gli_path = 'BraTS/GLI/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2'
    train_men_rt_path = 'BraTS/MEN-RT/BraTS2024-MEN-RT-TrainingData/BraTS-MEN-RT-Train-v2'
    output_path = 'data/processed_slices'

    # Process all available labeled data into a single directory
    print("\nProcessing All Labeled Data...")
    all_folders = get_patient_folders(train_gli_path) + get_patient_folders(train_men_rt_path)
    process_and_save_slices(all_folders, output_path)
    print(f"Data preprocessing complete. All slices saved to {output_path}")

    # Create a timestamped directory for this training run
    run_dir = os.path.join('training_runs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_dir, exist_ok=True)

    # Step 2: Segmentation Model Training
    print("\nBuilding and training the segmentation model...")
    # Build model to accept 4-channel input (t1c, t1n, t2f, t2w for GLI, padded for MEN-RT)
    segmentation_model = build_model(in_channels=4)
    train_model(segmentation_model, output_path, run_dir)
    print("Segmentation model training complete.")

    # Step 3: Classification Model Training
    print("\nBuilding and training the classification model...")
    # Build classifier to accept 4-channel input
    classifier_model = build_classifier(in_channels=4)
    train_classifier(classifier_model, output_path, run_dir)
    print("Classification model training complete.")

if __name__ == '__main__':
    main()