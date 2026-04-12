# 3D Brain Tumor Segmentation and Classification

This project provides a complete pipeline for identifying and classifying brain tumors from 3D MRI scans. It includes a web-based application for easy interaction, allowing users to upload MRI scans and visualize the results in both 2D and 3D.

## Features

- **Tumor Segmentation**: Identifies the presence and location of a tumor within the brain.
- **Tumor Classification**: Classifies the detected tumor into one of two types: `GLI` (Glioma) or `MEN` (Meningioma).
- **Interactive Web UI**: A user-friendly interface for uploading scans and viewing results.
- **2D Slice Viewer**: Displays all four MRI modalities (`t1c`, `t1n`, `t2f`, `t2w`) for each slice, with a bounding box drawn around the tumor on affected slices.
- **3D Model Viewer**: Renders an interactive 3D model of the brain with the tumor highlighted, providing spatial context.

## Project Structure

```
.
├── app/                  # Contains the FastAPI web application
│   ├── static/           # CSS and JavaScript files
│   ├── templates/        # HTML templates
│   └── main.py           # Backend logic for the web app
├── data/                 # Data directory
│   └── processed_slices/ # Stores preprocessed 2D slices for training
├── models/               # Model definitions
│   ├── model.py          # U-Net segmentation model
│   └── classifier.py     # CNN classification model
├── preprocessing/        # Data preprocessing scripts
│   └── preprocess.py     # Script to convert 3D NIfTI scans to 2D slices
├── training/             # Model training scripts
│   └── train_multitask.py # Unified script for multi-task training
├── best_model.pth        # Saved weights for the best segmentation model
├── best_classifier.pth   # Saved weights for the best classification model
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The model is trained on the BraTS 2024 dataset. The data should be organized as follows, with each patient folder containing four NIfTI files corresponding to the four MRI modalities:

```
BraTS/
├── GLI/
│   └── ... (patient folders)
└── MEN/
    └── ... (patient folders)
```

To preprocess the data for training, run the following command. This will process the raw NIfTI files and save them as 2D `.npy` slices in the `data/processed_slices` directory.

```bash
python preprocessing/preprocess.py
```

## Model Training

The project uses a multi-task learning approach to train the segmentation and classification models simultaneously. To start the training process, run the unified training script:

```bash
python training/train_multitask.py
```

This script will:
- Load the preprocessed data.
- Train both the U-Net segmentation model and the CNN classification model.
- Validate the models and save the best-performing weights to `best_model.pth` and `best_classifier.pth`.

## Running the Application

Once the models are trained, you can launch the web application:

```bash
uvicorn app.main:app --reload
```

Navigate to `http://127.0.0.1:8000` in your web browser.

### How to Use the App

1.  The interface provides four labeled upload slots, one for each required MRI modality (`t1c`, `t1n`, `t2f`, `t2w`).
2.  Select the corresponding NIfTI file for each slot.
3.  Click the "Analyze Scan" button.
4.  The application will process the scans and display the results, including the tumor type, a 3D model, and a 2D slice-by-slice view with bounding boxes.