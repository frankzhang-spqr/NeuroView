# NeuroView MRI

NeuroView MRI is a Windows desktop application for brain MRI review, tumor segmentation, and tumor-type classification. The project combines an Electron shell, a FastAPI inference service, PyTorch models, preprocessing utilities, and training scripts into one repository.

At runtime, the user loads four NIfTI modalities:

- `T1C` for contrast-enhanced T1
- `T1N` for native T1
- `T2F` for T2 FLAIR
- `T2W` for T2 weighted

The backend normalizes the volumes, performs slice-wise tumor segmentation, classifies tumor-positive slices, aggregates the classification result across slices, and returns data used by the desktop UI for:

- 2D slice inspection across axial, coronal, and sagittal axes
- 3D point-cloud previews
- 3D raymarched volume rendering with tumor overlay

## What This Repository Contains

The codebase is split into five major layers:

1. Desktop shell and process orchestration with Electron
2. Web UI served by FastAPI and rendered in the Electron window
3. Inference backend in Python using FastAPI, PyTorch, nibabel, and PIL
4. Offline preprocessing and training scripts for segmentation and classification
5. Packaging for Windows distribution

## Repository Structure

Top-level layout:

- `main.js`: Electron main process, native menus, backend startup, lifecycle management
- `preload.js`: secure Electron preload bridge exposing a limited IPC surface
- `run.py`: packaged Python entry point for FastAPI
- `app/main.py`: FastAPI app, inference pipeline, cache, and HTTP endpoints
- `app/templates/index.html`: main UI document
- `app/static/dashboard.js`: dashboard behavior, fetch flows, 2D/3D rendering logic
- `app/static/style.css`: desktop UI styling
- `models/model.py`: segmentation architectures
- `models/classifier.py`: classification architectures
- `preprocessing/preprocess.py`: MRI-to-slice conversion pipeline
- `training/dataset_index.py`: metadata index and dataset summary utilities
- `training/train_segmentation.py`: segmentation training
- `training/train_classifier.py`: classification training
- `training/train_multitask.py`: older multitask training path for joint optimization
- `build_dist.ps1`: Windows packaging script
- `requirements.txt`: Python dependencies
- `package.json`: Electron dependencies and packaging configuration

Project data and artifacts:

- `BraTS/`: raw source data location used by preprocessing
- `data/processed_slices/`: generated slice dataset for training
- `training_runs/`: local training outputs, checkpoints, CSV metrics, plots
- `best_model.pth`: active segmentation weights used by inference
- `best_classifier.pth`: active classifier weights used by inference

## End-to-End Application Flow

The operational flow is:

1. Electron starts and launches the Python backend.
2. FastAPI loads the trained segmentation and classification models into memory.
3. The renderer UI loads from `http://127.0.0.1:8000/`.
4. The user supplies four MRI modalities through the desktop UI.
5. The `/predict` endpoint loads all four NIfTI volumes, normalizes them channel-by-channel, and stacks them into a single 4-channel volume.
6. Each axial slice is resized to `240 x 240`, passed through the segmentation model, and written into a binary tumor mask volume.
7. Only slices with predicted tumor are passed to the classification model.
8. Slice-level class predictions are combined with majority voting into a final tumor type label:
   `GLI` for class `0`, `MEN` for class `1`
9. The backend caches the normalized volume and predicted mask under a generated `scan_id`.
10. The UI uses follow-up endpoints to request slice previews, tumor slice lists, point clouds, and binary volume textures.

## Desktop Application Architecture

### Electron Main Process

The desktop shell lives in `main.js`.

Responsibilities:

- creates the `BrowserWindow`
- configures the app icon, minimum size, and background color
- launches the backend process
- polls FastAPI until the backend is ready
- builds the native application menu
- terminates the backend cleanly on app exit

Behavior differs by environment:

- development mode:
  Electron spawns `python -m uvicorn app.main:app --port 8000`
- packaged mode:
  Electron spawns the bundled `neuroview_backend.exe`

Important implementation details:

- `nodeIntegration` is disabled
- `contextIsolation` is enabled
- the renderer only gets access to an explicit preload bridge
- the app waits for `GET /` to return `200` before opening the window

### Preload Bridge

The preload layer lives in `preload.js`.

It exposes menu-driven callbacks into the renderer for:

- open files
- reset workspace
- export results
- copy slice
- preferences
- toggle sidebar
- toggle status bar
- run analysis
- change axis
- toggle tumor overlay
- open docs
- show keyboard shortcuts

This keeps the renderer decoupled from raw Electron APIs and is the project’s main security boundary on the desktop side.

## Backend Architecture

The FastAPI app lives in `app/main.py`.

### Startup

At import time, the backend:

- selects `cuda` when available, otherwise `cpu`
- loads `best_model.pth`
- infers whether the segmentation checkpoint is the `legacy` or `enhanced` architecture
- builds the segmentation model
- loads `best_classifier.pth`
- infers whether the classifier checkpoint is `legacy` or `enhanced`
- builds the classifier
- switches both models to `eval()` mode

The backend also:

- mounts `app/static`
- serves `index.html` through Jinja2 templates
- maintains an in-memory `analysis_cache` keyed by `scan_id`

### Inference Pipeline

The primary inference endpoint is `POST /predict`.

Input contract:

- `t1c_file`
- `t1n_file`
- `t2f_file`
- `t2w_file`

Processing stages:

1. Uploaded files are written into `app/static/scans/`.
2. Each modality is loaded with nibabel using `get_fdata()`.
3. Modalities are stacked into a shape roughly equivalent to `(H, W, D, 4)`.
4. Each channel is min-max normalized independently to `[0, 1]`.
5. Each axial slice is resized to `240 x 240` before inference.
6. The segmentation model predicts a tumor mask for every non-empty slice.
7. Predicted masks are thresholded at `0.5`.
8. Slice masks are resized back to original in-plane resolution and written into a 3D `mask_volume`.
9. Tumor-positive slices are collected for downstream classification.
10. The classifier predicts class labels only on slices flagged by segmentation.
11. Majority vote across positive slices becomes the final tumor type.

Returned fields include:

- `has_tumor`
- `tumor_type`
- `slices_with_tumor`
- `scan_id`
- `slice_counts`
- `tumor_slices_by_axis`
- `volume_points`

### Cached Follow-Up Endpoints

The backend keeps normalized data and predicted masks in memory per scan session.

#### `GET /slice-preview/{scan_id}`

Returns one preview slice for a requested axis and index.

Each response includes:

- one PNG preview per modality
- whether that slice contains tumor
- a normalized bounding box if tumor is present

#### `GET /slice-stack/{scan_id}`

Returns compact previews for every tumor-positive slice on a chosen axis.

#### `GET /volume-preview/{scan_id}`

Returns sparse point clouds for:

- a selected modality volume
- the tumor mask

These points drive the lightweight 3D point-cloud mode in the UI.

#### `GET /volume-binary/{scan_id}`

Returns flattened `uint8` binary data for a modality or the mask. This is used by the browser-side `DataTexture3D` path for volume rendering.

### Utility Behavior in the Backend

The backend contains several helper functions worth understanding:

- `preprocess_slice`: resizes each modality channel independently to `240 x 240`
- `get_bounding_box`: extracts a simple rectangular tumor box from a 2D mask
- `to_display_orientation`: rotates slices for display consistency
- `encode_png`: generates base64 PNG payloads for the dashboard
- `sample_point_cloud`: sparsifies volume data into a manageable number of 3D points
- `get_tumor_slices_by_axis`: finds tumor-positive slices along each orthogonal axis

## Frontend Architecture

The UI is rendered by `index.html`, styled by `style.css`, and driven by `dashboard.js`.

### UI Layout

The desktop UI is organized into:

- top toolbar
- left sidebar
- 3D volume panel
- slice viewer panel
- bottom status bar
- keyboard shortcuts modal

### Core User Features

The current interface supports:

- upload of all four modalities
- run analysis
- reset workspace
- axis switching between axial, coronal, sagittal
- switching 3D modality source
- toggling between point-cloud and volume rendering
- toggling sidebar visibility
- toggling status bar visibility
- viewing tumor-positive slices
- stepping through slices with buttons and slider

### 2D Slice Rendering

The backend supplies rendered PNG previews per modality. The frontend overlays:

- tumor status
- modality labels
- optional bounding boxes
- crosshair guides

The slice grid always shows the current slice across all four modalities side by side.

### 3D Rendering Modes

The UI supports two different 3D strategies.

#### Point-Cloud Mode

This is the lighter path and uses `volume_points`.

It renders:

- gray points for brain structure
- red points for tumor voxels
- grid and axes helpers
- decorative bounding boxes

This path is less precise but faster and easier to render on modest hardware.

#### Volume Mode

This path fetches raw `uint8` voxel data via `/volume-binary`.

It then:

- builds `THREE.DataTexture3D` textures for the selected modality and the mask
- uses a shader-based raymarching material
- shades brain tissue in grayscale
- shades tumor regions bright red
- rotates the volume slowly for presentation

The renderer infers dimensions from the backend-provided `slice_counts`.

## Model Architecture

The repository contains both `legacy` and `enhanced` model definitions. The backend automatically infers which variant to instantiate by inspecting checkpoint keys.

### Segmentation Models

Defined in `models/model.py`.

#### Legacy Segmentation Model

The legacy model is a straightforward 2D U-Net:

- 4 input channels
- encoder-decoder structure
- max pooling downsampling
- transpose convolution upsampling
- skip connections between encoder and decoder blocks
- single output channel for binary segmentation

#### Enhanced Segmentation Model

The enhanced model is the main current architecture.

Key design choices:

- residual blocks
- squeeze-excitation channel attention
- multi-stage encoder with strided residual downsampling
- residual bridge
- decoder blocks with learned upsampling and skip fusion
- `SiLU` activations
- 1-channel segmentation head

Default channel widths:

- `32`
- `64`
- `128`
- `256`
- `384`

The segmentation model is built with:

```python
build_model(in_channels=4, n_class=1, variant="enhanced")
```

### Classification Models

Defined in `models/classifier.py`.

#### Legacy Classification Model

The legacy classifier is a simple CNN with:

- stacked convolutions
- max pooling
- flattened dense head
- dropout before the final classification layer

#### Enhanced Classification Model

The enhanced classifier is the main current architecture.

Key design choices:

- convolutional stem
- residual classifier blocks
- squeeze-excitation attention
- adaptive average pooling
- compact MLP head with dropout
- 2 output classes

Default widths:

- `32`
- `64`
- `128`
- `256`

The classifier is built with:

```python
build_classifier(in_channels=4, num_classes=2, variant="enhanced")
```

## Data Preparation Pipeline

The preprocessing script is `preprocessing/preprocess.py`.

Its purpose is to convert raw 3D MRI studies into slice-level numpy training data.

### Expected Raw Data Assumptions

The script expects patient folders containing `.nii.gz` studies. It searches for:

- `-t1c.nii.gz`
- `-t1n.nii.gz`
- `-t2f.nii.gz`
- `-t2w.nii.gz`
- segmentation labels with either `-seg.nii.gz` or `-GTV.nii.gz`

Tumor type is inferred from the folder path:

- folders containing `GLI` are labeled glioma
- folders containing `MEN` are labeled meningioma

### Preprocessing Steps

For each patient:

1. load all four modalities
2. load the segmentation volume
3. min-max normalize each modality independently
4. walk through axial slices
5. skip completely empty slices
6. resize each modality slice to `240 x 240`
7. resize the segmentation mask with nearest-neighbor interpolation
8. binarize the mask
9. write numpy artifacts to disk

Generated files for each slice:

- `slice_<id>.npy`: float32 image tensor shaped like `(240, 240, 4)`
- `label_<id>.npy`: uint8 binary mask shaped like `(240, 240)`
- `type_<id>.npy`: class label array containing `0` for `GLI` or `1` for `MEN`
- `group_<id>.npy`: patient identifier used to prevent leakage across train/validation splits

Output directory:

- `data/processed_slices/`

## Dataset Indexing

The training scripts rely on `training/dataset_index.py`.

This utility builds `training_index.json` so the code does not have to repeatedly rescan every slice file.

Each indexed entry stores:

- `slice_id`
- `group`
- `type_label`
- `has_tumor`
- `tumor_pixels`
- `total_pixels`

The helper also produces split summaries such as:

- total tumor pixels
- total background pixels
- count of positive slices
- class counts for `GLI` and `MEN`

## Training Pipelines

There are three training scripts in the repository.

### 1. Segmentation Training

Script:

- `training/train_segmentation.py`

Goal:

- learn a binary tumor mask from 4-channel 2D slices

Dataset split strategy:

- `GroupShuffleSplit`
- `20%` validation
- grouping by patient ID to avoid patient leakage

Training setup:

- model: enhanced segmentation model
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- batch size default: `16`
- epochs default: `25`
- learning rate default: `1e-4`

Class imbalance handling:

- positive slices are upweighted in the sampler
- `pos_weight` is computed from tumor versus background pixels

Loss:

- `0.4 * BCEWithLogits + 0.6 * Dice loss`

Validation metrics:

- loss
- Dice
- IoU
- precision
- recall

Artifacts:

- `best_model.pth`
- `best_checkpoint.pth`
- `last_checkpoint.pth`
- `segmentation_metrics.csv`

The best model is written both into the run directory and to the repo root as `best_model.pth`.

### 2. Classification Training

Script:

- `training/train_classifier.py`

Goal:

- classify slices into `GLI` or `MEN`

Dataset split strategy:

- `GroupShuffleSplit`
- `20%` validation
- patient-level grouping

Training setup:

- model: enhanced classifier
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- criterion: `CrossEntropyLoss(label_smoothing=0.05)`
- batch size default: `24`
- epochs default: `18`
- learning rate default: `3e-4`

Class imbalance handling:

- `WeightedRandomSampler`
- inverse-frequency weights by class label

Validation metrics:

- loss
- accuracy
- precision
- recall
- F1
- AUC

Artifacts:

- `best_classifier.pth`
- `best_checkpoint.pth`
- `last_checkpoint.pth`
- `classification_metrics.csv`

The best classifier is also copied to the repo root as `best_classifier.pth`.

### 3. Multitask Training

Script:

- `training/train_multitask.py`

Goal:

- jointly optimize segmentation and classification in one training loop

This path appears to be an older or experimental workflow. It still trains both models together, saves metrics, and generates plots, but the main inference flow currently loads separate root-level segmentation and classifier checkpoints.

Training traits:

- combined segmentation loss plus classification loss
- shared optimizer across both models
- weighted sampler based on tumor presence and class
- matplotlib plot generation

Artifacts may include:

- `best_model.pth`
- `best_classifier.pth`
- `segmentation_metrics.csv`
- `classification_metrics.csv`
- PNG metric plots

## Running Training

Example commands:

### Preprocess data

```powershell
python preprocessing\preprocess.py
```

### Train segmentation

```powershell
python training\train_segmentation.py --data-dir data\processed_slices --epochs 25 --batch-size 16 --learning-rate 1e-4
```

### Resume segmentation

```powershell
python training\train_segmentation.py --resume training_runs\seg_YYYY-MM-DD_HH-MM-SS\last_checkpoint.pth
```

### Train classification

```powershell
python training\train_classifier.py --data-dir data\processed_slices --epochs 18 --batch-size 24 --learning-rate 3e-4
```

### Resume classification

```powershell
python training\train_classifier.py --resume training_runs\cls_YYYY-MM-DD_HH-MM-SS\last_checkpoint.pth
```

### Run multitask training

```powershell
python training\train_multitask.py
```

## Running the Desktop App in Development

### Prerequisites

- Windows 10 or 11
- Python installed and available on `PATH`
- Node.js and npm
- Python dependencies installed
- the trained root-level model files present:
  `best_model.pth` and `best_classifier.pth`

### Install Python dependencies

```powershell
pip install -r requirements.txt
pip install pyinstaller
```

### Install Node dependencies

```powershell
npm install
```

### Start the desktop app

```powershell
npm start
```

This launches Electron, which then spawns the local FastAPI backend automatically.

## Packaging and Distribution

The Windows packaging pipeline is driven by `build_dist.ps1`.

### Packaging Stages

1. remove previous build outputs
2. run PyInstaller to package the FastAPI backend into `neuroview_backend`
3. copy the backend into `build/app_bundle/neuroview_backend`
4. run `npm run pack` with electron-builder
5. compress `dist-electron/win-unpacked` into `data.7z`
6. copy setup helper files into `dist-electron`

The intent is to create a Windows-friendly packaged distribution containing:

- the Electron app
- the packaged backend
- the trained model weights
- setup helper scripts

### Electron Builder Configuration

Configured in `package.json`.

Important settings:

- app id: `com.neuroview.mri`
- product name: `NeuroView MRI`
- Windows target: unpacked directory build
- output directory: `dist-electron`

Extra resources bundled into the app:

- packaged backend
- `app/`
- `best_model.pth`
- `best_classifier.pth`

### Manual Release Flow

Releases are currently prepared manually.

Recommended flow:

1. run `build_dist.ps1`
2. verify the contents of `dist-electron`
3. create `Installer.zip` from the release output
4. upload `Installer.zip` to the GitHub Release by hand

`Installer.zip` is intended to be a release artifact, not a source-controlled file in the repository.

## API Summary

Routes currently implemented:

- `GET /`
- `POST /predict`
- `GET /slice-preview/{scan_id}`
- `GET /slice-stack/{scan_id}`
- `GET /volume-preview/{scan_id}`
- `GET /volume-binary/{scan_id}`

## Current Assumptions and Limitations

This section is important because it reflects the actual code, not an idealized future design.

### Input Assumptions

- the app expects all four modalities
- segmentation and classification are both slice-wise 2D models
- the backend currently iterates through axial slices for inference
- file naming in preprocessing depends on substring matching

### Runtime Limitations

- `analysis_cache` is in memory only and has no persistence or eviction policy
- uploaded scans are written under `app/static/scans/`
- no user authentication or session isolation exists
- inference depends on root-level weight files being present

### UI Limitations

Several menu items are placeholders and currently show alerts instead of full implementations:

- export results
- copy slice
- preferences
- overlay toggle from menu
- file-to-modality assignment from native open dialog

### Training Limitations

- there is no dedicated automated evaluation script for held-out test sets in this repo
- there is no centralized experiment configuration system yet
- metrics and checkpoints are file-based and local
- training outputs in `training_runs/` are local artifacts and should not be relied on as the project’s canonical release mechanism

## Dependencies

### Python

Declared in `requirements.txt`:

- nibabel
- numpy
- torch
- torchvision
- scikit-learn
- Pillow
- tqdm
- matplotlib
- pandas
- fastapi
- uvicorn[standard]
- `pyhton-multipart` as currently written in the file

Note:

- the package name in `requirements.txt` is spelled `pyhton-multipart` in the current repository and may need correction if installs fail

### Node

Declared in `package.json`:

- electron
- electron-builder

The frontend also loads:

- Three.js from CDN
- OrbitControls from CDN

## Suggested Development Workflow

For day-to-day work, the practical loop is:

1. preprocess or refresh the slice dataset as needed
2. train or update segmentation and classification checkpoints
3. copy the winning checkpoints to `best_model.pth` and `best_classifier.pth`
4. run `npm start`
5. validate inference on representative 4-modality studies
6. package with `build_dist.ps1` when preparing a Windows release
7. upload `Installer.zip` to the release manually

## File-by-File Role Summary

Quick reference:

- `main.js`: desktop bootstrap and native shell
- `preload.js`: renderer-safe event bridge
- `run.py`: packaged backend runner
- `app/main.py`: service layer, model loading, inference, preview endpoints
- `app/templates/index.html`: DOM structure
- `app/static/dashboard.js`: UI state, fetch calls, 2D and 3D rendering
- `app/static/style.css`: visual design and responsive behavior
- `models/model.py`: segmentation model definitions
- `models/classifier.py`: classifier model definitions
- `preprocessing/preprocess.py`: raw MRI conversion into supervised slice data
- `training/dataset_index.py`: cached metadata index
- `training/train_segmentation.py`: segmentation training loop
- `training/train_classifier.py`: classification training loop
- `training/train_multitask.py`: multitask experiment loop
- `build_dist.ps1`: packaging script

## Summary

NeuroView MRI is a hybrid medical-imaging desktop application built around 4-channel MRI slice inference. It uses:

- Electron for the desktop shell
- FastAPI for the local inference service
- PyTorch for segmentation and classification
- nibabel for MRI volume loading
- Three.js for interactive 3D visualization

The central design choice in this project is a slice-based deep-learning pipeline paired with a desktop visualization layer that makes the model outputs explorable in both 2D and 3D.
