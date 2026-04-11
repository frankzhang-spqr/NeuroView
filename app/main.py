import os
import numpy as np
import nibabel as nib
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import base64
import io
from typing import List

from models.model import build_model
from models.classifier import build_classifier

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation_model = build_model(in_channels=4, n_class=1)
segmentation_model.load_state_dict(torch.load('best_model.pth', map_location=device))
segmentation_model.to(device)
segmentation_model.eval()

classifier_model = build_classifier(in_channels=4, num_classes=2)
classifier_model.load_state_dict(torch.load('best_classifier.pth', map_location=device))
classifier_model.to(device)
classifier_model.eval()

def preprocess_slice(slice_img):
    """Preprocesses a single slice for the models."""
    resized_channels = []
    for c in range(slice_img.shape[-1]):
        resized_channels.append(np.array(Image.fromarray(slice_img[..., c]).resize((240, 240))))
    
    slice_img = np.stack(resized_channels, axis=-1)
    
    num_channels = slice_img.shape[-1]
    if num_channels < 4:
        padding = np.zeros((240, 240, 4 - num_channels))
        slice_img = np.concatenate([slice_img, padding], axis=-1)

    # Normalize
    for c in range(slice_img.shape[-1]):
        channel = slice_img[..., c]
        min_val, max_val = np.min(channel), np.max(channel)
        if max_val > min_val:
            slice_img[..., c] = (channel - min_val) / (max_val - min_val)

    slice_tensor = torch.from_numpy(slice_img).float().permute(2, 0, 1).unsqueeze(0)
    return slice_tensor

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def get_bounding_box(mask):
    """Calculates the bounding box of a mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    scans_dir = "app/static/scans"
    os.makedirs(scans_dir, exist_ok=True)
    
    scan_url = None
    img_data = None

    if len(files) == 1:
        # Handle single-modality case (MEN-RT)
        file = files[0]
        file_path = os.path.join(scans_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        img_data = nib.load(file_path).get_fdata()
        if len(img_data.shape) == 3:
             img_data = np.expand_dims(img_data, axis=-1) # Add channel dimension
        scan_url = f"/static/scans/{file.filename}"

    elif len(files) == 4:
        # Handle multi-modality case (GLI)
        modalities = {}
        for file in files:
            file_path = os.path.join(scans_dir, file.filename)
            with open(file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)
            
            if "t1c" in file.filename:
                modalities["t1c"] = nib.load(file_path).get_fdata()
                scan_url = f"/static/scans/{file.filename}"
            elif "t1n" in file.filename:
                modalities["t1n"] = nib.load(file_path).get_fdata()
            elif "t2f" in file.filename:
                modalities["t2f"] = nib.load(file_path).get_fdata()
            elif "t2w" in file.filename:
                modalities["t2w"] = nib.load(file_path).get_fdata()

        if len(modalities) < 4:
            return {"error": "Please upload all 4 required GLI modalities (t1c, t1n, t2f, t2w)."}
        
        img_data = np.stack([modalities["t1c"], modalities["t1n"], modalities["t2f"], modalities["t2w"]], axis=-1)

    else:
        return {"error": "Please upload 1 file for MEN-RT or 4 files for GLI."}

    results = {
        "has_tumor": False,
        "tumor_type": "No Tumor Detected",
        "slices_with_tumor": [],
        "slice_data": [],
        "scan_url": scan_url
    }
    
    tumor_slices_indices = []
    num_slices = img_data.shape[2]
    
    for i in range(num_slices):
        slice_img = img_data[:, :, i, :]
        
        if np.sum(slice_img) == 0:
            continue
            
        slice_tensor = preprocess_slice(slice_img).to(device)
        
        with torch.no_grad():
            seg_output = segmentation_model(slice_tensor)
            mask = (seg_output > 0.5).cpu().numpy().squeeze()

        display_slice = slice_img[:, :, 0]
        display_slice = (display_slice - np.min(display_slice)) / (np.max(display_slice) - np.min(display_slice)) * 255
        img = Image.fromarray(display_slice.astype(np.uint8)).convert("L")
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        slice_info = {"slice_index": i, "image": img_str, "has_tumor": False}

        if np.sum(mask) > 0:
            results["has_tumor"] = True
            slice_info["has_tumor"] = True
            tumor_slices_indices.append(i)
            
            bbox = get_bounding_box(mask)
            if bbox:
                slice_info["bbox"] = bbox
        
        results["slice_data"].append(slice_info)

    if results["has_tumor"]:
        tumor_type_predictions = []
        for i in tumor_slices_indices:
            slice_img = img_data[:, :, i, :]
            
            slice_tensor = preprocess_slice(slice_img).to(device)
            with torch.no_grad():
                cls_output = classifier_model(slice_tensor)
                prediction = torch.max(cls_output, 1)[1].cpu().item()
                tumor_type_predictions.append(prediction)
        
        if tumor_type_predictions:
            majority_vote = max(set(tumor_type_predictions), key=tumor_type_predictions.count)
            results["tumor_type"] = "GLI" if majority_vote == 0 else "MEN-RT"
            results["slices_with_tumor"] = tumor_slices_indices

    return results