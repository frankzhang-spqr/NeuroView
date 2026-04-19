import os
import uuid
import numpy as np
import nibabel as nib
import torch
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import base64
import io
from typing import Dict

from models.model import build_model, infer_segmentation_variant
from models.classifier import build_classifier, infer_classifier_variant

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentation_state = torch.load("best_model.pth", map_location=device)
segmentation_variant = infer_segmentation_variant(segmentation_state)
segmentation_model = build_model(in_channels=4, n_class=1, variant=segmentation_variant)
segmentation_model.load_state_dict(segmentation_state)
segmentation_model.to(device)
segmentation_model.eval()

classifier_state = torch.load("best_classifier.pth", map_location=device)
classifier_variant = infer_classifier_variant(classifier_state)
classifier_model = build_classifier(in_channels=4, num_classes=2, variant=classifier_variant)
classifier_model.load_state_dict(classifier_state)
classifier_model.to(device)
classifier_model.eval()

analysis_cache: Dict[str, dict] = {}

def preprocess_slice(slice_img):
    resized_channels = []
    for c in range(slice_img.shape[-1]):
        resized_channels.append(np.array(Image.fromarray(slice_img[..., c]).resize((240, 240))))
    
    slice_img = np.stack(resized_channels, axis=-1)

    return torch.from_numpy(slice_img).float().permute(2, 0, 1).unsqueeze(0)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]


def normalize_to_uint8(slice_img):
    slice_img = slice_img.astype(np.float32)
    min_val, max_val = np.min(slice_img), np.max(slice_img)
    if max_val > min_val:
        slice_img = (slice_img - min_val) / (max_val - min_val)
    else:
        slice_img = np.zeros_like(slice_img, dtype=np.float32)
    return (slice_img * 255).astype(np.uint8)


def to_display_orientation(slice_img):
    return np.rot90(slice_img)


def encode_png(slice_img, size=240, is_mask=False):
    pil_mode = "L"
    image = Image.fromarray(slice_img.astype(np.uint8), mode=pil_mode)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    image = image.resize((size, size), resample=resample)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def resize_mask_to_shape(mask, target_shape):
    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    resized = mask_image.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST)
    return (np.array(resized) > 0).astype(np.uint8)


def extract_axis_slice(volume, axis, index):
    if axis == "axial":
        return volume[:, :, index]
    if axis == "coronal":
        return volume[:, index, :]
    if axis == "sagittal":
        return volume[index, :, :]
    raise ValueError(f"Unsupported axis: {axis}")


def build_slice_preview(img_data, mask_volume, axis, index, image_size=240):
    ordered_modalities = ["t1c", "t1n", "t2f", "t2w"]
    previews = []

    mask_slice = to_display_orientation(extract_axis_slice(mask_volume, axis, index))
    
    raw_bbox = get_bounding_box(mask_slice)
    if raw_bbox:
        cmin, rmin, w, h = raw_bbox
        h_total, w_total = mask_slice.shape
        bbox = [
            round(cmin / w_total * 100, 2),
            round(rmin / h_total * 100, 2),
            round(w / w_total * 100, 2),
            round(h / h_total * 100, 2)
        ]
    else:
        bbox = None
        
    has_tumor = bool(np.any(mask_slice))

    for c, modality in enumerate(ordered_modalities):
        image_slice = extract_axis_slice(img_data[:, :, :, c], axis, index)
        image_slice = to_display_orientation(image_slice)
        image_slice = normalize_to_uint8(image_slice)
        previews.append({
            "modality": modality,
            "image": encode_png(image_slice, size=image_size),
            "bbox": bbox,
            "has_tumor": has_tumor,
        })

    return {
        "axis": axis,
        "slice_index": int(index),
        "has_tumor": has_tumor,
        "bbox": bbox,
        "slices": previews,
    }


def sample_point_cloud(volume, *, mask=None, max_points=14000, min_points=1000):
    source = np.array(volume, dtype=np.float32)
    if mask is not None:
        coords = np.argwhere(mask > 0)
    else:
        non_zero = source[source > 0]
        if non_zero.size == 0:
            return []
        threshold = float(np.percentile(non_zero, 65))
        coords = np.argwhere(source >= threshold)

    if coords.size == 0:
        return []

    if len(coords) > max_points:
        step = max(1, len(coords) // max_points)
        coords = coords[::step]
    elif len(coords) < min_points and mask is None:
        non_zero_coords = np.argwhere(source > 0)
        if len(non_zero_coords) > 0:
            step = max(1, len(non_zero_coords) // min_points)
            coords = non_zero_coords[::step]

    dims = np.array(volume.shape, dtype=np.float32)
    normalized = (coords.astype(np.float32) / np.maximum(dims - 1, 1)) - 0.5
    normalized[:, 1] *= -1
    return normalized.tolist()


def get_tumor_slices_by_axis(mask_volume):
    return {
        "axial": [int(i) for i in np.where(np.any(mask_volume, axis=(0, 1)))[0].tolist()],
        "coronal": [int(i) for i in np.where(np.any(mask_volume, axis=(0, 2)))[0].tolist()],
        "sagittal": [int(i) for i in np.where(np.any(mask_volume, axis=(1, 2)))[0].tolist()],
    }

@app.post("/predict")
async def predict(
    t1c_file: UploadFile = File(...),
    t1n_file: UploadFile = File(...),
    t2f_file: UploadFile = File(...),
    t2w_file: UploadFile = File(...)
):
    scans_dir = "app/static/scans"
    os.makedirs(scans_dir, exist_ok=True)
    
    uploaded_files = {
        "t1c": t1c_file, "t1n": t1n_file,
        "t2f": t2f_file, "t2w": t2w_file
    }
    
    modalities = {}
    scan_url = None

    for name, file in uploaded_files.items():
        file_path = os.path.join(scans_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        modalities[name] = nib.load(file_path).get_fdata()
        if name == "t1c":
            scan_url = f"/static/scans/{file.filename}"

    ordered_modalities = ["t1c", "t1n", "t2f", "t2w"]
    img_data = np.stack([modalities[mod] for mod in ordered_modalities], axis=-1)

    for c in range(img_data.shape[-1]):
        channel = img_data[..., c]
        min_val, max_val = np.min(channel), np.max(channel)
        if max_val > min_val:
            img_data[..., c] = (channel - min_val) / (max_val - min_val)

    results = {
        "has_tumor": False,
        "tumor_type": "No Tumor Detected",
        "slices_with_tumor": [],
        "scan_url": scan_url,
    }
    
    tumor_slices_indices = []
    num_slices = img_data.shape[2]
    mask_volume = np.zeros(img_data.shape[:3], dtype=np.uint8)
    
    for i in range(num_slices):
        slice_img_4_channel = img_data[:, :, i, :]
        
        if np.sum(slice_img_4_channel) == 0:
            continue
            
        slice_tensor = preprocess_slice(slice_img_4_channel).to(device)
        
        with torch.no_grad():
            seg_output = segmentation_model(slice_tensor)
            mask = (seg_output > 0.5).cpu().numpy().squeeze()
            mask = resize_mask_to_shape(mask, slice_img_4_channel.shape[:2])

        has_tumor_in_slice = bool(np.sum(mask) > 0)
        if has_tumor_in_slice:
            results["has_tumor"] = True
            tumor_slices_indices.append(i)
        mask_volume[:, :, i] = mask.astype(np.uint8)

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
            results["tumor_type"] = "GLI" if majority_vote == 0 else "MEN"
            results["slices_with_tumor"] = tumor_slices_indices

    scan_id = uuid.uuid4().hex
    tumor_slices_by_axis = get_tumor_slices_by_axis(mask_volume)
    analysis_cache[scan_id] = {
        "img_data": img_data,
        "mask_volume": mask_volume,
    }

    results["has_tumor"] = bool(results["has_tumor"])
    results["scan_id"] = scan_id
    results["slice_counts"] = {
        "axial": int(img_data.shape[2]),
        "coronal": int(img_data.shape[1]),
        "sagittal": int(img_data.shape[0]),
    }
    results["tumor_slices_by_axis"] = tumor_slices_by_axis
    results["volume_points"] = {
        "brain": sample_point_cloud(img_data[:, :, :, 0], max_points=18000, min_points=4000),
        "tumor": sample_point_cloud(mask_volume, mask=mask_volume > 0, max_points=6000, min_points=0),
    }
    return results


@app.get("/slice-preview/{scan_id}")
async def slice_preview(
    scan_id: str,
    axis: str = Query(..., pattern="^(axial|coronal|sagittal)$"),
    index: int = Query(..., ge=0),
):
    cached = analysis_cache.get(scan_id)
    if cached is None:
        return JSONResponse({"error": "Scan session expired."}, status_code=404)

    img_data = cached["img_data"]
    slice_counts = {
        "axial": img_data.shape[2],
        "coronal": img_data.shape[1],
        "sagittal": img_data.shape[0],
    }
    if index >= slice_counts[axis]:
        return JSONResponse({"error": "Slice index out of range."}, status_code=400)

    return build_slice_preview(cached["img_data"], cached["mask_volume"], axis, index)


@app.get("/slice-stack/{scan_id}")
async def slice_stack(
    scan_id: str,
    axis: str = Query(..., pattern="^(axial|coronal|sagittal)$"),
):
    cached = analysis_cache.get(scan_id)
    if cached is None:
        return JSONResponse({"error": "Scan session expired."}, status_code=404)

    tumor_slices = get_tumor_slices_by_axis(cached["mask_volume"])[axis]
    previews = []
    for slice_index in tumor_slices:
        previews.append(build_slice_preview(cached["img_data"], cached["mask_volume"], axis, slice_index, image_size=64))

    return {"axis": axis, "items": previews}
