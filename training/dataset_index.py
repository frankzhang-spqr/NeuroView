import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def _index_path(data_dir: str) -> str:
    return os.path.join(data_dir, "training_index.json")


def build_or_load_index(data_dir: str, force_rebuild: bool = False) -> Dict:
    index_path = _index_path(data_dir)
    if os.path.exists(index_path) and not force_rebuild:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    slice_ids: List[int] = []
    for file_name in tqdm(os.listdir(data_dir), desc="Scanning slice files", leave=False):
        if file_name.startswith("slice_") and file_name.endswith(".npy"):
            slice_ids.append(int(file_name.split("_")[1].split(".")[0]))
    slice_ids = sorted(slice_ids)

    entries = []
    for slice_id in tqdm(slice_ids, desc="Indexing slice metadata", leave=False):
        label = np.load(os.path.join(data_dir, f"label_{slice_id}.npy"))
        type_label = int(np.load(os.path.join(data_dir, f"type_{slice_id}.npy"))[0])

        group_path = os.path.join(data_dir, f"group_{slice_id}.npy")
        if os.path.exists(group_path):
            group = str(np.load(group_path, allow_pickle=True)[0])
        else:
            group = f"slice-{slice_id}"

        entries.append({
            "slice_id": slice_id,
            "group": group,
            "type_label": type_label,
            "has_tumor": bool(label.sum() > 0),
            "tumor_pixels": int(label.sum()),
            "total_pixels": int(label.size),
        })

    index = {"entries": entries}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index


def summarize_entries(entries: List[Dict]) -> Dict:
    tumor_pixels = sum(entry["tumor_pixels"] for entry in entries)
    total_pixels = sum(entry["total_pixels"] for entry in entries)
    positive_slices = sum(1 for entry in entries if entry["has_tumor"])
    class_counts = {0: 0, 1: 0}
    for entry in entries:
        class_counts[entry["type_label"]] = class_counts.get(entry["type_label"], 0) + 1
    return {
        "tumor_pixels": tumor_pixels,
        "background_pixels": total_pixels - tumor_pixels,
        "positive_slices": positive_slices,
        "class_counts": class_counts,
    }
