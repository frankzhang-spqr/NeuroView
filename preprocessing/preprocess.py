import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image

def get_patient_folders(data_path):
    """
    Get a list of all patient folders in the dataset. This is a best-effort guess
    and might need adjustment based on the exact dataset structure.
    """
    print(f"Searching for patient folders in: {os.path.abspath(data_path)}")
    patient_folders = set()
    for root, dirs, files in os.walk(data_path):
        if any(f.endswith('.nii.gz') for f in files):
            # The patient folder is the one containing the NIfTI files.
            patient_folders.add(root)
    
    # The validation data has an extra 'validation_data' subfolder
    # This logic tries to handle that by adding the parent if it seems more correct.
    final_folders = set()
    for p in patient_folders:
        if 'validation_data' in p:
             final_folders.add(os.path.dirname(p))
        else:
             final_folders.add(p)

    print(f"Found {len(final_folders)} unique patient folders.")
    return sorted(list(final_folders))

def find_file_by_substring(folder, substring):
    """Finds the first file in a folder containing a specific substring."""
    for f in os.listdir(folder):
        if substring in f and f.endswith('.nii.gz'):
            return os.path.join(folder, f)
    return None

def process_and_save_slices(patient_folders, output_path, is_test=False):
    """
    Process MRI scans and save 2D slices, with robust file finding.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    slice_counter = 0
    for patient_folder in tqdm(patient_folders, desc="Processing Patients"):
        try:
            tumor_type = 'GLI' if 'GLI' in patient_folder else 'MEN'
            
            # Robustly find the ground truth file
            seg_path = find_file_by_substring(patient_folder, '-seg.nii.gz')
            if not seg_path:
                # Fallback for MEN scans which might use a different naming convention
                seg_path = find_file_by_substring(patient_folder, '-GTV.nii.gz')

            if not seg_path:
                print(f"Warning: Skipping {patient_folder} due to missing ground truth file.")
                continue

            modalities_to_load = []
            mod_types = ['t1c', 't1n', 't2f', 't2w']
            for mod in mod_types:
                mod_path = find_file_by_substring(patient_folder, f'-{mod}.nii.gz')
                if mod_path:
                    modalities_to_load.append(nib.load(mod_path).get_fdata())
                else:
                    # If a modality is missing, we can't proceed with this patient
                    print(f"Warning: Skipping {patient_folder} due to missing modality: {mod}")
                    modalities_to_load = [] # Reset
                    break
            
            if len(modalities_to_load) < 4:
                continue

            if not modalities_to_load:
                print(f"Warning: Skipping {patient_folder} as no required modalities were found.")
                continue

            stacked_modalities = np.stack(modalities_to_load, axis=-1)
            seg_img = nib.load(seg_path).get_fdata()

            for c in range(stacked_modalities.shape[-1]):
                channel = stacked_modalities[..., c]
                min_val, max_val = np.min(channel), np.max(channel)
                if max_val > min_val:
                    stacked_modalities[..., c] = (channel - min_val) / (max_val - min_val)

            for i in range(stacked_modalities.shape[2]):
                slice_img_multi_channel = stacked_modalities[:, :, i, :]
                slice_seg = seg_img[:, :, i]

                if np.sum(slice_img_multi_channel) == 0:
                    continue
                
                resized_channels = []
                for c in range(slice_img_multi_channel.shape[-1]):
                    resized_channels.append(np.array(Image.fromarray(slice_img_multi_channel[..., c]).resize((240, 240))))
                
                slice_img = np.stack(resized_channels, axis=-1)
                slice_seg = np.array(Image.fromarray(slice_seg).resize((240, 240), resample=Image.NEAREST))

                slice_seg[slice_seg > 0] = 1

                slice_filename = f"slice_{slice_counter}.npy"
                np.save(os.path.join(output_path, slice_filename), slice_img.astype(np.float32))
                
                label_filename = f"label_{slice_counter}.npy"
                np.save(os.path.join(output_path, label_filename), slice_seg.astype(np.uint8))

                type_filename = f"type_{slice_counter}.npy"
                np.save(os.path.join(output_path, type_filename), np.array([0 if tumor_type == 'GLI' else 1]))

                slice_counter += 1
        except Exception as e:
            print(f"Error processing {patient_folder}: {e}")
            
    return slice_counter

if __name__ == '__main__':
    gli_path = 'BraTS/GLI/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2'
    men_path = 'BraTS/MEN' # Updated path
    output_path = 'data/processed_slices'
    
    all_folders = get_patient_folders(gli_path) + get_patient_folders(men_path)
    
    slice_counter = process_and_save_slices(all_folders, output_path)
    print(f"Processed and saved {slice_counter} slices.")