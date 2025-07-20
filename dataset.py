import json
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.ndimage import zoom
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

filenames = {'r_TEP_cerveau_10min_Corrige_avec_psf.nii.gz', 'r_TEP_AC_VPFX.nii.gz', 'r_TEP_cerveau_10min_Corrige_avec_psfa.nii.gz', 'r_e+1_BRAIN_MAC_ZTE.nii.gz',
             'r_TEP_cerveau_10min_Corrige_sans_psf.nii.gz', 'r_Brain_MAC_ATLAS.nii.gz', 'r_TEP_cerveau_5min_Corrige_avec_psf.nii.gz', 'r_Brain_MAC_ZTE.nii.gz',
             'r_BRAIN_MAC_ZTE.nii.gz', 'r_e+1_TEP_AC_VPFX.nii.gz', 'r_TEP_AC_QC.nii.gz', 'r_BRAIN_MAC_ATLAS.nii.gz', 'r_Brain_MAC_Atlas.nii.gz', 'r_TEP_cerveau_5min_Corrige_sans_psf.nii.gz',
             'r_e+1_Brain_MAC_Atlas.nii.gz', 'r_e+1_TEP_AC_QC.nii.gz'}

def load_nifti(path):
    data = nib.load(path).get_fdata()
    data = np.nan_to_num(data, nan=0.0)
    return data

def pad_to_cube(image):
    max_dim = max(image.shape)
    pad_width = []
    for s in image.shape:
        total_pad = max_dim - s
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(image, pad_width, mode='constant')

def resample(image, target_shape=32):
    factors = [t / s for t, s in zip((target_shape, target_shape, target_shape), image.shape)]
    return zoom(image, factors, order=3)

def normalize(image, mode="masked-zscore"):
    if mode == "masked-zscore":
        mask = image > 0
        mean = image[mask].mean()
        std = image[mask].std()
        return (image - mean) / std if std > 0 else image
    elif mode == "minmax":
        return (image - image.min()) / (image.max() - image.min() + 1e-8)
    else:
        raise ValueError("Unknown normalization mode")

def random_shift_3d(volume, max_fraction=0.10):
    D, H, W = volume.shape
    max_shifts = (int(D * max_fraction), int(H * max_fraction), int(W * max_fraction))
    
    shifts = [np.random.randint(-max_shift, max_shift + 1) for max_shift in max_shifts]
    shifted_vol = np.zeros_like(volume)
    
    def get_slices(shift, size):
        if shift > 0:
            src_slice = slice(0, size - shift)
            dst_slice = slice(shift, size)
        elif shift < 0:
            src_slice = slice(-shift, size)
            dst_slice = slice(0, size + shift)
        else:
            src_slice = slice(0, size)
            dst_slice = slice(0, size)
        return src_slice, dst_slice

    d_src, d_dst = get_slices(shifts[0], D)
    h_src, h_dst = get_slices(shifts[1], H)
    w_src, w_dst = get_slices(shifts[2], W)

    shifted_vol[d_dst, h_dst, w_dst] = volume[d_src, h_src, w_src]

    return shifted_vol

def random_rotate_3d(volume, angle=20, probability=1):
    if np.random.rand() > probability:
        return volume  
    
    angles = np.random.uniform(-angle, angle, size=3)

    vol = rotate(volume, angle=angles[0], axes=(1, 2), reshape=False, order=3, mode='constant', cval=0)
    vol = rotate(vol, angle=angles[1], axes=(0, 2), reshape=False, order=3, mode='constant', cval=0)
    vol = rotate(vol, angle=angles[2], axes=(0, 1), reshape=False, order=3, mode='constant', cval=0)

    return vol

def transform(vol):
    vol = random_shift_3d(vol)
    vol = random_rotate_3d(vol)
    return vol


class RawDataset(Dataset):
    def __init__(self, data_root, json_path, filenames, target_shape):
        self.data_root = data_root
        self.filenames = filenames
        self.target_shape = target_shape

        with open(json_path, 'r') as f:
            raw = json.load(f)
        self.caption_lookup = {
            (entry['SubjectID'], entry['Date']): entry['PET_results']
            for entry in raw
        }

        # Find all NIfTI files and build list of (path, SubjectID, Date)
        self.samples = []

        # Convert root to Path object
        root = Path(data_root)

        # Match all NIfTI files under SubjectID/Date/
        for nii_path in root.glob("**/*.nii.gz"):
            if nii_path.name in self.filenames:
                subject_id = nii_path.parts[-3]
                date = nii_path.parts[-2]
                self.samples.append((str(nii_path), subject_id, date))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, subject_id, date = self.samples[idx]
        key = (subject_id, date)
        if key not in self.caption_lookup:
            raise KeyError(f"Missing caption for SubjectID={subject_id}, Date={date}")
        caption = self.caption_lookup[key]

        # Load and preprocess image
        img = load_nifti(path)
        img = pad_to_cube(img)
        img = resample(img, target_shape=self.target_shape)

        # img = normalize(img, mode="masked-zscore")
        

        return img, caption

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        image, caption = data
        if self.transform:
            image = self.transform(image)
        image = normalize(image, mode="masked-zscore")
        image = np.expand_dims(image, axis=0)

        return torch.tensor(image, dtype=torch.float32), caption

def show_slices(image_3d, title=None):
    """Display one central slice in each dimension (axial, coronal, sagittal)"""
    image_3d = image_3d.squeeze().numpy()

    d, h, w = image_3d.shape
    slices = [
        image_3d[d // 2, :, :],
        image_3d[:, h // 2, :],
        image_3d[:, :, w // 2],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if title:
        fig.suptitle(title, fontsize=16)

    for i, slice_ in enumerate(slices):
        slice_ = np.rot90(slice_, k=1)
        axes[i].imshow(slice_, cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    dataset = RawDataset(data_root="dataset/", json_path="dataset/reports.json", filenames=filenames, target_shape=64)
    transformed_dataset = TransformedDataset(dataset, transform=transform)
    image, caption = transformed_dataset[2]
    print(image.shape)
    print(caption)
    print(len(dataset))
    show_slices(image, title="Sample Volume Slices")

    caption_lengths = [len(dataset[i][1].split()) for i in range(len(dataset))]
    max_len = max(caption_lengths)
    median_len = int(np.median(caption_lengths))

    print(f"Max caption length: {max_len}")
    print(f"Median caption length: {median_len}")

if __name__ == "__main__":
    main()