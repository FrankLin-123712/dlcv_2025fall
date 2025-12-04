# data_flat.py
from pathlib import Path
from PIL import Image
import os, random
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class FlatFolderDataset(Dataset):
    """Load ALL images under a single folder (no subfolders). 
       Labels are ignored (for SSL). transform should return multi-crop list."""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = [p for p in self.root.iterdir() if p.suffix.lower() in IMG_EXTS]
        self.transform = transform
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under: {self.root}")
        # optional: deterministic order
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img), 0  # returns list of crops for DINO
        return img
