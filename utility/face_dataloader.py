# utility/face_dataloader.py

from pathlib import Path

import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class EmoticFaceDataset(Dataset):
    """
    Dataset for loading cropped EMOTIC face images and labels.

    Assumes CSV has at least:
        - 'filepath': path to image file (relative or absolute)
        - label column: e.g. 'coarse_label' or 'dominant'
    """

    def __init__(self, csv_path, label_column="coarse_label", transform=None, root_dir=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.label_column = label_column

        # ✅ If root_dir is None, we will trust the CSV paths as-is
        self.root_dir = Path(root_dir) if root_dir is not None else None

        if self.label_column not in self.df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in CSV. "
                f"Available columns: {list(self.df.columns)}"
            )

        unique_labels = sorted(self.df[self.label_column].unique())
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        print(f"[EmoticFaceDataset] Loaded {len(self.df)} rows from {self.csv_path}")
        print(f"[EmoticFaceDataset] Using label column: '{self.label_column}'")
        print(f"[EmoticFaceDataset] Num classes: {len(self.label_to_idx)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        row = self.df.iloc[idx]

        # --- Image path ---
        rel_path = Path(row["filepath"])

        if rel_path.is_absolute() or self.root_dir is None:
            img_path = rel_path
        else:
            img_path = self.root_dir / rel_path

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)

        label_name = row[self.label_column]
        label_idx = self.label_to_idx[label_name]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor