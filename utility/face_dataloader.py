# utility/face_dataloader.py

import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class EmoticFaceDataset(Dataset):
    """
    Dataset for loading cropped EMOTIC face images and labels.
    Assumes:
        - labels.csv with columns: filepath, dominant, all_labels
        - face images in 128x128 RGB format
    """

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Map each label to an integer
        unique_labels = sorted(self.df["dominant"].unique())
        self.label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load image ---
        path = row["filepath"]
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
        img_rgb = img_rgb.astype("float32") / 255.0

        # Convert to tensor (C,H,W)
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1)

        # --- Label ---
        label_name = row["dominant"]
        label_idx = self.label_to_idx[label_name]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # Optional transforms (augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor