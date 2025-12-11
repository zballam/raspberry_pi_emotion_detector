# utility/face_dataloader.py

from pathlib import Path
from typing import Optional, Callable, Any, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmoticFaceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label_column: str = "coarse_label",
        transform=None,
        root_dir: Optional[str] = None,
    ):
        """
        csv_path: path to labels CSV (labels_coarse.csv)
        label_column: which column to use as the class label
        root_dir: base directory for images. If None, uses csv_path.parent
        """
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        self.label_column = label_column
        self.transform = transform

        # Root dir for images
        if root_dir is None:
            self.root_dir = self.csv_path.parent  # e.g. data/emotic_faces_128
        else:
            self.root_dir = Path(root_dir)

        print(f"[EmoticFaceDataset] Loaded {len(self.df)} rows from {self.csv_path}")
        print(f"[EmoticFaceDataset] Using label column: '{self.label_column}'")
        print(f"[EmoticFaceDataset] Num classes: {self.df[self.label_column].nunique()}")
        print(f"[EmoticFaceDataset] Root dir for images: {self.root_dir}")

        # Build label <-> index mapping
        labels = sorted(self.df[self.label_column].unique())
        self.label_to_idx: Dict[Any, int] = {lab: i for i, lab in enumerate(labels)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> Path:
        """
        Resolve image path robustly:
        - If absolute, return as-is
        - If already starts with root_dir, don't prefix again
        - Otherwise, join root_dir / rel_path
        """
        p = Path(rel_path)

        # Already absolute -> just return
        if p.is_absolute():
            return p

        root = self.root_dir

        # If the relative path already begins with the root dir,
        # e.g. "data/emotic_faces_128/['Engagement']/sample_....jpg",
        # don't add root again.
        try:
            if len(p.parts) >= len(root.parts) and p.parts[: len(root.parts)] == root.parts:
                return p
        except Exception:
            # If anything weird happens, just fall back to root / p
            pass

        return root / p

    def __getitem__(self, idx):
        # Make sure idx is a plain integer, not a tensor or np.int64
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        else:
            idx = int(idx)

        row = self.df.iloc[idx]

        rel_path = row["filepath"]
        img_path = self._resolve_path(rel_path)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, float32 in [0,1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        x = torch.from_numpy(img)

        if self.transform is not None:
            x = self.transform(x)

        label_name = row[self.label_column]
        y = self.label_to_idx[label_name]

        return x, y