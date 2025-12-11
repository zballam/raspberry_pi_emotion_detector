from collections import Counter
from pathlib import Path

from utility.face_dataloader import EmoticFaceDataset

CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"
LABEL_COLUMN = "coarse_label"

ds = EmoticFaceDataset(CSV_PATH, LABEL_COLUMN, transform=None)
print("label_to_idx:", ds.label_to_idx)

# how many of each label?
counts = Counter(ds.df[LABEL_COLUMN])
print("\nClass counts:")
for label, c in counts.items():
    print(f"{label:15s} -> {c}")