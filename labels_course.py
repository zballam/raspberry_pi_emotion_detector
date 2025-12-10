import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_COARSE = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"

dfc = pd.read_csv(CSV_COARSE)
print("=== Coarse label distribution ===")
print(dfc["coarse_label"].value_counts())
print("\nPercent:\n", dfc["coarse_label"].value_counts(normalize=True) * 100)

ROOT = Path(__file__).resolve().parent
CSV_COARSE = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"

dfc = pd.read_csv(CSV_COARSE)
counts = dfc["coarse_label"].value_counts().sort_index()
freq = counts.values.astype(np.float32)
priors = freq / freq.sum()
print("Class priors:", priors.tolist())