import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_COARSE = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"

dfc = pd.read_csv(CSV_COARSE)
print("=== Coarse label distribution ===")
print(dfc["coarse_label"].value_counts())
print("\nPercent:\n", dfc["coarse_label"].value_counts(normalize=True) * 100)