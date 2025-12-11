from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import cv2


def analyze_dataset(
    csv_path: Path,
    root_dir: Optional[Path] = None,
    max_images_for_shape_stats: int = 2000,
) -> None:
    """
    Analyze the EMOTIC/FER-derived face dataset CSV and print dataset statistics:
    - Total images
    - # images from EMOTIC
    - # images from FER2013
    - Per-class distributions
    - Missing file check
    - Image shape stats on a sample
    """

    print(f"=== Analyzing dataset ===")
    print(f"CSV path: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    print(f"\nRows: {n_rows}, Columns: {n_cols}")
    print(f"Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # Count EMOTIC vs FER2013 images
    # ------------------------------------------------------------------
    if "filepath" not in df.columns:
        raise RuntimeError("CSV must contain a 'filepath' column.")

    fer_mask = df["filepath"].str.contains("images_fer2013", na=False)
    num_fer = int(fer_mask.sum())
    num_emotic = n_rows - num_fer

    print("\n=== Dataset Source Breakdown ===")
    print(f"Total images:      {n_rows}")
    print(f"EMOTIC images:     {num_emotic}  ({num_emotic / n_rows * 100:.2f}%)")
    print(f"FER2013 images:    {num_fer}      ({num_fer / n_rows * 100:.2f}%)")

    # Show examples of each dataset
    if num_emotic > 0:
        print("\nExample EMOTIC rows:")
        print(df[~fer_mask].head())

    if num_fer > 0:
        print("\nExample FER2013 rows:")
        print(df[fer_mask].head())

    # ------------------------------------------------------------------
    # Label distributions
    # ------------------------------------------------------------------
    if "coarse_label" in df.columns:
        print("\n=== Coarse 6-emotion label distribution (ALL rows) ===")
        coarse_counts = df["coarse_label"].value_counts().sort_values(ascending=False)
        coarse_perc = (coarse_counts / n_rows * 100).round(2)
        print(pd.DataFrame({"count": coarse_counts, "percent": coarse_perc}))

    # Show FER-only distribution too
    if num_fer > 0:
        print("\n=== FER2013 ONLY — coarse label distribution ===")
        fer_df = df[fer_mask]
        fer_counts = fer_df["coarse_label"].value_counts().sort_values(ascending=False)
        fer_perc = (fer_counts / num_fer * 100).round(2)
        print(pd.DataFrame({"count": fer_counts, "percent": fer_perc}))

    # ------------------------------------------------------------------
    # File existence check
    # ------------------------------------------------------------------
    print("\n=== Filepath checks ===")

    base_dir = csv_path.parent if root_dir is None else Path(root_dir)

    resolved_paths = []
    missing = 0
    for rel in df["filepath"]:
        p = Path(rel)
        if not p.is_absolute():
            p = base_dir / p
        resolved_paths.append(p)
        if not p.exists():
            missing += 1

    df["resolved_path"] = resolved_paths

    print(f"Total filepaths listed: {len(resolved_paths)}")
    print(f"Missing files:          {missing}")

    if missing > 0:
        print("\nExample missing files:")
        for p in [x for x in resolved_paths if not x.exists()][:10]:
            print("  ", p)

    # ------------------------------------------------------------------
    # Image shape stats
    # ------------------------------------------------------------------
    print("\n=== Image shape statistics (sample) ===")
    existing_df = df[df["resolved_path"].apply(lambda p: p.exists())]
    if len(existing_df) == 0:
        print("No existing images found.")
        return

    sample_df = existing_df.sample(
        min(max_images_for_shape_stats, len(existing_df)),
        random_state=42,
    )

    shapes = []
    for p in sample_df["resolved_path"]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w, c = img.shape
        shapes.append((h, w, c))

    shapes = np.array(shapes)
    if len(shapes) == 0:
        print("Could not load any images.")
        return

    mean_h = shapes[:, 0].mean()
    mean_w = shapes[:, 1].mean()

    print(f"Sampled {len(shapes)} images")
    print(f"Mean height: {mean_h:.1f}")
    print(f"Mean width:  {mean_w:.1f}")

    unique_shapes, counts = np.unique(shapes, axis=0, return_counts=True)
    print("\nTop 5 image shapes:")
    order = np.argsort(-counts)
    for idx in order[:5]:
        h, w, c = unique_shapes[idx]
        print(f"  {h} x {w} x {c}  →  {counts[idx]} images")

    print("\n=== Done ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EMOTIC + FER2013 dataset.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/emotic_faces_128/labels_coarse.csv",
        help="Path to labels CSV.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/emotic_faces_128",
        help="Root directory for resolving image paths.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=2000,
        help="Max images to load for shape stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_dataset(
        csv_path=Path(args.csv),
        root_dir=Path(args.root_dir),
        max_images_for_shape_stats=args.max_images,
    )


if __name__ == "__main__":
    main()