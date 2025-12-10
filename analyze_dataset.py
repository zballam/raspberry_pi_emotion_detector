from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import cv2


def analyze_dataset(
    csv_path: Path,
    root_dir: Optional[Path] = None,
    max_images_for_shape_stats: int = 2000,
) -> None:
    """
    Analyze the EMOTIC face dataset CSV and print useful statistics.

    Args:
        csv_path: Path to labels CSV (e.g. data/emotic_faces_128/labels.csv).
        root_dir: Optional root directory that 'filepath' is relative to.
                  If None, file paths are treated as absolute or relative
                  to the CSV location.
        max_images_for_shape_stats: Maximum number of images to load for
                  computing basic shape statistics (to avoid long runtimes).
    """
    print(f"=== Analyzing dataset ===")
    print(f"CSV path: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    print(f"\nRows: {n_rows}, Columns: {n_cols}")
    print(f"Columns: {list(df.columns)}")

    # Basic NaN info
    print("\n=== Missing values per column ===")
    print(df.isna().sum())

    # ------------------------------------------------------------------
    # Label distributions
    # ------------------------------------------------------------------
    if "dominant" in df.columns:
        print("\n=== EMOTIC 'dominant' label distribution ===")
        emotic_counts = df["dominant"].value_counts().sort_values(ascending=False)
        emotic_perc = (emotic_counts / n_rows * 100).round(2)
        label_stats = pd.DataFrame(
            {"count": emotic_counts, "percent": emotic_perc}
        )
        print(label_stats)

        if len(emotic_counts) > 0:
            max_class = emotic_counts.iloc[0]
            min_class = emotic_counts.iloc[-1]
            imbalance = max_class / max(1, min_class)
            print(
                f"\n[dominant] num_classes={len(emotic_counts)}, "
                f"max_count={max_class}, min_count={min_class}, "
                f"max/min imbalance ~ {imbalance:.1f}x"
            )
    else:
        print("\nNo 'dominant' column found in CSV.")

    if "coarse_label" in df.columns:
        print("\n=== Coarse 6-emotion label distribution ===")
        coarse_counts = df["coarse_label"].value_counts().sort_values(ascending=False)
        coarse_perc = (coarse_counts / n_rows * 100).round(2)
        coarse_stats = pd.DataFrame(
            {"count": coarse_counts, "percent": coarse_perc}
        )
        print(coarse_stats)

        if len(coarse_counts) > 0:
            max_c = coarse_counts.iloc[0]
            min_c = coarse_counts.iloc[-1]
            imbalance = max_c / max(1, min_c)
            print(
                f"\n[coarse_label] num_classes={len(coarse_counts)}, "
                f"max_count={max_c}, min_count={min_c}, "
                f"max/min imbalance ~ {imbalance:.1f}x"
            )
    else:
        print("\nNo 'coarse_label' column found. You can add this using your mapping.")

    # ------------------------------------------------------------------
    # Filepaths: existence and duplicates
    # ------------------------------------------------------------------
    if "filepath" not in df.columns:
        print("\nNo 'filepath' column found, skipping file checks.")
    else:
        print("\n=== Filepath checks ===")
        # Resolve base directory for relative paths
        if root_dir is None:
            base_dir = csv_path.parent
        else:
            base_dir = root_dir

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

        unique_paths = len(set(resolved_paths))
        print(f"Total paths listed: {len(resolved_paths)}")
        print(f"Unique paths     : {unique_paths}")
        print(f"Missing files    : {missing}")

        if missing > 0:
            print("\nExample missing files (up to 10):")
            missing_paths = [p for p in resolved_paths if not p.exists()]
            for p in missing_paths[:10]:
                print("  ", p)

    # ------------------------------------------------------------------
    # Image shape statistics (optional, on a subset)
    # ------------------------------------------------------------------
    if "resolved_path" in df.columns:
        print("\n=== Image shape statistics (sample) ===")
        # Restrict to existing files
        existing_df = df[df["resolved_path"].apply(lambda p: p.exists())]
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

        if not shapes:
            print("Could not read any images for shape stats.")
        else:
            shapes_arr = np.array(shapes)
            mean_h = shapes_arr[:, 0].mean()
            mean_w = shapes_arr[:, 1].mean()

            print(f"Sampled {len(shapes)} images for shape stats.")
            print(f"Mean height: {mean_h:.1f}, Mean width: {mean_w:.1f}")
            unique_shapes, counts = np.unique(shapes_arr, axis=0, return_counts=True)

            print("\nMost common shapes (up to 5):")
            # Sort by count descending
            order = np.argsort(-counts)
            for idx in order[:5]:
                h, w, c = unique_shapes[idx]
                print(f"  {h} x {w} x {c} -> {counts[idx]} images")
    else:
        print("\nSkipping image shape stats (no resolved paths).")

    print("\n=== Done ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze EMOTIC face dataset statistics."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/emotic_faces_128/labels.csv",
        help="Path to labels CSV.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Optional root directory for image paths (if filepaths are relative).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=2000,
        help="Max images to load for shape statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    root_dir = Path(args.root_dir) if args.root_dir is not None else None

    analyze_dataset(
        csv_path=csv_path,
        root_dir=root_dir,
        max_images_for_shape_stats=args.max_images,
    )


if __name__ == "__main__":
    main()