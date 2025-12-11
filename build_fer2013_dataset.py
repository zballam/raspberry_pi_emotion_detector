"""
build_fer2013_dataset.py

One-time script to:
- Download FER2013 from Kaggle (msambare/fer2013)
- Resize images to 128x128
- Map FER emotions to your 6 coarse labels
- Save images under data/emotic_faces_128/images_fer2013/...
- Append new rows to data/emotic_faces_128/labels_coarse.csv

Requires:
    pip install kaggle pillow
And a working Kaggle CLI (kaggle.json in the right place).
"""

import csv
import subprocess
import zipfile
from pathlib import Path
from collections import Counter

from PIL import Image


# -------------------------------
# Paths & constants
# -------------------------------

ROOT = Path(__file__).resolve().parent

# This is the directory your EmoticFaceDataset is already using
DATA_ROOT = ROOT / "data" / "emotic_faces_128"
CSV_PATH = DATA_ROOT / "labels_coarse.csv"

# Where we’ll put FER2013 images inside your existing dataset root
FER_IMAGES_ROOT = DATA_ROOT / "images_fer2013"

# Where we’ll download & unzip the raw Kaggle dataset
FER_DOWNLOAD_DIR = ROOT / "data" / "raw_fer2013"
FER_ZIP_PATH = FER_DOWNLOAD_DIR / "fer2013.zip"

KAGGLE_DATASET = "msambare/fer2013"

# ⚠️ Your CSV uses 'filepath' (not 'image_path')
IMAGE_COL = "filepath"

# Map FER2013 class folders → your coarse labels
# FER classes: angry, disgust, fear, happy, sad, surprise, neutral
FER_TO_COARSE = {
    "angry": "angry",
    "disgust": "angry",      # treat disgust as anger-like
    "fear": "confused",      # use fear as your "confused" class
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprised",
    "neutral": "neutral",
}

TARGET_SIZE = (128, 128)


# -------------------------------
# Helpers
# -------------------------------

def run_kaggle_download():
    FER_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"==> Downloading {KAGGLE_DATASET} to {FER_DOWNLOAD_DIR} ...")

    # Use Kaggle CLI
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(FER_DOWNLOAD_DIR),
        "--force",
    ]
    subprocess.run(cmd, check=True)

    # Find the zip (Kaggle may name it like fer2013.zip or similar)
    zips = list(FER_DOWNLOAD_DIR.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip file found in {FER_DOWNLOAD_DIR}")
    # Take the first one
    zip_path = zips[0]
    print(f"==> Using zip: {zip_path}")
    return zip_path


def unzip_fer(zip_path: Path) -> Path:
    print(f"==> Unzipping {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(FER_DOWNLOAD_DIR)

    # Find folder that contains a "train" subdir
    fer_root = None
    for p in FER_DOWNLOAD_DIR.rglob("train"):
        if p.is_dir():
            fer_root = p.parent
            break

    if fer_root is None:
        raise RuntimeError(
            f"Could not find a 'train' folder under {FER_DOWNLOAD_DIR}."
        )

    print(f"==> Found FER2013 root: {fer_root}")
    return fer_root


def load_csv_fieldnames(csv_path: Path):
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise RuntimeError(f"No header / fieldnames found in {csv_path}")
    if IMAGE_COL not in fieldnames or "coarse_label" not in fieldnames:
        raise RuntimeError(
            f"CSV must contain '{IMAGE_COL}' and 'coarse_label' columns. "
            f"Found: {fieldnames}"
        )
    return fieldnames


def iterate_fer_images(fer_root: Path):
    """
    Yield (split, class_name, image_path) for FER2013 images.
    Expects fer_root/train/<class>/*.jpg and fer_root/test/<class>/*.jpg
    """
    for split in ["train", "test"]:
        split_dir = fer_root / split
        if not split_dir.exists():
            print(f"  [WARN] Split dir not found: {split_dir}, skipping.")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            if class_name not in FER_TO_COARSE:
                print(f"  [WARN] Unknown FER class '{class_name}', skipping.")
                continue

            for img_path in class_dir.glob("*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                yield split, class_name, img_path


def prepare_output_dirs():
    if FER_IMAGES_ROOT.exists():
        print(
            f"==> WARNING: {FER_IMAGES_ROOT} already exists.\n"
            f"    To avoid duplicating entries in your CSV, delete this folder "
            f"    and revert labels_coarse.csv if you want a clean rebuild.\n"
        )
    FER_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


def process_and_append(fer_root: Path):
    fieldnames = load_csv_fieldnames(CSV_PATH)
    prepare_output_dirs()

    counts = Counter()
    written = 0

    with CSV_PATH.open("a", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)

        for split, fer_class, src_path in iterate_fer_images(fer_root):
            coarse = FER_TO_COARSE[fer_class]

            # Load and resize
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"  [ERROR] Failed to open image {src_path}: {e}")
                continue

            img = img.resize(TARGET_SIZE, Image.BILINEAR)

            # Save under data/emotic_faces_128/images_fer2013/<split>/<fer_class>/<filename>
            out_dir = FER_IMAGES_ROOT / split / fer_class
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / src_path.name

            try:
                img.save(out_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"  [ERROR] Failed to save image {out_path}: {e}")
                continue

            # Path stored in CSV should be relative to DATA_ROOT
            rel_path = out_path.relative_to(DATA_ROOT)

            # Build a row with all fieldnames present
            row = {name: "" for name in fieldnames}
            row[IMAGE_COL] = str(rel_path).replace("\\", "/")
            row["coarse_label"] = coarse

            writer.writerow(row)
            counts[coarse] += 1
            written += 1

            if written % 1000 == 0:
                print(f"  -> Processed {written} images so far...")

    print("\n==> Done appending FER2013 to labels_coarse.csv")
    print(f"    Total new images: {written}")
    print("    Class counts (coarse labels):")
    for label, c in sorted(counts.items()):
        print(f"      {label:10s}: {c}")


# -------------------------------
# Main
# -------------------------------

def main():
    print("=== Building FER2013-derived dataset into emotic_faces_128 ===")

    if not DATA_ROOT.exists():
        raise RuntimeError(
            f"{DATA_ROOT} does not exist. "
            "Run this from your project root where data/emotic_faces_128 lives."
        )

    # 1) Download via Kaggle (if needed)
    if FER_ZIP_PATH.exists():
        print(f"==> Found existing zip: {FER_ZIP_PATH}")
        zip_path = FER_ZIP_PATH
    else:
        zip_path = run_kaggle_download()
        # Normalize name to fer2013.zip for consistency
        if zip_path.name != "fer2013.zip":
            FER_ZIP_PATH.unlink(missing_ok=True)
            zip_path.rename(FER_ZIP_PATH)
            zip_path = FER_ZIP_PATH

    # 2) Unzip and locate FER root
    fer_root = unzip_fer(zip_path)

    # 3) Process images & append rows to labels_coarse.csv
    process_and_append(fer_root)

    print("\nAll done! You can now retrain your TinyCNN with the augmented dataset.")


if __name__ == "__main__":
    main()