# build_emotic_faces.py
from pathlib import Path

from scipy.io import loadmat

from utility.face_pipeline import (
    create_face_detector,
    build_face_dataset,
)

import pandas as pd


# -------------------------
# 26 EMOTIC -> 6 coarse classes
# -------------------------

EMOTIC_TO_COARSE = {
    "Peace":        "happy",
    "Affection":    "happy",
    "Esteem":       "happy",
    "Anticipation": "surprised",
    "Engagement":   "happy",
    "Confidence":   "happy",
    "Happiness":    "happy",
    "Pleasure":     "happy",
    "Excitement":   "happy",
    "Surprise":     "surprised",
    "Sympathy":     "happy",

    "Doubt/Confusion": "confused",
    "Disconnection":   "neutral",
    "Fatigue":         "sad",
    "Embarrassment":   "sad",
    "Yearning":        "sad",

    "Disapproval":  "angry",
    "Aversion":     "angry",
    "Annoyance":    "angry",
    "Anger":        "angry",

    "Sensitivity":  "sad",
    "Sadness":      "sad",
    "Disquietment": "confused",
    "Fear":         "confused",
    "Pain":         "sad",
    "Suffering":    "sad",
}


def _clean_dominant(raw: str) -> str:
    """
    Turn strings like "['Engagement']" into "Engagement".
    Handles extra spaces/quotes safely.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]
    return parts[0] if parts else ""


def add_coarse_labels(
    output_root: str | Path,
    base_csv_name: str = "labels.csv",
    coarse_csv_name: str = "labels_coarse.csv",
) -> None:
    """
    Read labels.csv written by build_face_dataset, clean 'dominant',
    map to 6-way 'coarse_label', and write labels_coarse.csv.
    """
    output_root = Path(output_root)
    csv_in = output_root / base_csv_name
    csv_out = output_root / coarse_csv_name

    if not csv_in.exists():
        raise FileNotFoundError(f"Base label CSV not found: {csv_in}")

    df = pd.read_csv(csv_in)
    print(f"\nLoaded {len(df)} rows from {csv_in}")

    # Clean dominant labels (e.g. "['Engagement']" -> "Engagement")
    df["dominant_clean"] = df["dominant"].apply(_clean_dominant)

    # Map to coarse 6-way labels
    df["coarse_label"] = df["dominant_clean"].map(EMOTIC_TO_COARSE)

    before = len(df)
    df = df.dropna(subset=["coarse_label"])
    after = len(df)

    print(f"Dropped {before - after} rows with unmapped dominant labels.")
    print("\nCoarse 6-way label distribution:")
    print(df["coarse_label"].value_counts())

    df.to_csv(csv_out, index=False)
    print(f"\nSaved CSV with coarse labels to: {csv_out}")


def main():
    # 1) Load EMOTIC annotations
    annotations_path = "emotic/Annotations/annotations.mat"
    data = loadmat(annotations_path)
    train_data = data["train"][0]

    print("Loaded EMOTIC train split:", len(train_data), "samples")

    # 2) Create face detector
    detector = create_face_detector(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    # 3) Build face dataset (crops + labels.csv)
    output_root = "data/emotic_faces_128"
    build_face_dataset(
        data=train_data,
        output_root=output_root,
        detector=detector,
        root="emotic",
        face_size=128,
        max_samples=None,      # or None for all
        min_face_score=0.5,
    )

    # 4) Read labels.csv and add 6-way coarse labels
    add_coarse_labels(output_root)


if __name__ == "__main__":
    main()