# utility/face_pipeline.py

import os
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp

from utility.dataset import (
    get_image_path,
    load_image,
    get_person,
    get_body_bbox,
    crop_bbox,
)

__all__ = [
    "create_face_detector",
    "detect_face_in_body",
    "resize_for_model",
    "process_sample_to_face",
    "build_face_dataset",
]

mp_face_detection = mp.solutions.face_detection


# ---------------------------------------------------------------------
# Face detector helpers
# ---------------------------------------------------------------------

def create_face_detector(model_selection=0, min_detection_confidence=0.5):
    """
    Creates and returns a MediaPipe FaceDetection object.
    Reuse this instead of recreating for every image.
    """
    return mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )


def detect_face_in_body(body_crop_rgb, detector, min_score=0.5):
    """
    Run MediaPipe on an RGB image and return (x1, y1, x2, y2) in pixel coords.
    Returns None if no face is found with sufficient score.
    """
    h, w, _ = body_crop_rgb.shape

    # MediaPipe expects RGB uint8
    results = detector.process(body_crop_rgb)

    if not results.detections:
        return None

    # Pick the highest-score detection
    best = max(results.detections, key=lambda d: d.score[0])
    if best.score[0] < min_score:
        return None

    bbox = best.location_data.relative_bounding_box
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    # clamp
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    return x1, y1, x2, y2


def resize_for_model(face_crop_rgb, size=128):
    """
    Resize a face crop to (size, size) for training / inference.
    """
    return cv2.resize(face_crop_rgb, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------

def _walk_labels(obj, out_list):
    """
    Recursively walk a nested EMOTIC annotations_categories object and
    append any string labels into out_list.
    """
    # Plain Python strings
    if isinstance(obj, str):
        out_list.append(obj)
        return

    # NumPy scalar (e.g., np.str_)
    if isinstance(obj, np.generic):
        try:
            out_list.append(str(obj.item()))
        except Exception:
            pass
        return

    # NumPy array
    if isinstance(obj, np.ndarray):
        # If this is an array of strings, flatten and collect
        if obj.dtype.kind in ("U", "S"):   # Unicode or bytes
            for x in obj.reshape(-1):
                out_list.append(str(x))
        else:
            # Otherwise, recurse into each element
            for x in obj.reshape(-1):
                _walk_labels(x, out_list)
        return

    # List or tuple
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _walk_labels(x, out_list)
        return

    # Fallback: try .item()
    if hasattr(obj, "item"):
        try:
            v = obj.item()
            _walk_labels(v, out_list)
        except Exception:
            pass


def _extract_categories(person):
    """
    Extract discrete category labels from EMOTIC person struct.
    Returns a list of clean strings (may be empty).
    """
    # This is the weird nested EMOTIC structure, usually something like:
    # (array([[array(['Affection']), array(['Anticipation']), ...]], dtype=object),)
    raw = person["annotations_categories"][0]

    # Flatten the outer container (tuple/array of arrays)
    flat = np.array(raw, dtype=object).reshape(-1)

    labels = []
    for arr in flat:
        # Each arr is typically array(['Affection'], dtype='<U9') or a small object array
        arr_np = np.array(arr)

        for s in arr_np.reshape(-1):
            lab = str(s).strip()
            if lab and lab != "0" and lab not in labels:
                labels.append(lab)

    return labels


def _get_dominant_category(person):
    """
    Returns a single 'dominant' discrete emotion label for a person.
    For now: take the FIRST discrete category if any exist.
    """
    cats = _extract_categories(person)
    if not cats:
        return "Unknown"
    return cats[0]


def _sanitize_label(label: str) -> str:
    """
    Make label safe for filesystem use (simple version).
    """
    if not label:
        return "Unknown"
    # Replace slashes and spaces to avoid path issues
    label = label.replace("/", "-").replace("\\", "-")
    label = label.strip().replace(" ", "_")
    # Truncate to something reasonable just in case
    if len(label) > 64:
        label = label[:64]
    return label or "Unknown"


# ---------------------------------------------------------------------
# Sample → face + labels
# ---------------------------------------------------------------------

def process_sample_to_face(
    sample,
    detector,
    root="emotic",
    person_index=0,
    face_size=128,
    min_face_score=0.5,
):
    # 1) load full image
    img_path = get_image_path(sample, root=root)
    img = load_image(img_path)

    # 2) body crop
    person = get_person(sample, index=person_index)
    body_bbox = get_body_bbox(person)
    body_crop = crop_bbox(img, body_bbox)

    # 3) detect face in body crop
    face_bbox = detect_face_in_body(body_crop, detector, min_score=min_face_score)
    if face_bbox is None:
        return None, None

    # 4) crop & resize
    face_crop = crop_bbox(body_crop, face_bbox)
    face_resized = resize_for_model(face_crop, size=face_size)

    # 5) labels
    all_cats = _extract_categories(person)
    dominant = _get_dominant_category(person)

    label_dict = {
        "dominant": dominant,
        "all": all_cats,
    }

    return face_resized, label_dict


# ---------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------

def build_face_dataset(
    data,
    output_root,
    detector,
    root="emotic",
    face_size=128,
    max_samples=None,
    min_face_score=0.5,
    csv_name="labels.csv",
):
    """
    Iterate over EMOTIC `data` (e.g., train_data) and save cropped faces.

    - data: EMOTIC dataset object (indexable)
    - output_root: root folder where faces+labels will be stored
    - detector: MediaPipe face detector created with create_face_detector()
    - max_samples: optional cap for debugging (None = all)

    Saves images as:
        output_root/<dominant_label>/sample_<idx>_p0.jpg

    Also writes a CSV mapping:
        filepath, dominant, all_labels
    """
    os.makedirs(output_root, exist_ok=True)

    n = len(data) if max_samples is None else min(len(data), max_samples)
    saved = 0
    skipped = 0

    records = []

    for i in range(n):
        sample = data[i]

        try:
            face_resized, label_info = process_sample_to_face(
                sample,
                detector,
                root=root,
                person_index=0,
                face_size=face_size,
                min_face_score=min_face_score,
            )
        except Exception as e:
            print(f"[{i}] Error processing sample: {e}")
            skipped += 1
            continue

        if face_resized is None or label_info is None:
            skipped += 1
            continue

        dominant = label_info["dominant"]
        all_labels = label_info["all"]

        dominant_safe = _sanitize_label(dominant)

        # Folder per dominant label
        label_dir = Path(output_root) / dominant_safe
        label_dir.mkdir(parents=True, exist_ok=True)

        out_path = label_dir / f"sample_{i:06d}_p0.jpg"

        # cv2.imwrite expects BGR
        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_bgr)
        saved += 1

        records.append(
            (
                str(out_path),
                dominant,
                ";".join(all_labels),
            )
        )

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{n} samples  |  saved: {saved}, skipped: {skipped}")

    print(f"Done. Saved {saved} faces, skipped {skipped}.")

    # Write CSV
    import csv

    csv_path = Path(output_root) / csv_name
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "dominant", "all_labels"])
        writer.writerows(records)

    print(f"Wrote label CSV to {csv_path}")