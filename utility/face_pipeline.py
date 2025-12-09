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
    Run MediaPipe on a body crop and return (x1, y1, x2, y2) in *body-crop* pixels.
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


def process_sample_to_face(
    sample,
    detector,
    root="emotic",
    person_index=0,
    face_size=128,
    min_face_score=0.5,
):
    """
    Given an EMOTIC sample, returns:
        - face_resized (face_size x face_size x 3) or None
        - label_dict (placeholder for emotion labels in future)
    Does:
        sample -> image -> body crop -> face bbox -> face crop -> resize.
    """

    # 1) load full image
    img_path = get_image_path(sample, root=root)
    img = load_image(img_path)

    # 2) body crop for chosen person
    person = get_person(sample, index=person_index)
    body_bbox = get_body_bbox(person)
    body_crop = crop_bbox(img, body_bbox)

    # 3) detect face in body crop
    face_bbox = detect_face_in_body(body_crop, detector, min_score=min_face_score)
    if face_bbox is None:
        return None, None  # no face

    # 4) crop & resize
    face_crop = crop_bbox(body_crop, face_bbox)
    face_resized = resize_for_model(face_crop, size=face_size)

    label_dict = {}

    return face_resized, label_dict


def build_face_dataset(
    data,
    output_root,
    detector,
    root="emotic",
    face_size=128,
    max_samples=None,
    min_face_score=0.5,
):
    """
    Iterate over EMOTIC `data` (e.g., train_data) and save cropped faces.

    - data: EMOTIC dataset object (indexable)
    - output_root: folder where faces will be stored
    - detector: MediaPipe face detector created with create_face_detector()
    - max_samples: optional cap for debugging (None = all)
    - Currently saves faces as:
        output_root/sample_<idx>_p0.jpg
      You can later change this to include label subfolders.
    """
    os.makedirs(output_root, exist_ok=True)

    n = len(data) if max_samples is None else min(len(data), max_samples)
    saved = 0
    skipped = 0

    for i in range(n):
        sample = data[i]

        try:
            face_resized, _ = process_sample_to_face(
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

        if face_resized is None:
            skipped += 1
            continue

        out_path = Path(output_root) / f"sample_{i:06d}_p0.jpg"
        # cv2.imwrite expects BGR
        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_bgr)
        saved += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{n} samples  |  saved: {saved}, skipped: {skipped}")

    print(f"Done. Saved {saved} faces, skipped {skipped}.")