# live_emotion_pi.py
#
# Lightweight, text-only live emotion detector for Raspberry Pi camera.
# - Uses TinyCNN trained on EMOTIC face crops (128x128).
# - Prints one summary line ~once per second (not every frame).
# - If Haar cascade is unavailable, uses a centered crop of the frame.

import time
from pathlib import Path

import numpy as np
import cv2
import torch

from train_model import (
    build_model,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# -----------------------------------------------------
# Paths & constants
# -----------------------------------------------------

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"
MODEL_NAME = "tinycnn"

# These MUST match the training order
LABELS = ["angry", "confused", "happy", "neutral", "sad", "surprised"]
NUM_CLASSES = len(LABELS)

POSITIVE_LABELS = {"happy", "surprised"}
NEGATIVE_LABELS = {"angry", "sad"}
NEUTRAL_LABELS = {"confused", "neutral"}

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}

# -----------------------------------------------------
# Device
# -----------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# -----------------------------------------------------
# Model loading
# -----------------------------------------------------


def load_model():
    model = build_model(MODEL_NAME, NUM_CLASSES).to(device)
    state = torch.load(MODEL_PATH, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    print("Num classes:", NUM_CLASSES)
    print("Labels:", LABELS)
    return model


# -----------------------------------------------------
# Face detection / preprocessing
# -----------------------------------------------------


def load_face_cascade():
    """
    Try to load a frontal face Haar cascade.
    If unavailable on Pi, returns None and we fall back to full-frame crop.
    """
    # 1) Try cv2.data.haarcascades if available
    try:
        haar_dir = cv2.data.haarcascades  # may not exist on Pi
        candidate = Path(haar_dir) / "haarcascade_frontalface_default.xml"
        if candidate.exists():
            cascade = cv2.CascadeClassifier(str(candidate))
            if not cascade.empty():
                print(f"✅ Loaded face cascade from: {candidate}")
                return cascade
    except Exception as e:
        print(f"⚠️ Could not use cv2.data.haarcascades: {e}")

    # 2) Common Raspberry Pi OpenCV paths
    candidates = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            cascade = cv2.CascadeClassifier(str(p))
            if not cascade.empty():
                print(f"✅ Loaded face cascade from: {p}")
                return cascade

    print("⚠️ Face cascade not found; using full-frame crop.")
    return None


def choose_face_bbox(frame, face_cascade):
    """
    If we have a cascade, detect faces & return the largest bbox.
    Otherwise, return None (caller will use center crop).
    """
    if face_cascade is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Choose largest face by area
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return (x, y, x + w, y + h)


def preprocess_face(frame, bbox, device):
    """
    Crop face (or center of frame), resize to 128x128, normalize like training.
    Returns a 1x3x128x128 tensor on the given device.
    """
    h, w, _ = frame.shape

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop = frame[y1:y2, x1:x2]
    else:
        # Center crop square if no face detected
        side = min(h, w)
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - side // 2)
        y2 = y1 + side
        x1 = max(0, cx - side // 2)
        x2 = x1 + side
        crop = frame[y1:y2, x1:x2]

    # BGR -> RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (128, 128))

    # HWC -> CHW, float32, [0,1]
    img = crop_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W

    tensor = torch.from_numpy(img)

    # Normalize like training (ImageNet stats)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    tensor = tensor.unsqueeze(0).to(device)  # 1x3x128x128
    return tensor


# -----------------------------------------------------
# Grouping helper
# -----------------------------------------------------


def group_probs(probs):
    """
    probs: 1D numpy array over LABELS (length 6).
    Returns (pos, neu, neg, best_group_label)
    """
    pos = sum(probs[LABEL_TO_IDX[lbl]] for lbl in POSITIVE_LABELS)
    neg = sum(probs[LABEL_TO_IDX[lbl]] for lbl in NEGATIVE_LABELS)
    neu = sum(probs[LABEL_TO_IDX[lbl]] for lbl in NEUTRAL_LABELS)

    if pos >= neu and pos >= neg:
        group = "Positive"
    elif neg >= pos and neg >= neu:
        group = "Negative"
    else:
        group = "Neutral"

    return float(pos), float(neu), float(neg), group


# -----------------------------------------------------
# Main loop
# -----------------------------------------------------


def main():
    model = load_model()
    face_cascade = load_face_cascade()

    # Open Pi camera via OpenCV (libcamera backend)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("❌ Could not open camera (index 0).")
        return

    print("✅ Live Emotion Detector running (Pi Camera, text-only).")
    print("Press Ctrl+C in this terminal to stop.")

    start_time = time.time()
    last_print_time = start_time
    frame_count = 0

    last_probs = None
    last_group = None
    last_group_scores = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Could not read frame from camera.")
                break

            frame_count += 1

            # Optional: quick diagnostic
            if frame_count == 1:
                mn, mx = frame.min(), frame.max()
                print(f"Frame min/max: {mn} {mx}")

            bbox = choose_face_bbox(frame, face_cascade)

            with torch.no_grad():
                tensor = preprocess_face(frame, bbox, device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Store last for printing
            last_probs = probs
            pos, neu, neg, group = group_probs(probs)
            last_group = group
            last_group_scores = (pos, neu, neg)

            # Print about once per second, not every frame
            now = time.time()
            if now - last_print_time >= 1.0 and last_probs is not None:
                elapsed = now - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                best_idx = int(np.argmax(last_probs))
                best_label = LABELS[best_idx]

                # Raw per-class line
                class_str = ", ".join(
                    f"{lbl}={last_probs[i]:.2f}" for i, lbl in enumerate(LABELS)
                )
                print(f"[RAW] best={best_label} ({class_str})")

                pos, neu, neg = last_group_scores
                print(
                    f"[GROUP] Pos={pos:.2f} | Neu={neu:.2f} | Neg={neg:.2f} "
                    f"-> {last_group} (FPS ~ {fps:.1f})"
                )

                last_print_time = now

    except KeyboardInterrupt:
        print("\nStopping live emotion detector...")

    finally:
        cap.release()


if __name__ == "__main__":
    main()