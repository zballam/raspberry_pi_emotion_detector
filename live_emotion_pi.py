#!/usr/bin/env python3
"""
Live Emotion Detector for Raspberry Pi (Pi Camera, text-only console output).

- Loads TinyCNN trained on EMOTIC faces.
- Captures frames from the Pi camera using Picamera2.
- Optionally uses a Haar cascade to detect faces; falls back to center crop.
- Prints a compact, updating status line with the current prediction.
- Periodically prints detailed debug info every N frames.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from picamera2 import Picamera2

from train_model import build_model
from utility.face_dataloader import EmoticFaceDataset

# =====================================================
# Paths & device
# =====================================================

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"
IMG_ROOT = ROOT / "data" / "emotic_faces_128"
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# =====================================================
# Load dataset just to get labels / num classes
# =====================================================

dataset = EmoticFaceDataset(
    csv_path=str(CSV_PATH),
    root_dir=str(IMG_ROOT),
    label_column="coarse_label",
    transform=None,
)

num_classes = dataset.num_classes
# EmoticFaceDataset should have idx_to_label mapping; if it's a dict, sort by key
idx_to_label = dataset.idx_to_label
if isinstance(idx_to_label, dict):
    labels = [idx_to_label[i] for i in range(num_classes)]
else:
    labels = list(idx_to_label)

print(f"Num classes: {num_classes}")
print(f"Labels: {labels}")

# Grouping: map class indices into Positive / Neutral / Negative buckets
POSITIVE_LABELS = {"happy", "surprised"}
NEUTRAL_LABELS = {"neutral", "confused"}
NEGATIVE_LABELS = {"angry", "sad"}

positive_indices = [i for i, name in enumerate(labels) if name in POSITIVE_LABELS]
neutral_indices = [i for i, name in enumerate(labels) if name in NEUTRAL_LABELS]
negative_indices = [i for i, name in enumerate(labels) if name in NEGATIVE_LABELS]

# =====================================================
# Load model
# =====================================================

def load_checkpoint_into_model(model, ckpt_path):
    state = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    model.load_state_dict(state)
    return model

model = build_model("tinycnn", num_classes=num_classes).to(device)
model = load_checkpoint_into_model(model, str(MODEL_PATH))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# =====================================================
# Face cascade (optional)
# =====================================================

def load_face_cascade():
    cascade = None
    cascade_path = None

    try:
        # Try cv2.data.haarcascades if available
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidate = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(candidate):
                cascade_path = candidate

        # Try common Raspberry Pi / Debian OpenCV locations
        if cascade_path is None:
            candidates = [
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            ]
            for c in candidates:
                if os.path.exists(c):
                    cascade_path = c
                    break

        if cascade_path is not None:
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade is None or cascade.empty():
                cascade = None
                print(f"⚠️ Failed to load face cascade from: {cascade_path}")
            else:
                print(f"✅ Loaded face cascade from: {cascade_path}")
        else:
            print("⚠️ Could not find a Haar cascade file; using center crop only.")

    except Exception as e:
        print(f"⚠️ Error loading face cascade: {e}")
        cascade = None

    return cascade

face_cascade = load_face_cascade()

# =====================================================
# Preprocessing helpers
# =====================================================

TARGET_SIZE = (128, 128)

def crop_face_or_center(frame_rgb):
    """
    frame_rgb: H x W x 3, uint8, RGB
    Returns a cropped RGB image focusing on the face if possible,
    otherwise a center crop.
    """
    h, w, _ = frame_rgb.shape

    if face_cascade is not None:
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) > 0:
            # Choose the largest face
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(x + fw, w)
            y2 = min(y + fh, h)
            return frame_rgb[y1:y2, x1:x2, :]

    # Fallback: center square crop
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return frame_rgb[y1:y1 + side, x1:x1 + side, :]


def preprocess(frame_rgb):
    """
    frame_rgb: H x W x 3, RGB uint8
    Returns a torch tensor of shape [1, 3, 128, 128] on the correct device.
    """
    # Optional debug on the very first frame:
    # print("Frame min/max:", frame_rgb.min(), frame_rgb.max())

    cropped = crop_face_or_center(frame_rgb)
    resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # Convert to float32 in [0,1]
    img = resized.astype(np.float32) / 255.0

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # To tensor
    x = torch.from_numpy(img).unsqueeze(0)  # [1, 3, 128, 128]
    x = x.to(device)
    return x


def group_probs(probs: np.ndarray):
    """
    probs: [num_classes] numpy array
    Returns (pos, neu, neg, best_label_idx, best_label_name)
    """
    pos = float(probs[positive_indices].sum()) if positive_indices else 0.0
    neu = float(probs[neutral_indices].sum()) if neutral_indices else 0.0
    neg = float(probs[negative_indices].sum()) if negative_indices else 0.0

    best_idx = int(np.argmax(probs))
    best_label = labels[best_idx]

    return pos, neu, neg, best_idx, best_label


def group_label_from_triplet(pos, neu, neg):
    """
    Basic argmax over (pos, neu, neg) to decide Positive / Neutral / Negative.
    """
    group_names = ["Positive", "Neutral", "Negative"]
    idx = int(np.argmax([pos, neu, neg]))
    return group_names[idx]


# =====================================================
# Camera loop
# =====================================================

def main():
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("✅ Live Emotion Detector running (Pi Camera, text-only).")
    print("Press Ctrl+C in this terminal to stop.\n")

    # Debug / status settings
    DEBUG_EVERY = 120  # print detailed debug every N frames
    frame_idx = 0
    last_time = time.time()

    try:
        while True:
            frame_rgb = picam2.capture_array()  # RGB888
            frame_idx += 1

            # Optional single sanity print at the very start
            if frame_idx == 1:
                print(
                    f"Frame min/max: {frame_rgb.min()} {frame_rgb.max()}"
                )

            # Preprocess
            x = preprocess(frame_rgb)

            # Inference
            t0 = time.time()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            t1 = time.time()

            # Grouping
            pos, neu, neg, best_idx, best_label = group_probs(probs)
            group = group_label_from_triplet(pos, neu, neg)

            # FPS estimation (smoothed a bit)
            now = t1
            dt = now - last_time
            fps = 1.0 / dt if dt > 0 else 0.0
            last_time = now

            # Compact status line: overwrites itself each frame
            print(
                f"\r{group.upper():>8} "
                f"| best={best_label:9} "
                f"| Pos={pos:4.2f} Neu={neu:4.2f} Neg={neg:4.2f} "
                f"| FPS={fps:5.1f}",
                end="",
                flush=True,
            )

            # Periodic detailed debug
            if frame_idx % DEBUG_EVERY == 0:
                print("\n[RAW] best={} ({})".format(
                    best_label,
                    ", ".join(
                        f"{labels[i]}={probs[i]:.2f}" for i in range(len(labels))
                    ),
                ))
                print(
                    f"[GROUP] Pos={pos:.2f} | Neu={neu:.2f} | Neg={neg:.2f} -> {group}"
                )

    except KeyboardInterrupt:
        print("\nStopping live emotion detector...")

    finally:
        picam2.stop()


if __name__ == "__main__":
    main()