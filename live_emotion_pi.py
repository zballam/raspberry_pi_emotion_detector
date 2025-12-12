# live_emotion_pi.py
#
# Live emotion detector for Raspberry Pi camera.
# - Uses TinyCNN trained on EMOTIC face crops (128x128).
# - Prefers Picamera2 for capturing frames (best for Pi 5).
# - Falls back to OpenCV VideoCapture(0) if Picamera2 is unavailable.
# - Text-only: prints one summary line ~once per second.

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

# Try to import Picamera2 (preferred on Raspberry Pi)
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA2 = True
except ImportError:
    Picamera2 = None
    HAVE_PICAMERA2 = False

# -----------------------------------------------------
# Config / constants
# -----------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"
MODEL_NAME = "tinycnn"

# These MUST match training label order
LABELS = ["angry", "confused", "happy", "neutral", "sad", "surprised"]
NUM_CLASSES = len(LABELS)

POSITIVE_LABELS = {"happy", "surprised"}
NEGATIVE_LABELS = {"angry", "sad"}
NEUTRAL_LABELS = {"confused", "neutral"}

# If True, also print full probability vector each second
VERBOSE = False

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
    """
    Handles both:
      - build_model(name, num_classes) -> model
      - build_model(name, num_classes) -> (model, something)
    """
    res = build_model(MODEL_NAME, NUM_CLASSES)
    if isinstance(res, tuple):
        model = res[0]
    else:
        model = res

    model = model.to(device)

    state = torch.load(MODEL_PATH, map_location=device)
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
    If unavailable, returns None and we fall back to center crop.
    """
    # Try cv2.data.haarcascades if available (not always on Pi)
    try:
        haar_dir = cv2.data.haarcascades
        candidate = Path(haar_dir) / "haarcascade_frontalface_default.xml"
        if candidate.exists():
            cascade = cv2.CascadeClassifier(str(candidate))
            if not cascade.empty():
                print(f"✅ Loaded face cascade from: {candidate}")
                return cascade
    except Exception as e:
        print(f"⚠️ Could not use cv2.data.haarcascades: {e}")

    # Common Raspberry Pi OpenCV paths
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

    print("⚠️ Face cascade not found; using full-frame center crop.")
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


def preprocess_face(frame_bgr, bbox, device):
    """
    Crop face (or center of frame), resize to 128x128, normalize like training.
    Returns a 1x3x128x128 tensor on the given device.
    """
    h, w, _ = frame_bgr.shape

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        # Center square crop if no face detected
        side = min(h, w)
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - side // 2)
        y2 = y1 + side
        x1 = max(0, cx - side // 2)
        x2 = x1 + side
        crop = frame_bgr[y1:y2, x1:x2]

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
# Group helper (simpler & less confusing)
# -----------------------------------------------------


def best_label_and_group(probs):
    """
    probs: 1D numpy array over LABELS.

    Returns:
        best_label (str), best_conf (float), group (str)
    where group is derived from the BEST label, not the sum.
    """
    best_idx = int(np.argmax(probs))
    best_label = LABELS[best_idx]
    best_conf = float(probs[best_idx])

    if best_label in POSITIVE_LABELS:
        group = "Positive"
    elif best_label in NEGATIVE_LABELS:
        group = "Negative"
    else:
        group = "Neutral"

    return best_label, best_conf, group


# -----------------------------------------------------
# Main loop
# -----------------------------------------------------


def main():
    model = load_model()
    face_cascade = load_face_cascade()

    start_time = time.time()
    last_print_time = start_time
    frame_count = 0

    last_probs = None

    # Backend selection
    using_picamera2 = False
    picam2 = None
    cap = None

    try:
        if HAVE_PICAMERA2:
            print("🎥 Using Picamera2 backend.")
            picam2 = Picamera2()
            # Small-ish resolution; RGB888 so we can convert to BGR
            config = picam2.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)
            using_picamera2 = True
        else:
            print("🎥 Picamera2 not available, falling back to OpenCV VideoCapture(0).")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            time.sleep(0.5)
            if not cap.isOpened():
                print("❌ Could not open camera with VideoCapture(0).")
                return

        print("✅ Live Emotion Detector running (Pi Camera, text-only).")
        print("Press Ctrl+C in this terminal to stop.")

        while True:
            # --- Get a frame ---
            if using_picamera2:
                # Picamera2 gives RGB; convert to BGR for OpenCV-style processing
                frame_rgb = picam2.capture_array()
                if frame_rgb is None:
                    print("⚠️ Picamera2 returned no frame.")
                    continue
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("⚠️ Could not read frame from camera.")
                    break

            frame_count += 1

            # Only print min/max once at the beginning as a sanity check
            if frame_count == 1:
                mn, mx = frame.min(), frame.max()
                print(f"Frame min/max: {mn} {mx}")

            # --- Face / crop selection ---
            bbox = choose_face_bbox(frame, face_cascade)

            # --- Inference ---
            with torch.no_grad():
                tensor = preprocess_face(frame, bbox, device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            last_probs = probs

            # --- Print once per second ---
            now = time.time()
            if now - last_print_time >= 1.0 and last_probs is not None:
                elapsed = now - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                best_label, best_conf, group = best_label_and_group(last_probs)

                # If model isn't confident, surface that explicitly
                if best_conf < 0.40:
                    group_display = "Uncertain"
                else:
                    group_display = group

                # Clean single-line summary
                print(
                    f"[EMOTION] {best_label.upper():9s} | "
                    f"Group={group_display:9s} | "
                    f"Conf={best_conf:.2f} | FPS~{fps:.1f}"
                )

                # Optional: full class probabilities (toggle at top)
                if VERBOSE:
                    class_str = " ".join(
                        f"{lbl}={last_probs[i]:.2f}" for i, lbl in enumerate(LABELS)
                    )
                    print(f"          probs: {class_str}")

                last_print_time = now

    except KeyboardInterrupt:
        print("\nStopping live emotion detector...")

    finally:
        if using_picamera2 and picam2 is not None:
            picam2.stop()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()