# live_emotion_pi.py
#
# Raspberry Pi Live Emotion Detector (Picamera2 preferred)
# MATCHES Mac inference pipeline:
# - Uses EmoticFaceDataset to load idx_to_label (label order)
# - Uses get_transforms() from train_model (same resize/normalize)
# - Preprocess = BGR -> GRAY -> replicate to 3 channels -> transform
# - Groups = sum probs into Positive/Neutral/Negative like Mac script
# - Prints only group + Pos/Neu/Neg with small separation

import time
from pathlib import Path

import numpy as np
import cv2
import torch

from train_model import build_model, get_transforms
from utility.face_dataloader import EmoticFaceDataset

# Try to import Picamera2 (preferred on Raspberry Pi)
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA2 = True
except ImportError:
    Picamera2 = None
    HAVE_PICAMERA2 = False

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"

MODEL_NAME = "tinycnn"
LABEL_COLUMN = "coarse_label"

# Groups (same as Mac)
GROUPS = {
    "Positive": ["happy", "surprised"],
    "Neutral":  ["neutral", "confused"],
    "Negative": ["angry", "sad"],
}

# --------------------------------------------------
# Device
# --------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# --------------------------------------------------
# Load label mapping (EXACT order used in training code)
# --------------------------------------------------
def load_label_mapping():
    ds = EmoticFaceDataset(
        csv_path=str(CSV_PATH),
        label_column=LABEL_COLUMN,
        transform=None,
        root_dir=None,
    )
    return ds.idx_to_label  # {idx: label_name}

idx_to_label = load_label_mapping()
num_classes = len(idx_to_label)
label_list = [idx_to_label[i] for i in range(num_classes)]
label_to_idx = {label_list[i]: i for i in range(num_classes)}

print("Num classes:", num_classes)
print("Labels (idx order):", label_list)

# --------------------------------------------------
# Model loading (robust to different checkpoint formats)
# --------------------------------------------------
def load_model():
    res = build_model(MODEL_NAME, num_classes)
    model = res[0] if isinstance(res, tuple) else res
    model = model.to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    # Handle ckpt wrappers if present
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    return model

model = load_model()

# --------------------------------------------------
# Transforms (EXACT same as Mac)
# --------------------------------------------------
transform = get_transforms(MODEL_NAME, is_train=False)

def preprocess_frame_like_mac(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    EXACT Mac pipeline:
      BGR -> GRAY -> replicate to 3 channels -> CHW tensor -> get_transforms -> add batch
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    img = np.stack([gray, gray, gray], axis=2)               # HWC (3ch gray)
    tensor = torch.from_numpy(img).permute(2, 0, 1)          # CHW
    tensor = transform(tensor)                               # resize/normalize like training
    return tensor.unsqueeze(0)                               # 1xCxHxW

# --------------------------------------------------
# Face detection / cropping (same strategy as your Mac script)
# --------------------------------------------------
def load_face_cascade():
    # Pi sometimes lacks cv2.data, so check common paths
    candidates = []
    try:
        candidates.append(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    except Exception:
        pass

    candidates += [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]

    for p in candidates:
        if p and Path(p).exists():
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                print("✅ Loaded face cascade from:", p)
                return c

    print("⚠️ Could not load Haar cascade, will use full frame.")
    return None

face_cascade = load_face_cascade()

def detect_and_crop_face(frame_bgr: np.ndarray):
    """
    Detect largest face, return (face_crop, bbox) else (frame_bgr, None)
    """
    if face_cascade is None:
        return frame_bgr, None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return frame_bgr, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = int(0.15 * h)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2, y2)

# --------------------------------------------------
# Grouping (same as Mac script)
# --------------------------------------------------
def group_scores_from_probs(probs: np.ndarray):
    scores = {}
    for gname, labels in GROUPS.items():
        idxs = [label_to_idx[l] for l in labels if l in label_to_idx]
        scores[gname] = float(probs[idxs].sum()) if idxs else 0.0
    pred_group = max(scores, key=scores.get)
    return scores, pred_group

# --------------------------------------------------
# Camera capture (Picamera2 preferred)
# --------------------------------------------------
def open_camera():
    if HAVE_PICAMERA2:
        print("🎥 Using Picamera2 backend.")
        picam2 = Picamera2()
        # Keep this modest for CPU inference speed
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.3)
        return ("picamera2", picam2)

    print("🎥 Picamera2 not available, falling back to OpenCV VideoCapture(0).")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.3)
    if not cap.isOpened():
        return (None, None)
    return ("opencv", cap)

# --------------------------------------------------
# Main loop (text-only output once per second)
# --------------------------------------------------
def main():
    backend, cam = open_camera()
    if backend is None:
        print("❌ Could not open camera.")
        return

    print("✅ Live Emotion Detector running (text-only). Ctrl+C to stop.")
    start_time = time.time()
    last_print = start_time
    frames = 0

    try:
        while True:
            # --- Read frame ---
            if backend == "picamera2":
                frame_rgb = cam.capture_array()
                if frame_rgb is None:
                    continue
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame_bgr = cam.read()
                if not ret or frame_bgr is None:
                    break

            frames += 1

            # --- Face crop ---
            face_bgr, _ = detect_and_crop_face(frame_bgr)

            # --- Preprocess EXACT like Mac ---
            x = preprocess_frame_like_mac(face_bgr).to(device)

            # --- Inference ---
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            scores, pred_group = group_scores_from_probs(probs)

            # --- Print once per second ---
            now = time.time()
            if now - last_print >= 1.0:
                fps = frames / (now - start_time)

                # “small separation between three groups”
                print(
                    f"{pred_group:8s}  ||  "
                    f"Pos {scores['Positive']:.2f}  |  "
                    f"Neu {scores['Neutral']:.2f}  |  "
                    f"Neg {scores['Negative']:.2f}   (FPS~{fps:.1f})"
                )
                last_print = now

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if backend == "picamera2":
            cam.stop()
        else:
            cam.release()

if __name__ == "__main__":
    main()