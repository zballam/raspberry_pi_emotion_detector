import time
from pathlib import Path

import cv2
import numpy as np
import torch
from picamera2 import Picamera2

from train_model import build_model, get_transforms
from utility.face_dataloader import EmoticFaceDataset


# --------------------------------------------------
# Paths / constants
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"   # weighted ckpt
MODEL_NAME = "tinycnn"
LABEL_COLUMN = "coarse_label"


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
# Label mapping (exactly as on Mac)
# --------------------------------------------------
def load_label_mapping():
    base_ds = EmoticFaceDataset(
        csv_path=CSV_PATH,
        label_column=LABEL_COLUMN,
        transform=None,
        root_dir=None,
    )
    return base_ds.idx_to_label  # {idx: label_name}


idx_to_label = load_label_mapping()
num_classes = len(idx_to_label)
print("Num classes:", num_classes)

label_list = [idx_to_label[i] for i in range(num_classes)]
label_to_idx = {name: idx for idx, name in idx_to_label.items()}
print("Labels:", label_list)

GROUPS = {
    "Positive": ["happy", "surprised"],
    "Neutral":  ["neutral", "confused"],
    "Negative": ["sad", "angry"],
}


# --------------------------------------------------
# Model + transforms (same as Mac script)
# --------------------------------------------------
model, _ = build_model(MODEL_NAME, num_classes)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
print(f"Loaded model from {MODEL_PATH}")

transform = get_transforms(MODEL_NAME, is_train=False)


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Same preprocessing as your Mac webcam script:

      - BGR -> GRAY
      - [0, 1] float
      - replicate to 3 channels
      - CHW tensor
      - pass through get_transforms(...)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("float32") / 255.0  # H, W

    img = np.stack([gray, gray, gray], axis=2)  # H, W, 3
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W

    tensor = transform(tensor)
    return tensor.unsqueeze(0)  # (1, C, H, W)


# --------------------------------------------------
# Face detection / cropping (same logic, no drawing)
# --------------------------------------------------
try:
    FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print("⚠️ Could not load Haar cascade, will use full frame.")
        face_cascade = None
    else:
        print("✅ Loaded face cascade from:", FACE_CASCADE_PATH)
except Exception as e:
    print("⚠️ Error loading face cascade:", e)
    face_cascade = None


def detect_and_crop_face(frame_bgr: np.ndarray):
    """
    Detect the largest face.
    Returns:
      - face_crop (BGR)
      - bbox (x1, y1, x2, y2) or None
    """
    if face_cascade is None:
        return frame_bgr, None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
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
# Main loop (PiCamera2, text-only)
# --------------------------------------------------
def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # warmup

    print("✅ Live Emotion Detector running (Pi Camera, text-only).")
    print("Press Ctrl+C in this terminal to stop.\n")

    frame_count = 0
    fps = 0.0
    last_time = time.time()

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if frame_count == 0:
                print("Frame min/max:", frame_bgr.min(), frame_bgr.max())

            face_bgr, _ = detect_and_crop_face(frame_bgr)
            if face_bgr is None or face_bgr.size == 0:
                face_bgr = frame_bgr

            # FPS
            frame_count += 1
            now = time.time()
            dt = now - last_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
            last_time = now

            # Throttle a bit
            if frame_count % 3 != 0:
                continue

            input_tensor = preprocess_frame(face_bgr).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            best_idx = int(probs.argmax())
            best_label = label_list[best_idx]

            # Coarse group scores
            group_scores = {}
            for group_name, labels in GROUPS.items():
                idxs = [label_to_idx[l] for l in labels if l in label_to_idx]
                group_scores[group_name] = float(probs[idxs].sum()) if idxs else 0.0

            pred_group = max(group_scores, key=group_scores.get)

            # Raw debug (same style you liked)
            print(
                "[RAW] best={} ({})".format(
                    best_label,
                    ", ".join(
                        f"{name}={probs[label_to_idx[name]]:.2f}"
                        for name in label_list
                    ),
                )
            )
            print(
                "[GROUP] Pos={:.2f} | Neu={:.2f} | Neg={:.2f} -> {} (FPS ~ {:.1f})".format(
                    group_scores["Positive"],
                    group_scores["Neutral"],
                    group_scores["Negative"],
                    pred_group,
                    fps,
                )
            )

    except KeyboardInterrupt:
        print("\nStopping live emotion detector...")

    finally:
        picam2.stop()


if __name__ == "__main__":
    main()