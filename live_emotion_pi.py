import time
from pathlib import Path

import cv2
import numpy as np
import torch
from picamera2 import Picamera2

from utility.model_loader import build_model, get_transforms


# --------------------------------------------------
# Paths & basic config
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"
MODEL_NAME = "tinycnn"

# === Static labels: make sure order matches training ===
label_list = [
    "happy",
    "surprised",
    "neutral",
    "confused",
    "sad",
    "angry",
]
num_classes = len(label_list)
label_to_idx = {name: i for i, name in enumerate(label_list)}

print("Num classes:", num_classes)
print("Labels:", label_list)

GROUPS = {
    "Positive": ["happy", "surprised"],
    "Neutral": ["neutral", "confused"],
    "Negative": ["sad", "angry"],
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
# Build model + load weights
# --------------------------------------------------
model = build_model(MODEL_NAME, num_classes)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
print(f"Loaded model from {MODEL_PATH}")

transform = get_transforms()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Take a BGR OpenCV frame (full frame for now), return a 1x3xHxW tensor.

    We:
      - convert BGR -> GRAY
      - normalize to [0, 1]
      - replicate to 3 channels (H, W, 3)
      - feed NumPy array into torchvision transforms (ToTensor / Resize / Normalize)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("float32") / 255.0
    img = np.stack([gray, gray, gray], axis=2).astype("float32")  # HWC
    tensor = transform(img)  # (C, H, W)
    return tensor.unsqueeze(0)  # (1, C, H, W)


# --------------------------------------------------
# (Optional) Face detection / cascade loading (not used right now)
# --------------------------------------------------
def _find_haar_cascade():
    cascade_name = "haarcascade_frontalface_default.xml"
    candidates = []

    if hasattr(cv2, "data"):
        try:
            candidates.append(Path(cv2.data.haarcascades) / cascade_name)
        except Exception:
            pass

    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades") / cascade_name,
            Path("/usr/share/opencv/haarcascades") / cascade_name,
        ]
    )

    for p in candidates:
        if p.is_file():
            return str(p)
    return None


try:
    cascade_path = _find_haar_cascade()
    if cascade_path is None:
        print("⚠️ Could not find Haar cascade file (not fatal).")
        face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"⚠️ Failed to load cascade from {cascade_path}.")
            face_cascade = None
        else:
            print("✅ Loaded face cascade from:", cascade_path)
except Exception as e:
    print("⚠️ Error loading face cascade:", e)
    face_cascade = None


# --------------------------------------------------
# Main loop using Picamera2 + FPS counter
# --------------------------------------------------
def main():
    picam2 = Picamera2()

    # Smaller resolution for speed
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # Warmup so exposure/brightness can settle
    time.sleep(1.0)

    print("✅ Live Emotion Detector running (Pi Camera). Press 'q' to quit.")

    frame_count = 0
    last_pred_group = "Neutral"
    fps = 0.0
    last_time = time.time()

    try:
        while True:
            # Grab frame as RGB from PiCam
            frame_rgb = picam2.capture_array()  # (H, W, 3), RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # One-time debug to prove frames aren't black
            if frame_count == 0:
                print("Frame min/max:", frame_bgr.min(), frame_bgr.max())

            # For now: run on the full frame (no face crop)
            face_bgr = frame_bgr

            # FPS update
            frame_count += 1
            now = time.time()
            dt = now - last_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
            last_time = now

            # Run inference only every 3rd frame
            do_infer = (frame_count % 3 == 0)

            if do_infer:
                input_tensor = preprocess_frame(face_bgr).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                group_scores = {}
                for group_name, labels in GROUPS.items():
                    idxs = [label_to_idx[l] for l in labels if l in label_to_idx]
                    group_scores[group_name] = float(probs[idxs].sum()) if idxs else 0.0

                last_pred_group = max(group_scores, key=group_scores.get)

                print(
                    f"[GROUP] Pos={group_scores['Positive']:.2f} | "
                    f"Neu={group_scores['Neutral']:.2f} | "
                    f"Neg={group_scores['Negative']:.2f} -> {last_pred_group} "
                    f"(FPS ~ {fps:.1f})"
                )

            # Brighten the frame for display so it doesn't look black
            display_bgr = cv2.convertScaleAbs(frame_bgr, alpha=1.5, beta=40)

            # Draw last prediction + FPS on every frame
            cv2.putText(
                display_bgr,
                f"{last_pred_group}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_bgr,
                f"FPS: {fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Live Emotion Detector (PiCam)", display_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()