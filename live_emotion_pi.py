import cv2
import torch
import numpy as np
from pathlib import Path

from picamera2 import Picamera2   # NEW

from utility.model_loader import build_model, get_transforms


# --------------------------------------------------
# Paths & basic config
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"
MODEL_NAME = "tinycnn"

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
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("float32") / 255.0
    img = np.stack([gray, gray, gray], axis=2)
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    tensor = transform(tensor)
    return tensor.unsqueeze(0)


# --------------------------------------------------
# Face detection / cascade loading
# --------------------------------------------------
def _find_haar_cascade():
    cascade_name = "haarcascade_frontalface_default.xml"
    candidates = []

    if hasattr(cv2, "data"):
        try:
            candidates.append(Path(cv2.data.haarcascades) / cascade_name)
        except Exception:
            pass

    candidates.extend([
        Path("/usr/share/opencv4/haarcascades") / cascade_name,
        Path("/usr/share/opencv/haarcascades") / cascade_name,
    ])

    for p in candidates:
        if p.is_file():
            return str(p)
    return None


try:
    cascade_path = _find_haar_cascade()
    if cascade_path is None:
        print("⚠️ Could not find Haar cascade file, will use full frame.")
        face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"⚠️ Failed to load cascade from {cascade_path}, will use full frame.")
            face_cascade = None
        else:
            print("✅ Loaded face cascade from:", cascade_path)
except Exception as e:
    print("⚠️ Error loading face cascade:", e)
    face_cascade = None


def detect_and_crop_face(frame_bgr: np.ndarray):
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
# Main loop using Picamera2
# --------------------------------------------------
def main():
    # Initialize Pi camera through Picamera2
    picam2 = Picamera2()

    # Simple 640x480 RGB config for speed
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("✅ Live Emotion Detector running (Pi Camera). Press 'q' to quit.")

    try:
        while True:
            # Grab frame as RGB
            frame_rgb = picam2.capture_array()  # shape (H, W, 3), RGB
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Face detection + crop
            face_bgr, bbox = detect_and_crop_face(frame_bgr)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Preprocess + inference
            input_tensor = preprocess_frame(face_bgr).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            group_scores = {}
            for group_name, labels in GROUPS.items():
                idxs = [label_to_idx[l] for l in labels if l in label_to_idx]
                group_scores[group_name] = float(probs[idxs].sum()) if idxs else 0.0

            pred_group = max(group_scores, key=group_scores.get)

            print(
                f"[GROUP] Pos={group_scores['Positive']:.2f} | "
                f"Neu={group_scores['Neutral']:.2f} | "
                f"Neg={group_scores['Negative']:.2f} -> {pred_group}"
            )

            cv2.putText(
                frame_bgr,
                f"{pred_group}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Live Emotion Detector (PiCam)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()