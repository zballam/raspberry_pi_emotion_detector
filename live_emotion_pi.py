import cv2
import torch
import numpy as np
from pathlib import Path

from utility.model_loader import build_model, get_transforms


# --------------------------------------------------
# Paths & basic config
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"  # weighted TinyCNN checkpoint
MODEL_NAME = "tinycnn"  # using tinycnn for real-time on Pi

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

# Coarse groups for display
GROUPS = {
    "Positive": ["happy", "surprised"],
    "Neutral": ["neutral", "confused"],
    "Negative": ["sad", "angry"],
}


# --------------------------------------------------
# Device (Pi will end up on CPU)
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

# Transforms for inference (defined in utility/model_loader.py)
transform = get_transforms()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Take a BGR OpenCV frame (ideally a face crop), return a 1x3xHxW tensor.
    We convert to GRAYSCALE and then replicate to 3 channels, to match
    the grayscale-3channel training pipeline.
    """
    # BGR -> GRAY
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # H,W float32 in [0,1]
    gray = gray.astype("float32") / 255.0

    # Replicate into 3 channels: H,W -> H,W,3
    img = np.stack([gray, gray, gray], axis=2)

    # -> CHW tensor
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W

    # Resize + normalize (exactly like training)
    tensor = transform(tensor)

    # Add batch dimension: 1,C,H,W
    return tensor.unsqueeze(0)


# --------------------------------------------------
# Face detection / cascade loading
# --------------------------------------------------
def _find_haar_cascade():
    """
    Try several common locations for the frontal face cascade
    on Raspberry Pi / Debian OpenCV builds.
    """
    cascade_name = "haarcascade_frontalface_default.xml"
    candidates = []

    # Some builds have cv2.data, some don't
    if hasattr(cv2, "data"):
        try:
            candidates.append(Path(cv2.data.haarcascades) / cascade_name)
        except Exception:
            pass

    # Common Debian/RPi OpenCV locations
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
    """
    Detect the largest face in the frame and return:
      - face_crop (BGR)
      - bbox (x1, y1, x2, y2) or None if no face found

    If detection fails or cascade isn't loaded, returns (frame_bgr, None).
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

    # Pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Optional: add a bit of padding around the face
    pad = int(0.15 * h)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2, y2)


# --------------------------------------------------
# Camera selection helper
# --------------------------------------------------
def open_any_camera():
    """
    Try several /dev/videoN devices and backends, return an opened VideoCapture
    or None if all fail.
    """
    # Try a handful of indices; you have many /dev/video* nodes
    indices_to_try = [0, 1, 2, 3, 4, 5]

    for idx in indices_to_try:
        # First try V4L2 explicitly
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"✅ Opened camera /dev/video{idx} with CAP_V4L2")
            return cap

        # Fallback: let OpenCV pick the backend
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"✅ Opened camera /dev/video{idx} with default backend")
            return cap

    print("❌ Could not open any camera from indices:", indices_to_try)
    return None


# --------------------------------------------------
# Main webcam / Pi loop
# --------------------------------------------------
def main():
    cap = open_any_camera()
    if cap is None:
        return

    print("✅ Live Emotion Detector running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, exiting.")
            break

        # --- Face detection + cropping ---
        face_bgr, bbox = detect_and_crop_face(frame)

        # Draw bbox on the original frame if we found a face
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Preprocess (on the face crop if available, otherwise full frame)
        input_tensor = preprocess_frame(face_bgr).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # [num_classes]

        # ---- Map 6-class probs -> 3 coarse groups ----
        group_scores = {}
        for group_name, labels in GROUPS.items():
            idxs = [label_to_idx[l] for l in labels if l in label_to_idx]
            if idxs:
                group_scores[group_name] = float(probs[idxs].sum())
            else:
                group_scores[group_name] = 0.0

        pred_group = max(group_scores, key=group_scores.get)

        # Debug print
        print(
            f"[GROUP] Pos={group_scores['Positive']:.2f} | "
            f"Neu={group_scores['Neutral']:.2f} | "
            f"Neg={group_scores['Negative']:.2f} -> {pred_group}"
        )

        # Draw coarse label on frame
        cv2.putText(
            frame,
            f"{pred_group}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Show the frame
        cv2.imshow("Live Emotion Detector", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()