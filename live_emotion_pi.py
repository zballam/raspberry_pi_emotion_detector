# live_emotion_pi.py

import cv2
import torch
import numpy as np
from pathlib import Path

from train_model import build_model, get_transforms
from utility.face_dataloader import EmoticFaceDataset


# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"

MODEL_NAME = "tinycnn"          # we’re using tinycnn for real-time
LABEL_COLUMN = "coarse_label"   # same as training


# --------------------------------------------------
# Device (Pi will end up on CPU, Mac on mps)
# --------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# Load label mapping (reuses your dataset code)
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

# Make a stable list of labels ordered by index for nicer debug prints
label_list = [idx_to_label[i] for i in range(num_classes)]


# --------------------------------------------------
# Build model + load weights (reuses your pipeline)
# --------------------------------------------------
model, _ = build_model(MODEL_NAME, num_classes)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
print(f"Loaded model from {MODEL_PATH}")


# --------------------------------------------------
# Preprocessing (same as training, via get_transforms)
# --------------------------------------------------
transform = get_transforms(MODEL_NAME, is_train=False)


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Take a BGR OpenCV frame (ideally a face crop), return a 1x3xHxW tensor.
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # HWC float32 in [0,1]
    img = img_rgb.astype("float32") / 255.0

    # -> CHW tensor
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W

    # Resize + normalize (exactly like training)
    tensor = transform(tensor)

    # Add batch dimension
    return tensor.unsqueeze(0)


# --------------------------------------------------
# Face detection / cropping
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
# Main webcam / Pi loop
# --------------------------------------------------
def main():
    # 0 = default webcam OR Pi camera if exposed as /dev/video0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera (VideoCapture(0) failed).")
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

        # Inference + debugging prints
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # shape [num_classes]
            pred_idx = int(probs.argmax())
            pred_label = idx_to_label[pred_idx]

        # --- DEBUGGING: print probabilities for each class ---
        prob_str = " | ".join(
            f"{label_list[i]}: {probs[i]:.3f}" for i in range(num_classes)
        )
        print(f"[DEBUG] {prob_str} -> pred = {pred_label}")

        # Draw label on frame
        cv2.putText(
            frame,
            f"{pred_label}",
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