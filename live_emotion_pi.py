# live_emotion_pi.py
#
# Live emotion detector using your existing TinyCNN pipeline.
# - Reuses build_model + get_transforms from train_model.py
# - Reuses EmoticFaceDataset label mapping from face_dataloader.py
# - Uses OpenCV VideoCapture for webcam / Pi camera (if exposed as /dev/video0)

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
    Take a BGR OpenCV frame, return a 1x3xHxW tensor ready for the model.
    (We treat the full frame as the "face" for now.)
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

        # Preprocess frame
        input_tensor = preprocess_frame(frame).to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = logits.argmax(1).item()
            pred_label = idx_to_label[pred_idx]

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