# test.py

from pathlib import Path
import numpy as np
import torch

from utility.face_dataloader import EmoticFaceDataset
from train_model import build_model, get_transforms

# Use CPU for simple testing
device = torch.device("cpu")

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "emotic_faces_128" / "labels_coarse.csv"
MODEL_PATH = ROOT / "models" / "emotion_tinycnn_best.pt"

MODEL_NAME = "tinycnn"
LABEL_COLUMN = "coarse_label"


def main():
    # Load dataset without transforms so we can re-apply eval transforms
    base_ds = EmoticFaceDataset(
        csv_path=CSV_PATH,
        label_column=LABEL_COLUMN,
        transform=None,
        root_dir=None,
    )

    num_classes = len(base_ds.label_to_idx)
    idx_to_label = base_ds.idx_to_label  # {idx: label_name}

    # Build model and load weights
    model, _ = build_model(MODEL_NAME, num_classes)
    state = torch.load(MODEL_PATH, map_location=device)

    # Support both plain state_dict and wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    transform = get_transforms(MODEL_NAME, is_train=False)

    print("Testing on first 20 samples from the CSV...")
    for i in range(20):
        img_tensor, label_tensor = base_ds[i]

        x = transform(img_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        true_label = idx_to_label[int(label_tensor)]
        pred_idx = int(probs.argmax())
        pred_label = idx_to_label[pred_idx]

        print(
            f"True = {true_label:9s}, pred = {pred_label:9s}, "
            f"probs = {np.round(probs, 3)}"
        )


if __name__ == "__main__":
    main()