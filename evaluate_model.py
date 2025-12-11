from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from train_model import (
    make_dataloaders_single_csv,
    build_model,
)
from utility.face_dataloader import EmoticFaceDataset

CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"
MODEL_NAME = "tinycnn"  # or "mobilenetv2" or "tinycnn"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)


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


def main():
    # Recreate the same split (same seed inside make_dataloaders_single_csv)
    # NOTE: make_dataloaders_single_csv now returns extra stuff (e.g. class_weights),
    # so we use *_
    _, val_loader, num_classes, *_ = make_dataloaders_single_csv(
        MODEL_NAME,
        csv_path=CSV_PATH,
        label_column="coarse_label",
        batch_size=64,
    )

    # Build model and load weights
    model, _ = build_model(MODEL_NAME, num_classes)
    ckpt_path = Path("models") / f"emotion_{MODEL_NAME}_best.pt"
    model = load_checkpoint_into_model(model, ckpt_path)
    model.to(device)
    model.eval()

    # For label names
    base_ds = EmoticFaceDataset(CSV_PATH, label_column="coarse_label", transform=None)
    idx_to_label = base_ds.idx_to_label

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Replace indices with label names for printing
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

    print("Classification report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    print("Confusion matrix:")
    print(confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    main()