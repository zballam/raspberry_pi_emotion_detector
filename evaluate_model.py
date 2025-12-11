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

# ============================================================
# Config
# ============================================================
CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"
LABEL_COLUMN = "coarse_label"
BATCH_SIZE = 64
CKPT_DIR = Path("models")

# Default list of models to evaluate.
# You can change this or call main() with your own list.
MODEL_NAMES = [
    "tinycnn",
    "mobilenetv2",
    "efficientnet_lite0",
    # add others here if you used different names in training
]

# ============================================================
# Device
# ============================================================
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ============================================================
# Helpers
# ============================================================
def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    model.load_state_dict(state)
    return model


def evaluate_single_model(
    model_name: str,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int,
    idx_to_label: dict[int, str],
) -> None:
    """
    Build model, load its checkpoint, run evaluation, and print metrics.
    """
    ckpt_path = CKPT_DIR / f"emotion_{model_name}_best.pt"

    if not ckpt_path.exists():
        print(f"\n[WARN] Checkpoint not found for '{model_name}': {ckpt_path}")
        print("       Skipping this model.\n")
        return

    print("=" * 80)
    print(f"Evaluating model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 80)

    # Build model and load weights
    model, _ = build_model(model_name, num_classes)
    model = load_checkpoint_into_model(model, ckpt_path)
    model.to(device)
    model.eval()

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

    # Label names in sorted index order
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]

    print("\nClassification report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_targets, all_preds))
    print("\n")  # spacing between models


# ============================================================
# Main
# ============================================================
def main(model_names: list[str] | None = None) -> None:
    """
    Evaluate one or more models on the SAME validation split.

    Parameters
    ----------
    model_names : list[str] | None
        List of model names (e.g. ["tinycnn", "mobilenetv2"]).
        If None, uses the MODEL_NAMES constant defined above.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    if not model_names:
        print("No model names provided. Nothing to evaluate.")
        return

    # Build dataloaders ONCE so that all models see the exact same val split
    _, val_loader, num_classes, *_ = make_dataloaders_single_csv(
        model_names[0],  # model name only affects transforms; dataset & split are fixed by seed
        csv_path=CSV_PATH,
        label_column=LABEL_COLUMN,
        batch_size=BATCH_SIZE,
    )

    # Dataset just for mapping indices -> labels
    base_ds = EmoticFaceDataset(
        CSV_PATH,
        label_column=LABEL_COLUMN,
        transform=None,
    )
    idx_to_label = base_ds.idx_to_label

    print(f"Device: {device}")
    print(f"CSV: {CSV_PATH}")
    print(f"Label column: {LABEL_COLUMN}")
    print(f"Models to evaluate: {model_names}\n")

    for name in model_names:
        evaluate_single_model(name, val_loader, num_classes, idx_to_label)


if __name__ == "__main__":
    # Change MODEL_NAMES at the top or call main([...]) manually.
    main()