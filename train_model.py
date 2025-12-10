# train_model.py

import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd

from utility.face_dataloader import EmoticFaceDataset
from utility.tinycnn import TinyCNN

# Optional: MobileNetV2 (needs torchvision)
try:
    from torchvision import models
    HAS_TORCHVISION = True
except Exception as e:
    print("Error importing torchvision:", repr(e))
    HAS_TORCHVISION = False

# Optional: EfficientNet-Lite (via timm)
try:
    import timm
    HAS_TIMM = True
except Exception as e:
    print("Error importing timm:", repr(e))
    HAS_TIMM = False


# ---------------------------
# Config
# ---------------------------

# 👉 Use the coarse 6-way labels
DATA_CSV = "data/emotic_faces_128/labels_coarse.csv"

BATCH_SIZE = 32

# TinyCNN: simple single-stage training
TINYCNN_EPOCHS = 10
TINYCNN_LR = 1e-3

# Pretrained models: 2-stage training
WARMUP_EPOCHS = 5       # classifier-only
FINETUNE_EPOCHS = 5     # full-network fine-tune
WARMUP_LR = 1e-3        # classifier head
FINETUNE_LR = 1e-5      # tiny LR for whole network

VAL_FRACTION = 0.2

# Choose which model to train: will be overridden by the loop at bottom
MODEL_NAME = "tinycnn"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", DEVICE)


# ---------------------------
# Model builders
# ---------------------------

def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    # Tiny CNN baseline
    if model_name == "tinycnn":
        print("Building TinyCNN")
        return TinyCNN(num_classes)

    # MobileNetV2
    if model_name == "mobilenetv2":
        if not HAS_TORCHVISION:
            raise RuntimeError("torchvision not installed; cannot use MobileNetV2.")

        print("Building MobileNetV2 (pretrained on ImageNet)")
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        return net

    # EfficientNet-Lite0 via timm
    if model_name == "efficientnet_lite0":
        if not HAS_TIMM:
            raise RuntimeError("timm not installed; cannot use EfficientNet-Lite0.")

        print("Building EfficientNet-Lite0 (pretrained on ImageNet)")
        net = timm.create_model(
            "efficientnet_lite0",
            pretrained=True,
            num_classes=num_classes,
        )
        return net

    raise ValueError(f"Unknown model_name: {model_name}")


# ---------------------------
# Training / evaluation helpers
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# ---------------------------
# Freezing helpers for pretrained models
# ---------------------------

def freeze_backbone_and_unfreeze_classifier(model, model_name: str):
    """Freeze all params, then unfreeze classifier only."""
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "mobilenetv2":
        # mobilenet.classifier is a Sequential
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name == "efficientnet_lite0":
        # timm EfficientNet-Lite0 has a .classifier layer
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"freeze_backbone_and_unfreeze_classifier not defined for {model_name}")

    # Return only trainable params for optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params


# ---------------------------
# Main
# ---------------------------

def main():
    global MODEL_NAME

    # 1) Dataset + splits (use coarse_label as target)
    full_dataset = EmoticFaceDataset(
        csv_path=DATA_CSV,
        label_column="coarse_label",
        transform=None,         # you can plug in augmentations later
    )
    num_classes = len(full_dataset.label_to_idx)
    print("Num classes:", num_classes)
    print("Label mapping:", full_dataset.label_to_idx)

    n_total = len(full_dataset)
    n_val = int(VAL_FRACTION * n_total)
    n_train = n_total - n_val

    torch.manual_seed(0)
    random.seed(0)

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Sanity check
    imgs, labels = next(iter(train_loader))
    print("Batch shape:", imgs.shape, labels.shape)

    # 2) Class weights for imbalance (computed from train_ds)
    label_counts = Counter()
    for _, lbl in train_ds:
        label_counts[int(lbl)] += 1

    print("Train label counts (by index):", label_counts)

    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        count = label_counts.get(idx, 1)
        class_weights[idx] = 1.0 / count

    # Normalize weights so they don't explode
    class_weights = class_weights * (num_classes / class_weights.sum())
    class_weights = class_weights.to(DEVICE)

    print("Class weights (in loss):", class_weights.tolist())

    # 3) Build model
    model = build_model(MODEL_NAME, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4) Prep dirs + history
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    history = {
        "epoch": [],
        "phase": [],        # "warmup", "finetune", or "single"
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    global_epoch = 0

    # ---------------------------------------------------
    # Case 1: TinyCNN — single-stage training
    # ---------------------------------------------------
    if MODEL_NAME == "tinycnn":
        optimizer = optim.Adam(model.parameters(), lr=TINYCNN_LR)

        for epoch in range(1, TINYCNN_EPOCHS + 1):
            global_epoch += 1
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc = eval_one_epoch(
                model, val_loader, criterion, DEVICE
            )

            print(
                f"[tinycnn] Epoch {epoch:02d}/{TINYCNN_EPOCHS} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
            )

            history["epoch"].append(global_epoch)
            history["phase"].append("single")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = save_dir / f"emotion_{MODEL_NAME}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "label_to_idx": full_dataset.label_to_idx,
                        "model_name": MODEL_NAME,
                    },
                    ckpt_path,
                )
                print(f"  ✅ New best model saved to {ckpt_path} (val acc={val_acc:.3f})")

    # ---------------------------------------------------
    # Case 2: Pretrained models — 2-stage training
    # ---------------------------------------------------
    else:
        # ---------- Stage 1: freeze backbone, train classifier ----------
        print("=== Stage 1: classifier warmup (frozen backbone) ===")
        trainable_params = freeze_backbone_and_unfreeze_classifier(model, MODEL_NAME)
        optimizer = optim.Adam(trainable_params, lr=WARMUP_LR)

        for epoch in range(1, WARMUP_EPOCHS + 1):
            global_epoch += 1
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc = eval_one_epoch(
                model, val_loader, criterion, DEVICE
            )

            print(
                f"[{MODEL_NAME} warmup] Epoch {epoch:02d}/{WARMUP_EPOCHS} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
            )

            history["epoch"].append(global_epoch)
            history["phase"].append("warmup")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = save_dir / f"emotion_{MODEL_NAME}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "label_to_idx": full_dataset.label_to_idx,
                        "model_name": MODEL_NAME,
                    },
                    ckpt_path,
                )
                print(f"  ✅ New best model saved to {ckpt_path} (val acc={val_acc:.3f})")

        # ---------- Stage 2: unfreeze all, fine-tune ----------
        print("=== Stage 2: full fine-tuning (unfrozen backbone) ===")
        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)

        for epoch in range(1, FINETUNE_EPOCHS + 1):
            global_epoch += 1
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc = eval_one_epoch(
                model, val_loader, criterion, DEVICE
            )

            print(
                f"[{MODEL_NAME} finetune] Epoch {epoch:02d}/{FINETUNE_EPOCHS} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
            )

            history["epoch"].append(global_epoch)
            history["phase"].append("finetune")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = save_dir / f"emotion_{MODEL_NAME}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "label_to_idx": full_dataset.label_to_idx,
                        "model_name": MODEL_NAME,
                    },
                    ckpt_path,
                )
                print(f"  ✅ New best model saved to {ckpt_path} (val acc={val_acc:.3f})")

    # 5) Save history for plotting
    history_df = pd.DataFrame(history)
    history_path = logs_dir / f"history_{MODEL_NAME}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")
    print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":

    # List of models you want to train
    MODELS_TO_TRAIN = [
        "tinycnn",
        "mobilenetv2",
        "efficientnet_lite0",
    ]

    for MODEL_NAME in MODELS_TO_TRAIN:
        print("\n" + "=" * 80)
        print(f" Training model: {MODEL_NAME}")
        print("=" * 80 + "\n")

        try:
            main()  # runs the full training loop with the current MODEL_NAME
        except Exception as e:
            print(f" Error while training {MODEL_NAME}: {e}")
            continue

    print("\n🎉 All requested models have finished training!")