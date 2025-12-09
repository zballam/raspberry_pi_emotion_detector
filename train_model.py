# train_model.py

import random
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

DATA_CSV = "data/emotic_faces_128/labels.csv"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-3
VAL_FRACTION = 0.2

# Choose which model to train: "tinycnn", "mobilenetv2", "efficientnet_lite0"
MODEL_NAME = "efficientnet_lite0"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", DEVICE)


# ---------------------------
# Models
# ---------------------------

def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    # -------------------------
    # TinyCNN
    # -------------------------
    if model_name == "tinycnn":
        print("Building TinyCNN")
        return TinyCNN(num_classes)

    # -------------------------
    # MobileNetV2
    # -------------------------
    if model_name == "mobilenetv2":
        if not HAS_TORCHVISION:
            raise RuntimeError("torchvision not installed; cannot use MobileNetV2.")

        print("Building MobileNetV2 (pretrained on ImageNet)")
        # NOTE: expects mobilenet_v2-7ebf99e0.pth to be available in
        # ~/.cache/torch/hub/checkpoints/ for pretrained=True to work without SSL issues.
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
        return net

    # -------------------------
    # EfficientNet-Lite0 (timm)
    # -------------------------
    if model_name == "efficientnet_lite0":
        if not HAS_TIMM:
            raise RuntimeError("timm not installed; cannot use EfficientNet-Lite0.")

        print("Building EfficientNet-Lite0 (pretrained on ImageNet)")
        # NOTE: expects efficientnet_lite0 weights cached in ~/.cache/timm/
        net = timm.create_model(
            "efficientnet_lite0",
            pretrained=True,      # set to False if you don't have weights cached
            num_classes=num_classes,
        )
        return net

    raise ValueError(f"Unknown model_name: {model_name}")


# ---------------------------
# Training / evaluation
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
# Main
# ---------------------------

def main():
    # 1) Dataset + splits
    full_dataset = EmoticFaceDataset(DATA_CSV)
    num_classes = len(full_dataset.label_to_idx)
    print("Num classes:", num_classes)

    n_total = len(full_dataset)
    n_val = int(VAL_FRACTION * n_total)
    n_train = n_total - n_val

    # Fix seed for reproducibility
    torch.manual_seed(0)
    random.seed(0)

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Quick sanity check batch
    imgs, labels = next(iter(train_loader))
    print("Batch shape:", imgs.shape, labels.shape)

    # 2) Model, loss, optimizer
    model = build_model(MODEL_NAME, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Prep directories + history
    best_val_acc = 0.0
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # 4) Train loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
        )

        # log history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint
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

    # 5) Save training history for plotting
    history_df = pd.DataFrame(history)
    history_path = logs_dir / f"history_{MODEL_NAME}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()