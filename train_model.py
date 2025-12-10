# train_model.py

from pathlib import Path
import copy
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

# efficientnet_lite0 from timm
try:
    import timm
except ImportError:
    timm = None

# 🔹 Your existing code
from utility.face_dataloader import EmoticFaceDataset
from utility.tinycnn import TinyCNN   # ← USE YOUR IMPLEMENTATION


# =====================================================
# Device
# =====================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    print("Using device:", device)


# =====================================================
# Transforms (EmoticFaceDataset gives CHW float tensors)
# =====================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(model_name: str, is_train: bool):
    """
    Since EmoticFaceDataset returns a pre-converted tensor,
    we ONLY use tensor transforms (no ToTensor()).
    """
    size = 128 if model_name == "tinycnn" else 224

    ops = [transforms.Resize((size, size))]

    if is_train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
        ]

    ops += [
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]

    return transforms.Compose(ops)


# =====================================================
# Model builders
# =====================================================
def build_tinycnn(num_classes: int):
    model = TinyCNN(num_classes=num_classes)  # ← YOUR TinyCNN
    params = model.parameters()
    return model, params


def build_mobilenetv2(num_classes: int, freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    return model, params


def build_efficientnet_lite0(num_classes: int, freeze_backbone=True):
    if timm is None:
        raise ImportError("Install timm: pip install timm")

    model = timm.create_model("efficientnet_lite0", pretrained=True)
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    return model, params


def build_model(model_name: str, num_classes: int):
    if model_name == "tinycnn":
        return build_tinycnn(num_classes)
    elif model_name == "mobilenetv2":
        return build_mobilenetv2(num_classes)
    elif model_name == "efficientnet_lite0":
        return build_efficientnet_lite0(num_classes)
    else:
        raise ValueError(f"Unknown model name {model_name}")


# =====================================================
# ONE-CSV dataloader (labels_coarse.csv)
# =====================================================
def make_dataloaders_single_csv(
    model_name: str,
    csv_path: str,
    label_column="coarse_label",
    batch_size=64,
    val_frac=0.2,
    seed=42
):
    """
    Random split from ONE CSV.
    """
    base = EmoticFaceDataset(csv_path, label_column, transform=None)
    num_samples = len(base)
    num_classes = len(base.label_to_idx)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    n_val = int(val_frac * num_samples)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_tf = get_transforms(model_name, True)
    val_tf = get_transforms(model_name, False)

    full_train = EmoticFaceDataset(csv_path, label_column, transform=train_tf)
    full_val = EmoticFaceDataset(csv_path, label_column, transform=val_tf)

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_classes


# =====================================================
# Training loop
# =====================================================
def train_model(
    model_name,
    model,
    params_to_optimize,
    train_loader,
    val_loader,
    num_epochs=30,
    lr=1e-4,
    patience=5,
    save_path=None,
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_optimize, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}

    for epoch in range(1, num_epochs + 1):
        # -------- Train --------
        model.train()
        total, correct = 0, 0
        running_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # -------- Validate --------
        model.eval()
        total, correct = 0, 0
        running_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                running_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_loss = running_loss / total
        val_acc = correct / total

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[{model_name}] Epoch {epoch:02d} "
              f"Train {train_loss:.3f}/{train_acc:.3f} | "
              f"Val {val_loss:.3f}/{val_acc:.3f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_state)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[{model_name}] Saved best model to {save_path}")

    return model, history


# =====================================================
# Run one experiment + save CSV
# =====================================================
def run_experiment(model_name, csv_path):
    train_loader, val_loader, num_classes = make_dataloaders_single_csv(
        model_name, csv_path
    )

    model, params = build_model(model_name, num_classes)

    lr = 1e-3 if model_name == "tinycnn" else 1e-4
    save_path = f"models/emotion_{model_name}_best.pt"

    model, history = train_model(
        model_name,
        model,
        params,
        train_loader,
        val_loader,
        num_epochs=30,
        lr=lr,
        patience=5,
        save_path=save_path,
    )

    # Save training history
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    hist_path = logs / f"history_{model_name}.csv"

    with hist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["val_loss"][i],
                history["train_acc"][i],
                history["val_acc"][i],
            ])

    return model, history


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"

    for name in ["tinycnn"]: #, "mobilenetv2", "efficientnet_lite0"]:
        print(f"\n=== Training {name} ===")
        run_experiment(name, CSV_PATH)