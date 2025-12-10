# train_model.py

from pathlib import Path
import copy
import csv
import time
import subprocess  # ← NEW: for GPU stats via nvidia-smi

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms

# efficientnet_lite0 from timm
try:
    import timm
except ImportError:
    timm = None

# 🔹 Your existing code
from utility.face_dataloader import EmoticFaceDataset
from utility.tinycnn import TinyCNN   # ← YOUR IMPLEMENTATION


# =====================================================
# Device
# =====================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
if device.type == "cuda":
    try:
        print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception:
        pass


# =====================================================
# Helper: log GPU stats (CUDA only)
# =====================================================
def log_gpu_stats():
    """Print basic GPU util + memory stats if on CUDA and nvidia-smi is available."""
    if device.type != "cuda":
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Expect a single line like: "23, 1024, 24576"
        line = result.stdout.strip().splitlines()[0]
        util_str, mem_used_str, mem_total_str = [p.strip() for p in line.split(",")]
        print(
            f"[CUDA] GPU util: {util_str}% | Mem: {mem_used_str} / {mem_total_str} MiB"
        )
    except Exception as e:
        print(f"[CUDA] Could not query nvidia-smi: {e}")


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

    ops += [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    return transforms.Compose(ops)


# =====================================================
# Model builders
# =====================================================
def build_tinycnn(num_classes: int):
    model = TinyCNN(num_classes=num_classes)
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
#  + class weights + weighted sampler
#  + device-dependent batch_size / workers
# =====================================================
def make_dataloaders_single_csv(
    model_name: str,
    csv_path: str,
    label_column="coarse_label",
    batch_size=None,
    val_frac=0.2,
    seed=42,
):
    """
    Random split from ONE CSV.

    If batch_size is None:
      - MPS/CPU: batch_size = 64, num_workers = 4
      - CUDA:    batch_size = 256, num_workers = 8, persistent_workers=True

    Returns:
        train_loader, val_loader, num_classes, class_weights (tensor [num_classes])
    """
    # Device-specific defaults
    if batch_size is None:
        if device.type == "cuda":
            batch_size = 256
        else:
            batch_size = 64

    if device.type == "cuda":
        num_workers = 8
        persistent_workers = True
    else:
        num_workers = 4
        persistent_workers = False

    print(
        f"[make_dataloaders_single_csv] Using batch_size={batch_size}, "
        f"num_workers={num_workers}, persistent_workers={persistent_workers}"
    )

    # Base dataset (no transforms) just to access labels & mapping
    base = EmoticFaceDataset(csv_path, label_column, transform=None)
    num_samples = len(base)
    num_classes = len(base.label_to_idx)

    # Random split indices
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    n_val = int(val_frac * num_samples)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # -------------------------------------------------
    # Compute class counts from TRAIN split only
    # -------------------------------------------------
    train_label_indices = []
    for i in train_idx.tolist():
        row = base.df.iloc[i]
        label_name = row[label_column]
        label_idx = base.label_to_idx[label_name]
        train_label_indices.append(label_idx)

    train_label_indices = torch.tensor(train_label_indices, dtype=torch.long)
    class_counts = torch.bincount(train_label_indices, minlength=num_classes).float()

    # Avoid division by zero just in case (shouldn't happen, but safe)
    class_counts[class_counts == 0] = 1.0

    # Inverse-frequency weights
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    print("\n[make_dataloaders_single_csv] Class counts:", class_counts.tolist())
    print("[make_dataloaders_single_csv] Class weights:", class_weights.tolist())

    # -------------------------------------------------
    # WeightedRandomSampler for the TRAIN loader
    # -------------------------------------------------
    sample_weights = class_weights[train_label_indices]  # per-example weights
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Transforms
    train_tf = get_transforms(model_name, True)
    val_tf = get_transforms(model_name, False)

    full_train = EmoticFaceDataset(csv_path, label_column, transform=train_tf)
    full_val = EmoticFaceDataset(csv_path, label_column, transform=val_tf)

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    # NOTE: sampler for train, NO shuffle
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, num_classes, class_weights


# =====================================================
# Training loop (supports class_weights + timing + GPU stats)
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
    class_weights: torch.Tensor | None = None,
):
    model = model.to(device)

    # Use class-weighted loss if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"[{model_name}] Using class-weighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params_to_optimize, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # -------- Train --------
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
        running_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

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

        epoch_time = time.time() - epoch_start

        print(
            f"[{model_name}] Epoch {epoch:02d} "
            f"Train {train_loss:.3f}/{train_acc:.3f} | "
            f"Val {val_loss:.3f}/{val_acc:.3f} | "
            f"{epoch_time:.2f}s"
        )

        # 🔹 GPU usage stats (CUDA only)
        log_gpu_stats()

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

    total_time = time.time() - total_start
    print(f"[{model_name}] Total training time: {total_time:.2f}s")

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
    train_loader, val_loader, num_classes, class_weights = make_dataloaders_single_csv(
        model_name, csv_path, batch_size=None  # ← device-dependent inside
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
        class_weights=class_weights,
    )

    # Save training history
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    hist_path = logs / f"history_{model_name}.csv"

    with hist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(history["train_loss"])):
            writer.writerow(
                [
                    i + 1,
                    history["train_loss"][i],
                    history["val_loss"][i],
                    history["train_acc"][i],
                    history["val_acc"][i],
                ]
            )

    return model, history


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"

    for name in ["tinycnn"]:
        print(f"\n=== Training {name} ===")
        run_experiment(name, CSV_PATH)