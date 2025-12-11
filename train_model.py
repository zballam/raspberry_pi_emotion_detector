from pathlib import Path
import copy
import csv
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms
import matplotlib.pyplot as plt

# efficientnet_lite0 from timm
try:
    import timm
except ImportError:
    timm = None

from utility.face_dataloader import EmoticFaceDataset
from utility.tinycnn import TinyCNN


if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
# ONE-CSV dataloader (labels_coarse.csv) - WEIGHTED
# =====================================================
def make_dataloaders_single_csv(
    model_name: str,
    csv_path: str,
    label_column="coarse_label",
    batch_size=64,
    val_frac=0.2,
    seed=42,
    imbalance_gamma: float = 1.5,
):
    """
    Random split from ONE CSV, with strong imbalance handling
    (WeightedRandomSampler + class_weights).
    Returns:
        train_loader, val_loader, num_classes, class_weights
    """
    # Base dataset (no transforms) just to access labels & mapping
    base = EmoticFaceDataset(csv_path, label_column, transform=None)
    num_samples = len(base)
    num_classes = len(base.label_to_idx)

    # ------------------------------
    # Random split train / val
    # ------------------------------
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    n_val = int(val_frac * num_samples)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # ------------------------------
    # Class counts from TRAIN ONLY
    # ------------------------------
    train_label_indices = []
    for i in train_idx.tolist():
        row = base.df.iloc[i]
        label_name = row[label_column]
        label_idx = base.label_to_idx[label_name]
        train_label_indices.append(label_idx)

    train_label_indices = torch.tensor(train_label_indices, dtype=torch.long)
    class_counts = torch.bincount(train_label_indices, minlength=num_classes).float()
    class_counts[class_counts == 0] = 1.0  # safety to avoid div-by-zero

    print("[make_dataloaders_single_csv] Class counts:", class_counts.tolist())

    # Inverse-frequency weights with stronger exponent
    inv_freq = (1.0 / class_counts) ** imbalance_gamma
    inv_freq = inv_freq / inv_freq.mean()  # normalize around 1.0

    print(
        f"[make_dataloaders_single_csv] Sampler weights (inv_freq, gamma={imbalance_gamma}):",
        inv_freq.tolist(),
    )

    # ------------------------------
    # Device-dependent loader config
    # ------------------------------
    if device.type == "cuda":
        # Larger batch to keep 3090 busier, but still safe for Windows
        batch_size = 300
        num_workers = 4
        persistent_workers = False
        prefetch_factor = 2
    elif device.type == "mps":
        batch_size = 64
        num_workers = 4
        persistent_workers = False
        prefetch_factor = 2
    else:
        batch_size = 64
        num_workers = 2
        persistent_workers = False
        prefetch_factor = 2

    print(
        f"[make_dataloaders_single_csv] Using batch_size={batch_size}, "
        f"num_workers={num_workers}, persistent_workers={persistent_workers}"
    )

    # ------------------------------
    # WeightedRandomSampler for TRAIN
    # ------------------------------
    sample_weights = inv_freq[train_label_indices]
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

    # ------------------------------
    # DataLoaders
    # ------------------------------
    if num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # ------------------------------
    # Class weights for the LOSS
    # ------------------------------
    class_weights = inv_freq.clone()
    class_weights = class_weights / class_weights.mean()
    print("[make_dataloaders_single_csv] Loss class_weights:", class_weights.tolist())

    return train_loader, val_loader, num_classes, class_weights


# =====================================================
# Balanced dataloader (downsample train set per class)
# =====================================================
def make_dataloaders_balanced_single_csv(
    model_name: str,
    csv_path: str,
    label_column: str = "coarse_label",
    batch_size: int = 64,
    val_frac: float = 0.2,
    seed: int = 42,
    max_per_class: int | None = None,
):
    """
    Random split from ONE CSV, but then balances the TRAIN set by
    downsampling each class to the same count (min class count or max_per_class).
    No class weights or WeightedRandomSampler.
    Returns:
        train_loader, val_loader, num_classes, class_weights(None)
    """
    base = EmoticFaceDataset(csv_path, label_column, transform=None)
    num_samples = len(base)
    num_classes = len(base.label_to_idx)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=g)
    n_val = int(val_frac * num_samples)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Build mapping: class_idx -> list of train indices
    class_to_indices = {c: [] for c in range(num_classes)}
    for i in train_idx.tolist():
        row = base.df.iloc[i]
        label_name = row[label_column]
        label_idx = base.label_to_idx[label_name]
        class_to_indices[label_idx].append(i)

    # Determine per-class sample count
    counts = {c: len(idxs) for c, idxs in class_to_indices.items()}
    print("[make_dataloaders_balanced_single_csv] Raw train counts:", counts)

    min_count = min(counts.values())
    if max_per_class is not None:
        n_per_class = min(min_count, max_per_class)
    else:
        n_per_class = min_count

    print(
        f"[make_dataloaders_balanced_single_csv] Using n_per_class={n_per_class} "
        f"(min_count={min_count}, max_per_class={max_per_class})"
    )

    # Sample n_per_class from each class
    rng = random.Random(seed)
    balanced_indices = []
    for c, idxs in class_to_indices.items():
        if len(idxs) > n_per_class:
            balanced_indices.extend(rng.sample(idxs, n_per_class))
        else:
            balanced_indices.extend(idxs)

    # Shuffle balanced indices
    rng.shuffle(balanced_indices)
    balanced_idx = torch.tensor(balanced_indices, dtype=torch.long)

    # ------------------------------
    # Device-dependent loader config
    # ------------------------------
    if device.type == "cuda":
        batch_size = 300
        num_workers = 4
        persistent_workers = False
        prefetch_factor = 2
    elif device.type == "mps":
        batch_size = 64
        num_workers = 4
        persistent_workers = False
        prefetch_factor = 2
    else:
        batch_size = 64
        num_workers = 2
        persistent_workers = False
        prefetch_factor = 2

    print(
        f"[make_dataloaders_balanced_single_csv] Using batch_size={batch_size}, "
        f"num_workers={num_workers}, persistent_workers={persistent_workers}"
    )

    train_tf = get_transforms(model_name, True)
    val_tf = get_transforms(model_name, False)

    full_train = EmoticFaceDataset(csv_path, label_column, transform=train_tf)
    full_val = EmoticFaceDataset(csv_path, label_column, transform=val_tf)

    train_ds = Subset(full_train, balanced_idx)
    val_ds = Subset(full_val, val_idx)

    if num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # No class weights for balanced training
    class_weights = None
    return train_loader, val_loader, num_classes, class_weights


# =====================================================
# Training loop
# =====================================================
def _cuda_stats():
    """Helper to grab simple GPU stats via nvidia-smi, if available."""
    import subprocess, shutil

    if device.type != "cuda":
        return None

    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        util, mem_used, mem_total = result.strip().split(", ")
        return int(util), int(mem_used), int(mem_total)
    except Exception:
        return None


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
        print(f"[{model_name}] Using standard CrossEntropyLoss (no class weights)")
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

    # -------------------------------------------------
    # Warmup pass (CUDA only)
    # -------------------------------------------------
    warmup_time = None
    if device.type == "cuda":
        print(f"[{model_name}] Warmup: running one batch to initialize "
              f"dataloader workers & CUDA/cuDNN...")

        torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _ = criterion(out, y)
                break

        torch.cuda.synchronize()
        warmup_time = time.time() - t0

        print(f"[{model_name}] Warmup done in {warmup_time:.2f}s\n")

    # -------------------------------------------------
    # Epoch loop
    # -------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.time()

        # -------- Train --------
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

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
        running_loss = 0.0

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

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        msg = (f"[{model_name}] Epoch {epoch:02d} "
               f"Train {train_loss:.3f}/{train_acc:.3f} | "
               f"Val {val_loss:.3f}/{val_acc:.3f} | "
               f"{epoch_time:.2f}s")
        print(msg)

        stats = _cuda_stats()
        if stats is not None:
            util, mem_used, mem_total = stats
            print(f"[CUDA] GPU util: {util}% | Mem: {mem_used} / {mem_total} MiB")

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

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - total_start

    print(f"[{model_name}] Total training time: {total_time:.2f}s")
    if warmup_time is not None:
        print(f"[{model_name}]   ├─ Warmup/setup time: {warmup_time:.2f}s")
        print(f"[{model_name}]   └─ Time inside epoch loop: "
              f"{total_time - warmup_time:.2f}s")

    model.load_state_dict(best_state)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[{model_name}] Saved best model to {save_path}")

    return model, history


# =====================================================
# Run one experiment + save CSV
# =====================================================
def run_experiment(model_name, csv_path, regime="weighted", max_per_class=None):
    """
    regime: "weighted" (current behavior) or "balanced"
    """
    exp_start = time.time()

    print(f"\n=== Training {model_name} ({regime}) ===")
    print("[setup] Building dataloaders...")
    t0 = time.time()
    if regime == "weighted":
        train_loader, val_loader, num_classes, class_weights = make_dataloaders_single_csv(
            model_name, csv_path
        )
    elif regime == "balanced":
        train_loader, val_loader, num_classes, class_weights = make_dataloaders_balanced_single_csv(
            model_name, csv_path, max_per_class=max_per_class
        )
    else:
        raise ValueError(f"Unknown regime {regime}")
    t1 = time.time()
    print(f"[setup] Dataloaders ready in {t1 - t0:.2f}s")

    print("[setup] Building model...")
    t2 = time.time()
    model, params = build_model(model_name, num_classes)
    t3 = time.time()
    print(f"[setup] Model ready in {t3 - t2:.2f}s\n")

    lr = 1e-3 if model_name == "tinycnn" else 1e-4
    save_path = f"models/emotion_{model_name}_{regime}_best.pt"

    model, history = train_model(
        model_name=f"{model_name}_{regime}",
        model=model,
        params_to_optimize=params,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=40,
        lr=lr,
        patience=7,
        save_path=save_path,
        class_weights=class_weights,
    )

    # Save training history
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    hist_path = logs / f"history_{model_name}_{regime}.csv"

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

    exp_time = time.time() - exp_start
    print(f"[{model_name}_{regime}] Experiment finished in {exp_time:.2f}s\n")

    return model, history


# =====================================================
# Plotting helpers
# =====================================================
def plot_histories(all_histories, regime, metric, title_suffix=""):
    """
    all_histories: dict[(model_name, regime)] -> history dict
    regime: "weighted" or "balanced"
    metric: "train_loss", "val_loss", "train_acc", "val_acc"
    """
    plt.figure()
    for model_name in ["tinycnn", "mobilenetv2", "efficientnet_lite0"]:
        key = (model_name, regime)
        if key not in all_histories:
            continue
        hist = all_histories[key]
        y = hist[metric]
        x = list(range(1, len(y) + 1))
        plt.plot(x, y, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} ({regime}){title_suffix}")
    plt.legend()
    plt.grid(True)


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    print("Using device:", device)
    if device.type == "cuda":
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    CSV_PATH = "data/emotic_faces_128/labels_coarse.csv"

    model_names = ["tinycnn", "mobilenetv2", "efficientnet_lite0"]

    # You can cap per-class size for balanced if you want faster training
    max_per_class_balanced = None  # e.g. 4000 if you want

    all_histories = {}

    for name in model_names:
        # Weighted / class-weight regime (current behavior)
        _, hist_w = run_experiment(name, CSV_PATH, regime="weighted")
        all_histories[(name, "weighted")] = hist_w

        # Balanced-dataset regime
        _, hist_b = run_experiment(
            name, CSV_PATH, regime="balanced", max_per_class=max_per_class_balanced
        )
        all_histories[(name, "balanced")] = hist_b

    # --------- Plotting (8 plots total) ----------
    # Weighted: 4 plots
    for metric in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        plot_histories(all_histories, regime="weighted", metric=metric)

    # Balanced: 4 plots
    for metric in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        plot_histories(all_histories, regime="balanced", metric=metric)

    # Show all figures
    plt.show()