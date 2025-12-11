# plot_training_curves.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = Path("logs")

def load_history(model_name: str) -> pd.DataFrame | None:
    path = LOGS_DIR / f"history_{model_name}.csv"
    if not path.exists():
        print(f"[WARN] History file not found for {model_name}: {path}")
        return None
    return pd.read_csv(path)

def plot_metric(histories, metric, title, ylabel):
    plt.figure(figsize=(6, 4))

    for model_name, df in histories.items():
        plt.plot(df["epoch"], df[metric], label=model_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    model_names = ["tinycnn"] #, "mobilenetv2", "efficientnet_lite0"]

    raw_histories = {name: load_history(name) for name in model_names}
    # keep only those that loaded successfully
    histories = {name: df for name, df in raw_histories.items() if df is not None}

    if not histories:
        print("No histories found. Run train_model.py first.")
        return

    plot_metric(histories, "train_loss", "Train Loss", "Loss")
    plot_metric(histories, "val_loss",   "Val Loss",   "Loss")
    plot_metric(histories, "train_acc",  "Train Accuracy", "Accuracy")
    plot_metric(histories, "val_acc",    "Val Accuracy",   "Accuracy")

if __name__ == "__main__":
    main()