import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return None, None, None
    epochs, train, val = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return epochs, train, val


if __name__ == "__main__":
    baseline_path = Path("training_curve_baseline.csv")
    improved_path = Path("improved_model") / "training_curve_improved.csv"

    b_epochs, b_train, b_val = load_curve(baseline_path)
    i_epochs, i_train, i_val = load_curve(improved_path)

    if b_epochs and b_val:
        plt.plot(b_epochs, b_train, label="Baseline train", color="tab:blue", linestyle="--")
        plt.plot(b_epochs, b_val, label="Baseline val", color="tab:blue")
    if i_epochs and i_val:
        plt.plot(i_epochs, i_train, label="Improved train", color="tab:orange", linestyle="--")
        plt.plot(i_epochs, i_val, label="Improved val", color="tab:orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = Path("loss_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
