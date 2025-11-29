import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from transformer_model import get_model


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def unscale(y_scaled: np.ndarray, scaler, precip_idx: int) -> np.ndarray:
    """Map scaled precipitation values back to physical units."""
    dummy = np.zeros((len(y_scaled), scaler.scale_.size))
    dummy[:, precip_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, precip_idx]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float):
    yb_t = (y_true > threshold).astype(int)
    yb_p = (y_pred > threshold).astype(int)
    tp = np.sum(yb_t & yb_p)
    fp = np.sum(~yb_t & yb_p)
    fn = np.sum(yb_t & ~yb_p)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return pod, far, csi


def evaluate_model(
    name: str,
    cfg_path: Path,
    ckpt_path: Path,
    use_classifier: bool,
    data: dict,
    scaler,
    device: torch.device,
):
    cfg = load_config(cfg_path)
    precip_idx = cfg["data"]["variables"].index("precipitation")

    model = get_model(
        "transformer",
        input_dim=data["X_test"].shape[-1],
        seq_len=cfg["data"]["seq_len"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        embed_dim=cfg["model"]["embed_dim"],
        dropout=cfg["model"]["dropout"],
        num_locations=len(cfg["data"]["locations"]),
        feature_groups=cfg["model"]["feature_groups"],
        return_cls=use_classifier,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[{name}] Warning: missing keys {missing}, unexpected keys {unexpected}")
    model.eval()

    eval_batch_size = cfg.get("eval", {}).get("batch_size", 256)
    test_ds = TensorDataset(torch.FloatTensor(data["X_test"]), torch.LongTensor(data["loc_test"]))
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    preds = []
    cls_probs = []
    precip_max = float(scaler.data_max_[precip_idx])
    with torch.no_grad():
        for xb, locb in test_loader:
            xb, locb = xb.to(device), locb.to(device)
            if use_classifier:
                reg, cls = model(xb, locb, return_cls=True)
                cls_probs.append(cls.cpu())
            else:
                reg = model(xb, locb)
            reg = torch.clamp(reg, min=0.0, max=precip_max * scaler.scale_[precip_idx] + scaler.min_[precip_idx])
            preds.append(reg.cpu())

    pred_scaled = torch.cat(preds).numpy().flatten()
    y_pred = unscale(pred_scaled, scaler, precip_idx)
    y_true = unscale(data["y_test"], scaler, precip_idx)

    if use_classifier:
        cls_scores = torch.cat(cls_probs).numpy().flatten()
        class_thresh = cfg["eval"].get("class_threshold", 0.5)
        rain_mask = cls_scores >= class_thresh
        y_pred = y_pred.copy()
        y_pred[~rain_mask] = 0.0

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pod, far, csi = classification_metrics(y_true, y_pred, cfg["eval"]["rain_threshold"])
    extreme_thresh = np.percentile(y_true, cfg["eval"]["extreme_threshold"] * 100)
    pod_ext, _, _ = classification_metrics(y_true, y_pred, extreme_thresh)

    print(f"\n=== {name} ===")
    print("RMSE: root-mean-square error (lower is better)")
    print("MAE : mean absolute error (lower is better)")
    print("CSI : critical success index (1 is perfect), POD: probability of detection, FAR: false alarm ratio")
    print(f"RMSE {rmse:6.4f} | MAE {mae:6.4f}")
    print(f"CSI  {csi:6.4f} | POD {pod:6.4f} | FAR {far:6.4f}")
    print(f"Extreme POD ({cfg['eval']['extreme_threshold']*100:.0f}th pct): {pod_ext:6.4f}")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)
    print(f"Using device: {device}")

    data = np.load("processed_data.npz", allow_pickle=True)
    shared = {
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "loc_test": data["loc_test"],
    }
    scaler = data["scaler"].item()

    models = [
        {
            "name": "Baseline transformer",
            "cfg": Path("config.yaml"),
            "ckpt": Path("best_model.pth"),
            "use_cls": False,
        },
        {
            "name": "Improved transformer (two-head, gated)",
            "cfg": Path("improved_model/config.yaml"),
            "ckpt": Path("improved_model/best_model_deepened.pth"),
            "use_cls": True,
        },
    ]

    for info in models:
        if not info["ckpt"].exists():
            print(f"\n=== {info['name']} ===")
            print(f"Checkpoint not found at {info['ckpt']}")
            continue
        evaluate_model(
            name=info["name"],
            cfg_path=info["cfg"],
            ckpt_path=info["ckpt"],
            use_classifier=info["use_cls"],
            data=shared,
            scaler=scaler,
            device=device,
        )
