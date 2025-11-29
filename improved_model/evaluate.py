import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(repo_root))
from transformer_model import get_model


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with cfg_path.open("r") as f:
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


if __name__ == "__main__":
    # ----- reproducibility -----
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = load_config()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)
    print(f"Using device: {device}")

    # ----- load processed test data -----
    data = np.load(repo_root / "processed_data.npz", allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    loc_test = data["loc_test"]
    scaler = data["scaler"].item()

    precip_idx = cfg["data"]["variables"].index("precipitation")

    # ----- build & load model -----
    model = get_model(
        "transformer",
        input_dim=X_test.shape[-1],
        seq_len=cfg["data"]["seq_len"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        embed_dim=cfg["model"]["embed_dim"],
        dropout=cfg["model"]["dropout"],
        num_locations=len(cfg["data"]["locations"]),
        feature_groups=cfg["model"]["feature_groups"],
        return_cls=True,
    ).to(device)

    checkpoint = Path(__file__).resolve().parent / "best_model_deepened.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    eval_batch_size = cfg.get("eval", {}).get("batch_size", 256)

    # ----- build loader to avoid GPU OOM -----
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(loc_test))
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    precip_max = float(scaler.data_max_[precip_idx])

    # ----- inference -----
    preds = []
    cls_probs = []
    with torch.no_grad():
        for xb, locb in test_loader:
            xb, locb = xb.to(device), locb.to(device)
            batch_reg, batch_cls = model(xb, locb, return_cls=True)
            batch_reg = torch.clamp(batch_reg, min=0.0, max=precip_max * scaler.scale_[precip_idx] + scaler.min_[precip_idx])
            preds.append(batch_reg.cpu())
            cls_probs.append(batch_cls.cpu())

    pred_scaled = torch.cat(preds).numpy().flatten()
    cls_scores = torch.cat(cls_probs).numpy().flatten()

    # ----- un-scale -----
    y_pred = unscale(pred_scaled, scaler, precip_idx)
    y_true = unscale(y_test, scaler, precip_idx)

    # ----- gate regression with classifier to reduce false alarms -----
    # gate regression with classifier confidence to reduce false alarms
    class_thresh = cfg["eval"].get("class_threshold", 0.5)
    rain_mask = cls_scores >= class_thresh
    y_pred_gated = y_pred.copy()
    y_pred_gated[~rain_mask] = 0.0

    # ----- regression metrics -----
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_gated))
    mae = mean_absolute_error(y_true, y_pred_gated)

    # ----- rain-event metrics (threshold = 0.1 mm) -----
    pod, far, csi = classification_metrics(y_true, y_pred_gated, cfg["eval"]["rain_threshold"])

    # ----- extreme-event POD (90th percentile) -----
    extreme_thresh = np.percentile(y_true, cfg["eval"]["extreme_threshold"] * 100)
    pod_ext, _, _ = classification_metrics(y_true, y_pred_gated, extreme_thresh)

    # ----- print results -----
    print("\n=== DEEP+NARROW TRANSFORMER RESULTS (mm/h) ===")
    print(f"RMSE : {rmse:6.4f}")
    print(f"MAE  : {mae:6.4f}")
    print(f"CSI  : {csi:6.4f} | POD: {pod:6.4f} | FAR: {far:6.4f}")
    print(f"Extreme POD ({cfg['eval']['extreme_threshold']*100:.0f}th %): {pod_ext:6.4f}")

    # ----- persistence baseline (last observed = forecast) -----
    persist_scaled = X_test[:, -1, precip_idx]
    persist_pred = unscale(persist_scaled, scaler, precip_idx)

    rmse_p = np.sqrt(mean_squared_error(y_true, persist_pred))
    mae_p = mean_absolute_error(y_true, persist_pred)
    pod_p, far_p, csi_p = classification_metrics(y_true, persist_pred, cfg["eval"]["rain_threshold"])
    pod_ext_p, _, _ = classification_metrics(y_true, persist_pred, extreme_thresh)

    print("\n=== PERSISTENCE BASELINE ===")
    print(f"RMSE : {rmse_p:6.4f}")
    print(f"MAE  : {mae_p:6.4f}")
    print(f"CSI  : {csi_p:6.4f} | POD: {pod_p:6.4f} | FAR: {far_p:6.4f}")
    print(f"Extreme POD: {pod_ext_p:6.4f}")
