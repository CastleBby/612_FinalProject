import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(repo_root))
from transformer_model import get_model


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def regression_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    precip_obs: torch.Tensor,
    precip_min: torch.Tensor,
    precip_scale: torch.Tensor,
    precip_max: torch.Tensor,
    quantile: float = 0.9,
    extreme_weight: float = 1.0,
    overprediction_weight: float = 1.0,
    use_log1p: bool = True,
) -> torch.Tensor:
    """
    Regression loss computed in physical units with optional log1p compression.
    Also applies a mild penalty for over-forecasting/clipping to curb false positives.
    """
    # unscale to mm/h and clamp to a realistic range
    y_true_mm = torch.clamp((y_true - precip_min) / precip_scale, min=0.0, max=precip_max)
    raw_pred_mm = (y_pred - precip_min) / precip_scale
    y_pred_mm = torch.clamp(raw_pred_mm, min=0.0, max=precip_max * 1.5)
    # penalise if the model produces out-of-range negatives or extreme values
    clip_penalty = ((raw_pred_mm - y_pred_mm) ** 2).mean()
    precip_obs_mm = torch.clamp((precip_obs - precip_min) / precip_scale, min=0.0, max=precip_max)

    if use_log1p:
        y_true_proc = torch.log1p(y_true_mm)
        y_pred_proc = torch.log1p(y_pred_mm)
    else:
        y_true_proc = y_true_mm
        y_pred_proc = y_pred_mm

    mse = (y_pred_proc - y_true_proc) ** 2
    weights = torch.ones_like(y_true_proc)
    weights = torch.where(y_pred_mm > y_true_mm, weights * overprediction_weight, weights)
    heavy = precip_obs_mm > torch.quantile(precip_obs_mm, quantile)
    weights[heavy] = extreme_weight
    return (mse * weights).mean() + 0.05 * clip_penalty


if __name__ == "__main__":
    # --------------------------------------------------------------
    # 0. Reproducibility
    # --------------------------------------------------------------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config()

    # --------------------------------------------------------------
    # 1. Device
    # --------------------------------------------------------------
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)
    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # 2. Load processed data (includes loc_idx)
    # --------------------------------------------------------------
    data = np.load(repo_root / "processed_data.npz", allow_pickle=True)
    X_train, y_train, loc_train = data["X_train"], data["y_train"], data["loc_train"]
    X_val, y_val, loc_val = data["X_val"], data["y_val"], data["loc_val"]
    scaler = data["scaler"].item()
    precip_idx = cfg["data"]["variables"].index("precipitation")
    precip_max_val = torch.tensor(float(scaler.data_max_[precip_idx]), device=device, dtype=torch.float32)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.LongTensor(loc_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.LongTensor(loc_val),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["model"]["batch_size"], shuffle=False)

    # --------------------------------------------------------------
    # 3. Build model (deeper encoder, narrower width)
    # --------------------------------------------------------------
    input_dim = X_train.shape[-1]
    model = get_model(
        "transformer",
        input_dim=input_dim,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["model"]["lr"],
        weight_decay=cfg["model"].get("weight_decay", 0.0),
    )
    warmup_epochs = cfg["model"].get("warmup_epochs", 0)
    total_epochs = cfg["model"]["epochs"]
    eta_min = cfg["model"]["lr"] * 0.2

    def lr_lambda(epoch: int) -> float:
        # epoch is zero-based in LambdaLR
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        # cosine decay from 1 → eta_min_ratio
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return eta_min / cfg["model"]["lr"] + (1 - eta_min / cfg["model"]["lr"]) * cosine

    # --------------------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------------------
    best_val = float("inf")
    history = []
    best_path = Path(__file__).resolve().parent / "best_model_deepened.pth"
    patience = cfg["model"].get("early_stop_patience")
    epochs_since_improve = 0
    cls_weight = cfg["model"].get("cls_weight", 1.0)
    precip_scale = torch.tensor(scaler.scale_[precip_idx], device=device, dtype=torch.float32)
    precip_min = torch.tensor(scaler.min_[precip_idx], device=device, dtype=torch.float32)
    for epoch in range(1, cfg["model"]["epochs"] + 1):
        lr_factor = lr_lambda(epoch - 1)  # zero-based epoch index for schedule
        current_lr = cfg["model"]["lr"] * lr_factor
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        model.train()
        train_loss = 0.0
        for xb, yb, locb in tqdm(train_loader, desc=f"Epoch {epoch} [train] (lr={current_lr:.2e})"):
            xb, yb, locb = xb.to(device), yb.to(device), locb.to(device)

            precip_obs = xb[:, -1, cfg["data"]["variables"].index("precipitation")]

            optimizer.zero_grad()
            pred_reg, pred_cls = model(xb, locb, return_cls=True)

            # regression + classification heads are trained jointly
            reg_loss = regression_loss(
                pred_reg,
                yb,
                precip_obs,
                precip_min=precip_min,
                precip_scale=precip_scale,
                precip_max=precip_max_val,
                quantile=cfg["eval"]["extreme_threshold"],
                extreme_weight=cfg["model"]["extreme_weight"],
                overprediction_weight=cfg["model"].get("overprediction_weight", 1.0),
                use_log1p=True,
            )
            # invert MinMax scaling to compute rain/no-rain label in mm/h
            # compute rain/no-rain targets in mm/h (post inverse scaling)
            yb_unscaled = torch.clamp((yb - precip_min) / precip_scale, min=0.0)
            target_cls = (yb_unscaled > cfg["eval"]["rain_threshold"]).float()
            cls_loss = F.binary_cross_entropy(pred_cls, target_cls)

            loss = reg_loss + cls_weight * cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ---------------- validation ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, locb in val_loader:
                xb, yb, locb = xb.to(device), yb.to(device), locb.to(device)
                precip_obs = xb[:, -1, cfg["data"]["variables"].index("precipitation")]
                pred_reg, pred_cls = model(xb, locb, return_cls=True)

                reg_loss = regression_loss(
                    pred_reg,
                    yb,
                    precip_obs,
                    precip_min=precip_min,
                    precip_scale=precip_scale,
                    precip_max=precip_max_val,
                    quantile=cfg["eval"]["extreme_threshold"],
                    extreme_weight=cfg["model"]["extreme_weight"],
                    overprediction_weight=cfg["model"].get("overprediction_weight", 1.0),
                    use_log1p=True,
                )
                yb_unscaled = torch.clamp((yb - precip_min) / precip_scale, min=0.0)
                target_cls = (yb_unscaled > cfg["eval"]["rain_threshold"]).float()
                cls_loss = F.binary_cross_entropy(pred_cls, target_cls)

                val_loss += (reg_loss + cls_weight * cls_loss).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch:02d} (lr {current_lr:.2e}) | Train {train_loss:.6f} | Val {val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  → best model saved to {best_path.name}")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if patience is not None and epochs_since_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # save loss history for plotting
    out_path = Path(__file__).resolve().parent / "training_curve_improved.csv"
    with out_path.open("w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']},{row['val_loss']}\n")
    print(f"Training finished. Loss curve saved to {out_path.name}.")
