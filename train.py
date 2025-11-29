# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from tqdm import tqdm
import os
import random
from pathlib import Path
from transformer_model import get_model


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def weighted_mse(y_pred, y_true, precip_obs, quantile=0.9, extreme_weight=5.0):
    """MSE that gives more weight to heavy-rain hours."""
    mse = (y_pred - y_true) ** 2
    weights = torch.ones_like(y_true)
    heavy = precip_obs > torch.quantile(precip_obs, quantile)
    weights[heavy] = extreme_weight
    return (mse * weights).mean()


if __name__ == '__main__':
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

    # --------------------------------------------------------------
    # 1. Config & device
    # --------------------------------------------------------------
    cfg = load_config()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    device = torch.device(device)
    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # 2. Load processed data (includes loc_idx)
    # --------------------------------------------------------------
    data = np.load('processed_data.npz', allow_pickle=True)
    X_train, y_train, loc_train = data['X_train'], data['y_train'], data['loc_train']
    X_val,   y_val,   loc_val   = data['X_val'],   data['y_val'],   data['loc_val']

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.LongTensor(loc_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.LongTensor(loc_val)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['model']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['model']['batch_size'], shuffle=False)

    # --------------------------------------------------------------
    # 3. Build model
    # --------------------------------------------------------------
    input_dim = X_train.shape[-1]
    model = get_model(
        'transformer',
        input_dim=input_dim,
        seq_len=cfg['data']['seq_len'],
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=cfg['model']['num_layers'],
        embed_dim=cfg['model']['embed_dim'],
        dropout=cfg['model']['dropout'],
        num_locations=len(cfg['data']['locations']),
        feature_groups=cfg['model']['feature_groups']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'])

    # --------------------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------------------
    best_val = float('inf')
    history = []
    for epoch in range(1, cfg['model']['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, locb in tqdm(train_loader, desc=f'Epoch {epoch} [train]'):
            xb, yb, locb = xb.to(device), yb.to(device), locb.to(device)

            # precipitation observed at the *last* input hour → proxy for "current rain"
            precip_obs = xb[:, -1, cfg['data']['variables'].index('precipitation')]

            optimizer.zero_grad()
            pred = model(xb, locb)                     # (B,)
            loss = weighted_mse(
                pred, yb, precip_obs,
                quantile=cfg['eval']['extreme_threshold'],
                extreme_weight=cfg['model']['extreme_weight']
            )
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
                precip_obs = xb[:, -1, cfg['data']['variables'].index('precipitation')]
                pred = model(xb, locb)
                val_loss += weighted_mse(
                    pred, yb, precip_obs,
                    quantile=cfg['eval']['extreme_threshold'],
                    extreme_weight=cfg['model']['extreme_weight']
                ).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        print(f'Epoch {epoch:02d} | Train {train_loss:.6f} | Val {val_loss:.6f}')
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('  → best model saved')

    # save loss history for plotting
    out_path = Path("training_curve_baseline.csv")
    with out_path.open("w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']},{row['val_loss']}\n")
    print(f'Training finished. Loss curve saved to {out_path}')
