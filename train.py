# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
from tqdm import tqdm
import os
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['model']['lr'],
        weight_decay=cfg['model'].get('weight_decay', 0.01)
    )

    # Learning rate scheduler with warmup
    warmup_epochs = cfg['model'].get('warmup_epochs', 10)
    total_epochs = cfg['model']['epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision training (only for CUDA)
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # --------------------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------------------
    best_val = float('inf')
    patience = 15
    patience_counter = 0
    start_epoch = 1

    # Check if we should resume from checkpoint
    resume_checkpoint = os.path.exists('best_model.pth')
    if resume_checkpoint:
        print("Found existing checkpoint. Resuming training...")
        checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['val_loss']

        # Adjust scheduler to the current epoch
        for _ in range(checkpoint['epoch']):
            scheduler.step()

        print(f"Resuming from epoch {checkpoint['epoch']}, best val loss: {best_val:.6f}")

        # Open log file in append mode
        log_file = open('train.log', 'a')
    else:
        print("Starting training from scratch...")
        # Open log file in write mode
        log_file = open('train.log', 'w')
        log_file.write("Epoch,Train_Loss,Val_Loss,Learning_Rate\n")

    for epoch in range(start_epoch, cfg['model']['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, locb in tqdm(train_loader, desc=f'Epoch {epoch} [train]'):
            xb, yb, locb = xb.to(device), yb.to(device), locb.to(device)

            # precipitation observed at the *last* input hour → proxy for "current rain"
            precip_obs = xb[:, -1, cfg['data']['variables'].index('precipitation')]

            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    pred = model(xb, locb)
                    loss = weighted_mse(
                        pred, yb, precip_obs,
                        quantile=cfg['eval']['extreme_threshold'],
                        extreme_weight=cfg['model']['extreme_weight']
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model'].get('gradient_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb, locb)
                loss = weighted_mse(
                    pred, yb, precip_obs,
                    quantile=cfg['eval']['extreme_threshold'],
                    extreme_weight=cfg['model']['extreme_weight']
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model'].get('gradient_clip', 1.0))
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

        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:02d} | Train {train_loss:.6f} | Val {val_loss:.6f} | LR {current_lr:.6f}')

        # Log to file
        log_file.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{current_lr:.8f}\n")
        log_file.flush()

        # Step scheduler
        scheduler.step()

        # Early stopping and model checkpointing
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print('  → best model saved')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break

    log_file.close()
    print('Training finished. Log saved to train.log')
