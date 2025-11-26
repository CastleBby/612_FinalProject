# train_multitask.py
# Multi-task training: regression + classification
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def binary_cross_entropy_with_logits_weighted(logits, y_true_binary, rain_weight=2.0):
    """
    BCE loss using logits (numerically stable).
    logits: (batch,) raw outputs from classification head (no sigmoid)
    y_true_binary: (batch,) binary labels {0, 1}
    """
    # Use PyTorch's stable BCE with logits
    # pos_weight: weight for positive class (rain events)
    pos_weight = torch.tensor([rain_weight], device=logits.device)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        y_true_binary,
        pos_weight=pos_weight,
        reduction='mean'
    )

    return loss


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
    # 2. Load processed data
    # --------------------------------------------------------------
    data = np.load('processed_data.npz', allow_pickle=True)
    X_train, y_train, loc_train = data['X_train'], data['y_train'], data['loc_train']
    X_val,   y_val,   loc_val   = data['X_val'],   data['y_val'],   data['loc_val']
    scaler = data['scaler'].item()
    precip_idx = cfg['data']['variables'].index('precipitation')

    # Create binary labels for classification (rain > threshold)
    rain_threshold_scaled = cfg['eval']['rain_threshold']
    # Need to scale the threshold
    dummy = np.zeros((1, scaler.scale_.size))
    dummy[0, precip_idx] = rain_threshold_scaled
    scaled_threshold = scaler.transform(dummy)[0, precip_idx]

    y_train_binary = (y_train > scaled_threshold).astype(np.float32)
    y_val_binary = (y_val > scaled_threshold).astype(np.float32)

    print(f"Binary labels created with threshold: {rain_threshold_scaled} mm")
    print(f"Training set: {y_train_binary.mean()*100:.1f}% positive (rain)")
    print(f"Validation set: {y_val_binary.mean()*100:.1f}% positive (rain)")

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(y_train_binary),
        torch.LongTensor(loc_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.FloatTensor(y_val_binary),
        torch.LongTensor(loc_val)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['model']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['model']['batch_size'], shuffle=False)

    # --------------------------------------------------------------
    # 3. Build multi-task model
    # --------------------------------------------------------------
    input_dim = X_train.shape[-1]
    model = get_model(
        'multitask',  # Use multi-task model
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nMulti-Task Model: {total_params/1e6:.2f}M parameters")

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
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision training (only for CUDA)
    use_amp = device.type == 'cuda'
    scaler_amp = GradScaler() if use_amp else None

    # --------------------------------------------------------------
    # 4. Training loop with dual loss
    # --------------------------------------------------------------
    best_val = float('inf')
    patience = cfg['model'].get('patience', 15)
    patience_counter = 0

    # Loss weights (reduce classification to prevent instability)
    regression_weight = 0.9  # 90% weight on regression (RMSE/MAE)
    classification_weight = 0.1  # 10% weight on classification (CSI/POD)

    # Resume from checkpoint if exists
    resume_checkpoint = os.path.exists('best_multitask_model.pth')
    if resume_checkpoint:
        print("Found existing checkpoint. Resuming training...")
        checkpoint = torch.load('best_multitask_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['val_loss']
        for _ in range(checkpoint['epoch']):
            scheduler.step()
        print(f"Resuming from epoch {checkpoint['epoch']}, best val loss: {best_val:.6f}")
        log_file = open('train_multitask.log', 'a')
    else:
        print("Starting training from scratch...")
        log_file = open('train_multitask.log', 'w')
        log_file.write("Epoch,Train_Reg_Loss,Train_Class_Loss,Train_Total_Loss,Val_Reg_Loss,Val_Class_Loss,Val_Total_Loss,Learning_Rate\n")
        start_epoch = 1

    print(f"\n{'='*70}")
    print(f"Multi-Task Training Configuration:")
    print(f"  Regression weight: {regression_weight:.1f}")
    print(f"  Classification weight: {classification_weight:.1f}")
    print(f"  Extreme weight (regression): {cfg['model']['extreme_weight']}")
    print(f"  Rain weight (classification): 2.0")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, total_epochs + 1):
        # --- Training ---
        model.train()
        train_reg_loss_sum = 0.0
        train_class_loss_sum = 0.0
        train_total_loss_sum = 0.0

        for batch_X, batch_y, batch_y_binary, batch_loc in tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}"):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_y_binary = batch_y_binary.to(device)
            batch_loc = batch_loc.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    reg_out, class_out = model(batch_X, batch_loc)
                    reg_loss = weighted_mse(reg_out, batch_y, batch_y,
                                          quantile=cfg['eval']['extreme_threshold'],
                                          extreme_weight=cfg['model']['extreme_weight'])
                    class_loss = binary_cross_entropy_with_logits_weighted(class_out, batch_y_binary, rain_weight=2.0)
                    total_loss = regression_weight * reg_loss + classification_weight * class_loss

                scaler_amp.scale(total_loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model']['gradient_clip'])
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                reg_out, class_out = model(batch_X, batch_loc)
                reg_loss = weighted_mse(reg_out, batch_y, batch_y,
                                      quantile=cfg['eval']['extreme_threshold'],
                                      extreme_weight=cfg['model']['extreme_weight'])
                class_loss = binary_cross_entropy_with_logits_weighted(class_out, batch_y_binary, rain_weight=2.0)
                total_loss = regression_weight * reg_loss + classification_weight * class_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model']['gradient_clip'])
                optimizer.step()

            train_reg_loss_sum += reg_loss.item()
            train_class_loss_sum += class_loss.item()
            train_total_loss_sum += total_loss.item()

        avg_train_reg = train_reg_loss_sum / len(train_loader)
        avg_train_class = train_class_loss_sum / len(train_loader)
        avg_train_total = train_total_loss_sum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_reg_loss_sum = 0.0
        val_class_loss_sum = 0.0
        val_total_loss_sum = 0.0

        with torch.no_grad():
            for batch_X, batch_y, batch_y_binary, batch_loc in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_y_binary = batch_y_binary.to(device)
                batch_loc = batch_loc.to(device)

                reg_out, class_out = model(batch_X, batch_loc)
                reg_loss = weighted_mse(reg_out, batch_y, batch_y,
                                      quantile=cfg['eval']['extreme_threshold'],
                                      extreme_weight=cfg['model']['extreme_weight'])
                class_loss = binary_cross_entropy_with_logits_weighted(class_out, batch_y_binary, rain_weight=2.0)
                total_loss = regression_weight * reg_loss + classification_weight * class_loss

                val_reg_loss_sum += reg_loss.item()
                val_class_loss_sum += class_loss.item()
                val_total_loss_sum += total_loss.item()

        avg_val_reg = val_reg_loss_sum / len(val_loader)
        avg_val_class = val_class_loss_sum / len(val_loader)
        avg_val_total = val_total_loss_sum / len(val_loader)

        # Logging
        current_lr = scheduler.get_last_lr()[0]
        log_file.write(f"{epoch},{avg_train_reg:.8f},{avg_train_class:.8f},{avg_train_total:.8f},"
                      f"{avg_val_reg:.8f},{avg_val_class:.8f},{avg_val_total:.8f},{current_lr:.8f}\n")
        log_file.flush()

        print(f"Epoch {epoch:3d} | Train: Reg={avg_train_reg:.6f} Class={avg_train_class:.6f} Total={avg_train_total:.6f} | "
              f"Val: Reg={avg_val_reg:.6f} Class={avg_val_class:.6f} Total={avg_val_total:.6f} | LR={current_lr:.6f}")

        # Save best model
        if avg_val_total < best_val:
            best_val = avg_val_total
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_total,
                'val_reg_loss': avg_val_reg,
                'val_class_loss': avg_val_class
            }, 'best_multitask_model.pth')
            print(f"  --> Saved best model (val_total_loss={avg_val_total:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience})")
                break

        scheduler.step()

    log_file.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val:.6f}")
