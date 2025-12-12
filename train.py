# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
from tqdm import tqdm
import os
import random
import argparse
from transformer_model import get_model, weighted_physics_mse, set_seed, RANDOM_SEED


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train encoder-decoder precipitation transformer')
    parser.add_argument('--config', default=os.environ.get('CONFIG_PATH', 'config.yaml'),
                        help='Path to config YAML (default: config.yaml or CONFIG_PATH env)')
    return parser.parse_args()


def weighted_mse(y_pred, y_true, precip_obs, quantile=0.9, extreme_weight=5.0):
    """MSE that gives more weight to heavy-rain hours."""
    mse = (y_pred - y_true) ** 2
    weights = torch.ones_like(y_true)
    heavy = precip_obs > torch.quantile(precip_obs, quantile)
    weights[heavy] = extreme_weight
    return (mse * weights).mean()


def get_loss_function(cfg, use_physics=True):
    """
    Get the appropriate loss function based on config.
    
    Args:
        cfg: Configuration dictionary
        use_physics: Whether to use physics-informed loss
    
    Returns:
        Loss function callable
    """
    if use_physics and cfg['model'].get('use_physics_loss', False):
        # Create variable indices mapping
        variable_indices = {
            var: idx for idx, var in enumerate(cfg['data']['variables'])
        }
        
        def physics_loss_fn(y_pred, y_true, x_input):
            return weighted_physics_mse(
                y_pred, y_true, x_input, variable_indices,
                quantile=cfg['eval']['extreme_threshold'],
                extreme_weight=cfg['model']['extreme_weight'],
                physics_weight=cfg['model'].get('physics_weight', 0.1)
            )
        
        print("Using physics-informed loss function")
        return physics_loss_fn
    else:
        # Standard weighted MSE
        def standard_loss_fn(y_pred, y_true, x_input):
            precip_idx = cfg['data']['variables'].index('precipitation')
            precip_obs = x_input[:, -1, precip_idx]
            return weighted_mse(
                y_pred, y_true, precip_obs,
                quantile=cfg['eval']['extreme_threshold'],
                extreme_weight=cfg['model']['extreme_weight']
            )
        
        print("Using standard weighted MSE loss")
        return standard_loss_fn


if __name__ == '__main__':
    args = parse_args()
    os.environ['CONFIG_PATH'] = args.config

    # --------------------------------------------------------------
    # 0. Set random seeds for reproducibility
    # --------------------------------------------------------------
    cfg = load_config(args.config)
    
    # Get seed from config or use default
    seed = cfg.get('reproducibility', {}).get('random_seed', RANDOM_SEED)
    set_seed(seed)
    
    # Additional Python random seed
    random.seed(seed)
    
    print("="*80)
    print("Enhanced Transformer Training with Geographic & Temporal Layers")
    print("="*80)
    print(f"Random seed set to: {seed} for reproducibility")
    
    # --------------------------------------------------------------
    # 1. Config & device
    # --------------------------------------------------------------
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
    # 3. Build model with location coordinates for geographic attention
    # --------------------------------------------------------------
    input_dim = X_train.shape[-1]
    
    # Extract location coordinates for geographic attention
    location_coords = torch.tensor([
        [loc['lat'], loc['lon']] for loc in cfg['data']['locations']
    ], dtype=torch.float32)
    
    use_advanced = cfg['model'].get('use_advanced_layers', True)
    print(f"Building model with advanced layers: {use_advanced}")
    
    # Use encoder-decoder architecture (V3)
    model = get_model(
        'encoder_decoder',
        input_dim=input_dim,
        seq_len=cfg['data']['seq_len'],
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_encoder_layers=cfg['model'].get('num_encoder_layers', 4),
        num_decoder_layers=cfg['model'].get('num_decoder_layers', 2),
        embed_dim=cfg['model']['embed_dim'],
        dropout=cfg['model']['dropout'],
        num_locations=len(cfg['data']['locations']),
        feature_groups=cfg['model']['feature_groups'],
        location_coords=location_coords,
        use_advanced_layers=use_advanced,
        use_series_decomposition=cfg['model'].get('use_series_decomposition', False)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    if use_advanced:
        print("\nModel components:")
        print("  ✓ Geographic Attention Layer")
        print("  ✓ Multi-Scale Temporal Layer")
        print("  ✓ Weather Regime Adapter")
        print("  ✓ Domain-Aware Feature Embeddings")

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

    # Get loss function
    loss_fn = get_loss_function(cfg, use_physics=cfg['model'].get('use_physics_loss', False))
    
    # Mixed precision training (only for CUDA)
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # --------------------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------------------
    best_val = float('inf')
    patience = cfg['model'].get('patience', 15)
    patience_counter = 0
    start_epoch = 1
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

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

            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    pred = model(xb, locb)
                    loss = loss_fn(pred, yb, xb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model'].get('gradient_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb, locb)
                loss = loss_fn(pred, yb, xb)
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
                pred = model(xb, locb)
                val_loss += loss_fn(pred, yb, xb).item()

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
