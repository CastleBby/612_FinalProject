"""
Training Script for Proper NowcastingGPT Baseline
Two-stage training matching original architecture:
  Stage 1: Train VQ-VAE (encoder + codebook + decoder)
  Stage 2: Train GPT Transformer (autoregressive token prediction)

Scaled to ~20-30M parameters for 2-hour training on H100.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader

from nowcastinggpt_vqvae import NowcastingGPT, VQVAE, GPTDecoder
from nowcasting_gpt_collector_hourly import HourlyWeatherDataset

# Disable tqdm for clean logs
os.environ['TQDM_DISABLE'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_vqvae_epoch(model, dataloader, optimizer):
    """Train VQ-VAE for one epoch."""
    model.train()
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    for batch_sequence in dataloader:
        batch_sequence = batch_sequence.to(device)
        
        # VQ-VAE expects (batch, seq_len, 1)
        # batch_sequence is (batch, seq_len), reshape to (batch, seq_len, 1)
        batch_x = batch_sequence[:, :-1].unsqueeze(-1)  # (batch, 24, 1)
        
        # Forward pass
        x_recon, vq_loss, perplexity, _ = model(batch_x)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(x_recon, batch_x)
        
        # Total loss
        loss = recon_loss + vq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        num_batches += 1
    
    return (total_recon_loss / num_batches, 
            total_vq_loss / num_batches,
            total_perplexity / num_batches)


@torch.no_grad()
def validate_vqvae(model, dataloader):
    """Validate VQ-VAE."""
    model.eval()
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    for batch_sequence in dataloader:
        batch_sequence = batch_sequence.to(device)
        # Reshape to (batch, seq_len, 1)
        batch_x = batch_sequence[:, :-1].unsqueeze(-1)  # (batch, 24, 1)
        
        x_recon, vq_loss, perplexity, _ = model(batch_x)
        recon_loss = nn.MSELoss()(x_recon, batch_x)
        
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        num_batches += 1
    
    return (total_recon_loss / num_batches,
            total_vq_loss / num_batches,
            total_perplexity / num_batches)


def train_gpt_epoch(vqvae, gpt, dataloader, optimizer):
    """Train GPT transformer for one epoch."""
    vqvae.eval()  # VQ-VAE is frozen
    gpt.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_sequence in dataloader:
        batch_sequence = batch_sequence.to(device)
        # Reshape to (batch, seq_len, 1)
        batch_x = batch_sequence[:, :-1].unsqueeze(-1)  # (batch, 24, 1)
        
        # Encode to discrete tokens (no gradients through VQ-VAE)
        with torch.no_grad():
            token_indices = vqvae.encode(batch_x)
        
        # GPT predicts next token
        logits = gpt(token_indices[:, :-1], causal_mask=True)  # Shift by 1
        targets = token_indices[:, 1:]  # Next tokens
        
        # Cross-entropy loss
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def validate_gpt(vqvae, gpt, dataloader):
    """Validate GPT transformer."""
    vqvae.eval()
    gpt.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_sequence in dataloader:
        batch_sequence = batch_sequence.to(device)
        # Reshape to (batch, seq_len, 1)
        batch_x = batch_sequence[:, :-1].unsqueeze(-1)  # (batch, 24, 1)
        
        token_indices = vqvae.encode(batch_x)
        logits = gpt(token_indices[:, :-1], causal_mask=True)
        targets = token_indices[:, 1:]
        
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Script started at: {start_timestamp}")
    print()
    
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set random seed
    random_seed = cfg['reproducibility']['random_seed']
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Output directory
    output_dir = 'nowcastinggpt_baseline_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data config
    seq_len = cfg['data']['seq_len']
    batch_size = 128  # Smaller batch for larger model
    
    print("="*80)
    print("NowcastingGPT Baseline Training (Proper Architecture)")
    print("="*80)
    print(f"Random seed: {random_seed}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    data_dir = 'nowcasting_gpt_data'
    
    train_dataset = HourlyWeatherDataset(
        f'{data_dir}/formatted_data_train.csv',
        obs_time=seq_len,
        pred_time=1
    )
    
    val_dataset = HourlyWeatherDataset(
        f'{data_dir}/formatted_data_val.csv',
        obs_time=seq_len,
        pred_time=1
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize NowcastingGPT model (scaled for 2-hour training)
    print("\nInitializing model...")
    model = NowcastingGPT(
        input_dim=1,
        hidden_dims=[64, 128, 256],
        embedding_dim=256,
        num_embeddings=512,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    vqvae_params = sum(p.numel() for p in model.vqvae.parameters())
    gpt_params = sum(p.numel() for p in model.gpt.parameters())
    
    print(f"\nModel Parameters:")
    print(f"  VQ-VAE (Stage 1): {vqvae_params/1e6:.2f}M")
    print(f"  GPT Transformer (Stage 2): {gpt_params/1e6:.2f}M")
    print(f"  Total: {total_params/1e6:.2f}M")
    
    #########################################
    # Stage 1: Train VQ-VAE
    #########################################
    print("\n" + "="*80)
    print("STAGE 1: Training VQ-VAE (Encoder + Codebook + Decoder)")
    print("="*80)
    
    vqvae_epochs = 30
    vqvae_optimizer = optim.AdamW(model.vqvae.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_vqvae_loss = float('inf')
    stage1_start = time.time()
    
    for epoch in range(vqvae_epochs):
        train_recon, train_vq, train_perp = train_vqvae_epoch(model.vqvae, train_loader, vqvae_optimizer)
        val_recon, val_vq, val_perp = validate_vqvae(model.vqvae, val_loader)
        
        total_train_loss = train_recon + train_vq
        total_val_loss = val_recon + val_vq
        
        print(f"Epoch {epoch+1:02d}/{vqvae_epochs} | "
              f"Train: recon={train_recon:.4f} vq={train_vq:.4f} perp={train_perp:.1f} | "
              f"Val: recon={val_recon:.4f} vq={val_vq:.4f} perp={val_perp:.1f}")
        
        # Save best model
        if total_val_loss < best_vqvae_loss:
            best_vqvae_loss = total_val_loss
            torch.save(model.vqvae.state_dict(), f'{output_dir}/vqvae_best.pth')
            print(f"  --> Saved VQ-VAE (val_loss={total_val_loss:.4f})")
    
    stage1_time = time.time() - stage1_start
    print(f"\nStage 1 completed in {stage1_time/60:.1f} minutes")
    
    # Load best VQ-VAE
    model.vqvae.load_state_dict(torch.load(f'{output_dir}/vqvae_best.pth'))
    
    #########################################
    # Stage 2: Train GPT Transformer
    #########################################
    print("\n" + "="*80)
    print("STAGE 2: Training GPT Transformer (Autoregressive)")
    print("="*80)
    
    gpt_epochs = 40
    gpt_optimizer = optim.AdamW(model.gpt.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_gpt_loss = float('inf')
    stage2_start = time.time()
    
    for epoch in range(gpt_epochs):
        train_loss = train_gpt_epoch(model.vqvae, model.gpt, train_loader, gpt_optimizer)
        val_loss = validate_gpt(model.vqvae, model.gpt, val_loader)
        
        print(f"Epoch {epoch+1:02d}/{gpt_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_gpt_loss:
            best_gpt_loss = val_loss
            # Save complete model
            torch.save({
                'epoch': epoch,
                'vqvae_state_dict': model.vqvae.state_dict(),
                'gpt_state_dict': model.gpt.state_dict(),
                'model_config': {
                    'input_dim': 1,
                    'hidden_dims': [64, 128, 256],
                    'embedding_dim': 256,
                    'num_embeddings': 512,
                    'd_model': 256,
                    'nhead': 8,
                    'num_layers': 8,
                    'dim_feedforward': 1024
                },
                'model_info': {
                    'architecture': 'NowcastingGPT (VQ-VAE + GPT)',
                    'total_parameters': int(total_params),
                    'vqvae_parameters': int(vqvae_params),
                    'gpt_parameters': int(gpt_params)
                }
            }, f'{output_dir}/nowcastinggpt_complete.pth')
            print(f"  --> Saved complete model (val_loss={val_loss:.4f})")
    
    stage2_time = time.time() - stage2_start
    print(f"\nStage 2 completed in {stage2_time/60:.1f} minutes")
    
    # Calculate total time
    total_time = time.time() - start_time
    end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Started: {start_timestamp}")
    print(f"Ended: {end_timestamp}")
    print(f"Stage 1 (VQ-VAE): {stage1_time/60:.1f} minutes")
    print(f"Stage 2 (GPT): {stage2_time/60:.1f} minutes")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\nModel saved to: {output_dir}/nowcastinggpt_complete.pth")
    print(f"Model size: ~{total_params/1e6:.1f}M parameters")
    print("="*80)


if __name__ == '__main__':
    main()

