"""
NowcastingGPT Baseline Evaluation Script
Evaluate the trained VQ-VAE + GPT baseline model.
"""

import torch
import numpy as np
import pandas as pd
import yaml
import json
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader

from nowcastinggpt_vqvae import NowcastingGPT
from nowcasting_gpt_collector_hourly import HourlyWeatherDataset

# Disable tqdm progress bars in logs
os.environ['TQDM_DISABLE'] = '1'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_metrics(y_true, y_pred, y_true_binary, y_pred_probs, threshold=0.5, extreme_threshold=0.9):
    """Compute all evaluation metrics."""
    # Regression metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Classification metrics
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    
    # CSI
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # POD
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # FAR
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Extreme POD (90th percentile)
    extreme_val = np.percentile(y_true, extreme_threshold * 100)
    extreme_mask = y_true >= extreme_val
    extreme_pred = y_pred >= extreme_val
    extreme_pod = np.sum(extreme_mask & extreme_pred) / np.sum(extreme_mask) if np.sum(extreme_mask) > 0 else 0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'csi': float(csi),
        'pod': float(pod),
        'far': float(far),
        'extreme_pod': float(extreme_pod)
    }


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    all_y_true_binary = []
    all_y_pred_probs = []
    
    with torch.no_grad():
        for batch_sequence in test_loader:
            batch_sequence = batch_sequence.to(device)
            
            # Split sequence into input and target
            batch_x = batch_sequence[:, :-1]  # All but last timestep: (batch, 24)
            batch_y = batch_sequence[:, -1]   # Last timestep: (batch,)
            
            # Reshape to (batch, seq_len, 1)
            batch_x = batch_x.unsqueeze(-1)  # (batch, 24, 1)
            
            # Forward pass
            reconstructed, _ = model(batch_x)
            
            # Get prediction (last timestep)
            reg_pred = reconstructed[:, -1, 0]  # (batch,) precipitation value
            
            # For classification, use same prediction
            class_pred = reg_pred  # Binary classification based on threshold
            
            all_y_true.append(batch_y.cpu().numpy())
            all_y_pred.append(reg_pred.cpu().numpy())
            all_y_true_binary.append((batch_y > 0.1).cpu().numpy())
            all_y_pred_probs.append(torch.sigmoid(class_pred).cpu().numpy())
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_true_binary = np.concatenate(all_y_true_binary)
    y_pred_probs = np.concatenate(all_y_pred_probs)
    
    metrics = compute_metrics(y_true, y_pred, y_true_binary, y_pred_probs)
    
    return metrics


def compute_persistence_baseline(test_loader):
    """Compute persistence baseline (next = current)."""
    all_y_true = []
    all_y_pred = []
    all_y_true_binary = []
    all_y_pred_probs = []
    
    for batch_sequence in test_loader:
        # Persistence: predict t+1 = t
        batch_x = batch_sequence[:, :-1]  # All but last timestep
        batch_y = batch_sequence[:, -1]   # Last timestep (target)
        
        # Persistence prediction: use the last input value
        persistence_pred = batch_x[:, -1]  # Last timestep of input
        
        all_y_true.append(batch_y.cpu().numpy())
        all_y_pred.append(persistence_pred.cpu().numpy())
        all_y_true_binary.append((batch_y > 0.1).cpu().numpy())
        all_y_pred_probs.append((persistence_pred > 0.1).float().cpu().numpy())
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_true_binary = np.concatenate(all_y_true_binary)
    y_pred_probs = np.concatenate(all_y_pred_probs)
    
    metrics = compute_metrics(y_true, y_pred, y_true_binary, y_pred_probs)
    
    return metrics


def main():
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    output_dir = 'nowcastinggpt_baseline_results'
    
    # Data config
    seq_len = cfg['data']['seq_len']
    pred_horizon = cfg['data']['pred_horizon']
    rain_threshold = cfg['eval']['rain_threshold']
    
    print("="*80)
    print("NowcastingGPT Baseline Evaluation")
    print("="*80)
    print(f"Started at: {start_timestamp}")
    print(f"Output directory: {output_dir}")
    print(f"Rain threshold: {rain_threshold} mm/h\n")
    
    # Data paths
    data_dir = 'nowcasting_gpt_data'
    test_data_path = f'{data_dir}/formatted_data_test.csv'
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = HourlyWeatherDataset(
        test_data_path,
        obs_time=seq_len,
        pred_time=pred_horizon
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model checkpoint
    print("Loading model...")
    model_path = f'{output_dir}/nowcastinggpt_complete.pth'
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    model_cfg = checkpoint['model_config']
    
    # Initialize model with saved config
    model = NowcastingGPT(
        input_dim=model_cfg['input_dim'],
        hidden_dims=model_cfg['hidden_dims'],
        embedding_dim=model_cfg['embedding_dim'],
        num_embeddings=model_cfg['num_embeddings'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg['dim_feedforward']
    ).to(device)
    
    # Load weights
    model.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
    model.gpt.load_state_dict(checkpoint['gpt_state_dict'])
    
    # Print model info
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model size: {model_size_mb:.1f} MB")
    print(f"Model parameters: {total_params/1e6:.2f}M")
    print(f"Architecture: {checkpoint['model_info']['architecture']}")
    
    # Evaluate baseline
    print("\nEvaluating NowcastingGPT baseline...")
    baseline_metrics = evaluate_model(model, test_loader)
    
    # Compute persistence baseline
    print("Computing persistence baseline...")
    persistence_metrics = compute_persistence_baseline(test_loader)
    
    # Calculate improvements
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nNowcastingGPT Baseline:")
    print(f"  RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"  MAE:  {baseline_metrics['mae']:.4f}")
    print(f"  CSI:  {baseline_metrics['csi']:.4f}")
    print(f"  POD:  {baseline_metrics['pod']:.4f}")
    print(f"  FAR:  {baseline_metrics['far']:.4f}")
    print(f"  Extreme POD: {baseline_metrics['extreme_pod']:.4f}")
    
    print("\nPersistence Baseline:")
    print(f"  RMSE: {persistence_metrics['rmse']:.4f}")
    print(f"  MAE:  {persistence_metrics['mae']:.4f}")
    print(f"  CSI:  {persistence_metrics['csi']:.4f}")
    print(f"  POD:  {persistence_metrics['pod']:.4f}")
    print(f"  FAR:  {persistence_metrics['far']:.4f}")
    print(f"  Extreme POD: {persistence_metrics['extreme_pod']:.4f}")
    
    print("\nImprovement over Persistence:")
    print(f"  RMSE: {(baseline_metrics['rmse'] - persistence_metrics['rmse']) / persistence_metrics['rmse'] * 100:.1f}%")
    print(f"  MAE:  {(baseline_metrics['mae'] - persistence_metrics['mae']) / persistence_metrics['mae'] * 100:.1f}%")
    print(f"  CSI:  {(baseline_metrics['csi'] - persistence_metrics['csi']) / persistence_metrics['csi'] * 100:.1f}%")
    print(f"  POD:  {(baseline_metrics['pod'] - persistence_metrics['pod']) / persistence_metrics['pod'] * 100:.1f}%")
    print(f"  FAR:  {(baseline_metrics['far'] - persistence_metrics['far']) / persistence_metrics['far'] * 100:.1f}%")
    print(f"  Extreme POD: {(baseline_metrics['extreme_pod'] - persistence_metrics['extreme_pod']) / persistence_metrics['extreme_pod'] * 100:.1f}%")
    
    # Save metrics
    results = {
        'baseline': baseline_metrics,
        'persistence': persistence_metrics,
        'model_info': {
            'architecture': checkpoint['model_info']['architecture'],
            'total_parameters': int(total_params),
            'parameters_M': float(total_params / 1e6),
            'model_size_mb': float(model_size_mb)
        },
        'evaluation_time': time.time() - start_time,
        'timestamp': start_timestamp
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save timing info
    end_time = time.time()
    execution_time = end_time - start_time
    
    timing_info = {
        'start_time': start_timestamp,
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'execution_time_seconds': execution_time,
        'execution_time_minutes': execution_time / 60,
        'execution_time_hours': execution_time / 3600
    }
    
    with open(f'{output_dir}/timing_evaluation.json', 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    print(f"\nEvaluation time: {execution_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}/metrics.json")
    print("="*80)


if __name__ == '__main__':
    main()

