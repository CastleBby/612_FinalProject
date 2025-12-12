# evaluate_multitask.py
# Evaluation for multi-task model (uses classification head for decisions)
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import argparse
from transformer_model import get_model, set_seed, RANDOM_SEED

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multi-task precipitation transformer')
    parser.add_argument('--config', default=os.environ.get('CONFIG_PATH', 'config.yaml'),
                        help='Path to config YAML (default: config.yaml or CONFIG_PATH env)')
    return parser.parse_args()


def unscale(y_scaled, scaler, precip_idx):
    """y_scaled: 1-D array of scaled precipitation values."""
    dummy = np.zeros((len(y_scaled), scaler.scale_.size))
    dummy[:, precip_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, precip_idx]


def classification_metrics(y_true, y_pred, threshold):
    """Compute POD, FAR, CSI for precipitation events above threshold."""
    actual_rain = y_true > threshold
    pred_rain   = y_pred > threshold

    tp = np.sum(actual_rain & pred_rain)
    fp = np.sum(~actual_rain & pred_rain)
    fn = np.sum(actual_rain & ~pred_rain)

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return pod, far, csi


if __name__ == '__main__':
    args = parse_args()
    os.environ['CONFIG_PATH'] = args.config
    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    data = np.load('processed_data.npz', allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    loc_test = data['loc_test']
    scaler = data['scaler'].item()
    precip_idx = cfg['data']['variables'].index('precipitation')

    # Load multi-task model
    num_encoder_layers = cfg['model'].get('num_encoder_layers', cfg['model']['num_layers'])

    model = get_model(
        'multitask',
        input_dim=X_test.shape[-1],
        seq_len=cfg['data']['seq_len'],
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=num_encoder_layers,
        embed_dim=cfg['model']['embed_dim'],
        dropout=cfg['model']['dropout'],
        num_locations=len(cfg['data']['locations']),
        feature_groups=cfg['model']['feature_groups'],
        # Architecture configuration flags (defaults match ver_4 settings)
        use_series_decomposition=cfg['model'].get('use_series_decomposition', False),
        use_decoder=cfg['model'].get('use_decoder', False),
        num_decoder_layers=cfg['model'].get('num_decoder_layers', 2),
        use_local_attention=cfg['model'].get('use_local_attention', False),
        local_kernel_size=cfg['model'].get('local_kernel_size', 3),
        encoder_causal=cfg['model'].get('encoder_causal', True)
    ).to(device)

    checkpoint = torch.load('best_multitask_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Run inference
    batch_size = cfg['model'].get('batch_size', 64)
    reg_predictions = []
    class_predictions = []

    print("Running inference on test set...")

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            batch_loc = torch.LongTensor(loc_test[i:i+batch_size]).to(device)
            batch_reg, batch_class = model(batch_X, batch_loc)
            reg_predictions.append(batch_reg.cpu().numpy())
            class_predictions.append(batch_class.cpu().numpy())

    reg_pred_scaled = np.concatenate(reg_predictions, axis=0)
    class_logits = np.concatenate(class_predictions, axis=0)  # Logits (raw outputs)

    # Convert logits to probabilities
    class_prob = 1.0 / (1.0 + np.exp(-class_logits))  # Sigmoid

    # Unscale regression predictions
    y_pred_amount = unscale(reg_pred_scaled, scaler, precip_idx)
    y_true = unscale(y_test, scaler, precip_idx)

    # Use classification head for rain/no-rain decisions
    # Convert probability to binary using 0.5 threshold
    class_threshold = 0.5  # Can be optimized
    y_pred_binary = (class_prob > class_threshold).astype(float)

    # Hybrid prediction: Use classification to decide rain/no-rain, regression for amount
    y_pred_hybrid = y_pred_amount * y_pred_binary  # If no rain predicted, set to 0

    print(f"Classification threshold: {class_threshold}")
    print(f"Predicted rain {y_pred_binary.mean()*100:.1f}% of the time")
    print()

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_hybrid))
    mae  = mean_absolute_error(y_true, y_pred_hybrid)

    rain_threshold = cfg['eval']['rain_threshold']
    pod, far, csi = classification_metrics(y_true, y_pred_hybrid, rain_threshold)

    extreme_thresh = np.percentile(y_true, cfg['eval']['extreme_threshold'] * 100)
    pod_ext, _, _ = classification_metrics(y_true, y_pred_hybrid, extreme_thresh)

    print("="*70)
    print("=== MULTI-TASK TRANSFORMER RESULTS (mm/h) ===")
    print("="*70)
    print(f"RMSE : {rmse:6.4f}")
    print(f"MAE  : {mae:6.4f}")
    print(f"CSI  : {csi:6.4f} | POD: {pod:6.4f} | FAR: {far:6.4f}")
    print(f"Extreme POD ({cfg['eval']['extreme_threshold']*100:.0f}th %): {pod_ext:6.4f}")

    # Persistence baseline (for comparison)
    persist_scaled = X_test[:, -1, precip_idx]
    persist_pred   = unscale(persist_scaled, scaler, precip_idx)

    rmse_p = np.sqrt(mean_squared_error(y_true, persist_pred))
    mae_p  = mean_absolute_error(y_true, persist_pred)
    pod_p, far_p, csi_p = classification_metrics(y_true, persist_pred, rain_threshold)
    pod_ext_p, _, _ = classification_metrics(y_true, persist_pred, extreme_thresh)

    print("\n" + "="*70)
    print("=== PERSISTENCE BASELINE ===")
    print("="*70)
    print(f"RMSE : {rmse_p:6.4f}")
    print(f"MAE  : {mae_p:6.4f}")
    print(f"CSI  : {csi_p:6.4f} | POD: {pod_p:6.4f} | FAR: {far_p:6.4f}")
    print(f"Extreme POD: {pod_ext_p:6.4f}")

    # Comparison
    print("\n" + "="*70)
    print("=== IMPROVEMENT OVER BASELINE ===")
    print("="*70)
    print(f"RMSE : {(rmse - rmse_p) / rmse_p * 100:+6.1f}%")
    print(f"MAE  : {(mae - mae_p) / mae_p * 100:+6.1f}%")
    print(f"CSI  : {(csi - csi_p) / csi_p * 100:+6.1f}%")
    print(f"POD  : {(pod - pod_p) / pod_p * 100:+6.1f}%")
    print(f"FAR  : {(far - far_p) / far_p * 100:+6.1f}%")
    print(f"Extreme POD: {(pod_ext - pod_ext_p) / pod_ext_p * 100:+6.1f}%")

    # Save outputs for further analysis
    np.savez('multitask_predictions.npz',
             y_true=y_true,
             y_pred_amount=y_pred_amount,
             y_pred_hybrid=y_pred_hybrid,
             class_prob=class_prob,
             y_pred_binary=y_pred_binary)

    print("\n" + "="*70)
    print("Predictions saved to: multitask_predictions.npz")
    print("="*70)
    
    # ========================================================================
    # Plot training and validation losses
    # ========================================================================
    print("\n" + "="*70)
    print("Generating loss plots...")
    print("="*70)
    
    log_file = 'train_multitask.log'
    if os.path.exists(log_file):
        try:
            # Read training log
            df_log = pd.read_csv(log_file)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Multi-Task Training History', fontsize=16, fontweight='bold')
            
            # Plot 1: Total Loss
            ax = axes[0, 0]
            ax.plot(df_log['Epoch'], df_log['Train_Total_Loss'], 
                   label='Train', linewidth=2, color='#2E86AB')
            ax.plot(df_log['Epoch'], df_log['Val_Total_Loss'], 
                   label='Validation', linewidth=2, color='#A23B72')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Total Loss', fontweight='bold')
            ax.set_title('Total Loss (Regression + Classification)', fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Regression Loss
            ax = axes[0, 1]
            ax.plot(df_log['Epoch'], df_log['Train_Reg_Loss'], 
                   label='Train', linewidth=2, color='#2E86AB')
            ax.plot(df_log['Epoch'], df_log['Val_Reg_Loss'], 
                   label='Validation', linewidth=2, color='#A23B72')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Regression Loss', fontweight='bold')
            ax.set_title('Regression Loss (Precipitation Amount)', fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Classification Loss
            ax = axes[1, 0]
            ax.plot(df_log['Epoch'], df_log['Train_Class_Loss'], 
                   label='Train', linewidth=2, color='#2E86AB')
            ax.plot(df_log['Epoch'], df_log['Val_Class_Loss'], 
                   label='Validation', linewidth=2, color='#A23B72')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Classification Loss', fontweight='bold')
            ax.set_title('Classification Loss (Rain/No-Rain)', fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Learning Rate
            ax = axes[1, 1]
            ax.plot(df_log['Epoch'], df_log['Learning_Rate'], 
                   linewidth=2, color='#F18F01')
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Learning Rate', fontweight='bold')
            ax.set_title('Learning Rate Schedule (Cosine Annealing)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig('training_history.jpg', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Training history plot saved to training_history.jpg")
            
            # Additional plot: Loss convergence
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(df_log['Epoch'], df_log['Train_Total_Loss'], 
                   label='Training Loss', linewidth=2.5, color='#2E86AB', alpha=0.8)
            ax.plot(df_log['Epoch'], df_log['Val_Total_Loss'], 
                   label='Validation Loss', linewidth=2.5, color='#A23B72', alpha=0.8)
            ax.fill_between(df_log['Epoch'], df_log['Train_Total_Loss'], 
                           alpha=0.3, color='#2E86AB')
            ax.fill_between(df_log['Epoch'], df_log['Val_Total_Loss'], 
                           alpha=0.3, color='#A23B72')
            
            # Mark best epoch
            best_epoch = df_log['Val_Total_Loss'].idxmin() + 1
            best_val_loss = df_log['Val_Total_Loss'].min()
            ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, 
                      label=f'Best Epoch: {best_epoch}')
            ax.scatter([best_epoch], [best_val_loss], color='green', s=200, 
                      zorder=5, edgecolors='black', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
            ax.set_ylabel('Total Loss', fontsize=14, fontweight='bold')
            ax.set_title('Training Convergence (Multi-Task Learning)', 
                        fontsize=16, fontweight='bold')
            ax.legend(frameon=True, shadow=True, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plt.savefig('loss_convergence.jpg', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Loss convergence plot saved to loss_convergence.jpg")
            
        except Exception as e:
            print(f"Warning: Could not generate loss plots: {e}")
    else:
        print(f"Warning: Training log not found at {log_file}")
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70)
