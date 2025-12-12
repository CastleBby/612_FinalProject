"""
NowcastingGPT Baseline Visualization Script
Generate training history and loss convergence plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime

# Disable tqdm progress bars
os.environ['TQDM_DISABLE'] = '1'


def plot_training_history(history, output_dir):
    """Generate training history plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NowcastingGPT Baseline Training History', fontsize=16, fontweight='bold')
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss
    axes[0, 1].plot(epochs, history['train_recon_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_recon_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: VQ Loss (if available)
    if 'train_vq_loss' in history:
        axes[1, 0].plot(epochs, history['train_vq_loss'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, history['val_vq_loss'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('VQ Loss')
        axes[1, 0].set_title('Vector Quantization Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Perplexity (if available)
    if 'train_perplexity' in history:
        axes[1, 1].plot(epochs, history['train_perplexity'], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, history['val_perplexity'], 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].set_title('Codebook Perplexity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history saved to: {output_dir}/training_history.jpg")


def plot_loss_convergence(history, output_dir):
    """Generate loss convergence plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, alpha=0.7)
    
    # Mark best model
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = np.min(history['val_loss'])
    ax.plot(best_epoch, best_val_loss, 'r*', markersize=20, label=f'Best (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('NowcastingGPT Baseline - Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate(f'Best Val Loss: {best_val_loss:.4f}',
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + len(epochs)*0.1, best_val_loss + (max(history['val_loss']) - min(history['val_loss']))*0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_convergence.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Loss convergence saved to: {output_dir}/loss_convergence.jpg")


def main():
    # Start timing
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    output_dir = 'nowcastinggpt_baseline_results'
    
    print("="*80)
    print("NowcastingGPT Baseline Visualization")
    print("="*80)
    print(f"Started at: {start_timestamp}")
    print(f"Output directory: {output_dir}\n")
    
    # Load training history
    history_file = f'{output_dir}/training_history.npy'
    if not os.path.exists(history_file):
        print(f"✗ Training history not found: {history_file}")
        print("  Please run training first.")
        return
    
    history = np.load(history_file, allow_pickle=True).item()
    print(f"✓ Loaded training history ({len(history['train_loss'])} epochs)")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_training_history(history, output_dir)
    plot_loss_convergence(history, output_dir)
    
    # Save timing info
    end_time = time.time()
    execution_time = end_time - start_time
    
    timing_info = {
        'start_time': start_timestamp,
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'execution_time_seconds': execution_time,
        'execution_time_minutes': execution_time / 60
    }
    
    with open(f'{output_dir}/timing_visualization.json', 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    print(f"\nVisualization time: {execution_time:.1f} seconds")
    print("="*80)


if __name__ == '__main__':
    main()

