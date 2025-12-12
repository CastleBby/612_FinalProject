#!/usr/bin/env bash
#SBATCH --job-name=precip-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
# #SBATCH --account=<your_allocation>   # uncomment if required
## change above according to cluster 

## --------------------------------------------------
## configure above by checking the idle gpu on HPC
## sinfo -p gpu -t idle -o "%n %G"
## --------------------------------------------------

set -euo pipefail

# activate the repo-local venv you just made
# source "$SLURM_SUBMIT_DIR/venv/bin/activate"

# optional: log GPU + torch
# checks CUDA availibility
nvidia-smi || true
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

# Optimization: Set thread counts for better CPU performance
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Optimization: CUDA memory management for large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.9

# Optimization: Enable TF32 for faster training on H100/A100
export NVIDIA_TF32_OVERRIDE=1

# cd "$SLURM_SUBMIT_DIR"

# --------------------------------------------------
# data loading and training
# --------------------------------------------------
# Check if processed data exists
if [ ! -f "processed_data.npz" ]; then
    echo "processed_data.npz not found - running data_loader.py"
    srun -u python3 data_loader.py
else
    echo "processed_data.npz found - skipping data loading"
fi

# Train model
echo "Starting training..."
srun -u python3 train_multitask.py

# --------------------------------------------------
# evaluation
# --------------------------------------------------
echo "Starting evaluation..."
srun -u python3 evaluate_multitask.py

