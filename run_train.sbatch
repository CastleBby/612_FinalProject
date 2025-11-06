#!/usr/bin/env bash
#SBATCH --job-name=precip-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
# #SBATCH --account=<your_allocation>   # uncomment if required
## change above according to cluster 

set -euo pipefail

# activate the repo-local venv you just made
# source "$SLURM_SUBMIT_DIR/venv/bin/activate"

# optional: log GPU + torch
# checks CUDA availibility
nvidia-smi || true
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.9

#cd "$SLURM_SUBMIT_DIR"

# prepare data (safe to re-run; comment out if processed_data.npz already exists)
srun -u python data_loader.py

# train
srun -u python train.py


#add more files to run...

