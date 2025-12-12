#!/bin/bash
#SBATCH --job-name=baseline_nowcastinggpt
#SBATCH --output=nowcastinggpt_baseline_results/baseline_%j.out
#SBATCH --error=nowcastinggpt_baseline_results/baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "======================================================================"
echo "NowcastingGPT Baseline Pipeline (VQ-VAE + GPT Transformer)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

## Load modules
#module load anaconda cuda/11.8

# Activate environment
source activate /home/shhuang/scratch/useNowcastingGPT_env/bin/activate

# Create output directory
mkdir -p nowcastinggpt_baseline_results

echo "----------------------------------------------------------------------"
echo "Step 0: Checking Data Availability"
echo "----------------------------------------------------------------------"
python check_and_prepare_data.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Data preparation failed. Pipeline cannot continue."
    exit 1
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 1: Training NowcastingGPT Baseline (VQ-VAE + GPT)"
echo "----------------------------------------------------------------------"
echo "This will train:"
echo "  - Stage 1: VQ-VAE (encoder + codebook + decoder)"
echo "  - Stage 2: GPT Transformer (autoregressive)"
echo ""

python train_nowcastinggpt_baseline.py 2>&1 | tee nowcastinggpt_baseline_results/train_baseline.log

if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed"
    exit 1
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 2: Evaluating Baseline Model"
echo "----------------------------------------------------------------------"

if [ ! -f "nowcastinggpt_baseline_results/nowcastinggpt_complete.pth" ]; then
    echo "✗ Model file not found"
    exit 1
fi

python evaluate_nowcastinggpt_baseline.py 2>&1 | tee nowcastinggpt_baseline_results/evaluate_baseline.log

if [ $? -eq 0 ]; then
    echo "✓ Evaluation completed successfully"
else
    echo "✗ Evaluation failed"
    exit 1
fi

echo ""
echo "----------------------------------------------------------------------"
echo "Step 3: Generating Visualizations"
echo "----------------------------------------------------------------------"

python visualize_nowcastinggpt.py 2>&1 | tee nowcastinggpt_baseline_results/visualize.log

if [ $? -eq 0 ]; then
    echo "✓ Visualization completed successfully"
else
    echo "✗ Visualization failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Baseline Pipeline Complete!"
echo "======================================================================"
echo "Results saved to: nowcastinggpt_baseline_results/"
echo "  - nowcastinggpt_complete.pth (full model)"
echo "  - vqvae_best.pth (Stage 1)"
echo "  - metrics.json"
echo "  - training_history.jpg"
echo "  - loss_convergence.jpg"
echo ""
echo "End time: $(date)"
echo "======================================================================"

