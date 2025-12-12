# Maryland Weather Forecasting - Multi-Task Transformer

**Branch**: `useNowcastingGPT` | **Seed**: 202511 | **Updated**: 2024-12-06

---

## Environment

**Python**: 3.8.16  
**Key packages**: PyTorch 1.11.0, pandas, numpy, pyyaml, matplotlib

**Setup (one-time)**:
```bash
bash setup_nowcasting_env.sh  # Creates 'nowcasting_env'
```

**Activate**:
```bash
module load anaconda cuda/11.8
source activate nowcasting_env
```

---

## Quick Start (Single Command)

### Option 1: NowcastingGPT Baseline (VQ-VAE + GPT)
```bash
sbatch run_baseline.sbatch
```
**Architecture**: VQ-VAE tokenizer + Autoregressive GPT Transformer (scaled for time-series)  
**Does everything**: Download data → Format → Train (2-stage) → Evaluate → Visualize  
**Training time**: ~2 hours on H100 (2-stage: VQ-VAE → GPT)

### Option 2: V5 NOVEL Transformer (5 Unique Innovations)
```bash
sbatch run_v5.sbatch
```
**Smart pipeline**:
- ✅ Checks if data exists (auto-downloads if missing)
- ✅ Validates each step's outputs before continuing
- ✅ Clear error messages if something fails
- ✅ Uses H100-optimized hyperparameters
- ✅ LSTM/GRU encoder + Multi-task heads

### Option 3: Advanced Pipeline Control (Optional)
```bash
# Check what files exist/missing
python run_pipeline.py --check-only

# Run only data preparation
python run_pipeline.py --pipeline data_only

# Resume from specific stage
python run_pipeline.py --start-from train_v5

# Force rerun everything
python run_pipeline.py --pipeline v5 --force
```

---

## Project Structure

```
# === ENTRY POINTS ===
run_baseline.sbatch                   # Baseline: VQ-VAE + GPT (NowcastingGPT-style)
run_v5.sbatch                         # V5 NOVEL: Novel Transformer with 5 Innovations

# === BASELINE MODEL (VQ-VAE + GPT) ===
nowcastinggpt_vqvae.py                # VQ-VAE + GPT architecture definition
train_nowcastinggpt_baseline.py       # 2-stage training (VQ-VAE → GPT)
evaluate_nowcastinggpt_baseline.py    # Compute metrics (to be created)
visualize_nowcastinggpt.py            # Generate plots (to be created)

# === V5 NOVEL MODEL (Novel Transformer with 5 Innovations) ===
model_v5.py                           # V5: Enhanced Transformer (ProbSparse, Instance Norm, Temporal Pooling)
train_v5_improved.py                  # Enhanced training: mixup, noise, warmup, cosine
evaluate_v5.py                        # Evaluation + baseline comparison
visualize_v5.py                       # Generate training plots

# === OLD V5 (DEPRECATED - for reference only) ===
# All old model files (model_v5_enhanced.py, model_v5_improved.py, model_v5_novel.py, etc.) have been DELETED
# Only model_v5.py exists now to avoid confusion
train_v5.py                           # OLD: No augmentation

# === DATA PIPELINE (SHARED) ===
nowcasting_gpt_data_downloader.py    # Download from Open-Meteo API (chunked, robust)
nowcasting_gpt_data_formatter.py     # Format to CSV (timestamp,precip)
nowcasting_gpt_collector_hourly.py   # PyTorch Dataset/DataLoader

# === UTILITIES ===
check_and_prepare_data.py            # Auto data checker & downloader
run_pipeline.py                       # Advanced pipeline control (optional)
config.yaml                           # Single source of truth (locations, hyperparams)
FIX_ISSUE.md                         # Troubleshooting & version history
```

---

## Data Format

**CSV**: `YYYYMMDDHHmm,precipitation_mm_per_hour`  
**Example**:
```
201306201200,28.53
201306201300,28.46
```

**Source**: Open-Meteo Historical API  
**Locations**: 5 Maryland stations (Baltimore, Annapolis, Cumberland, Ocean City, Hagerstown)  
**Date range**: 2000-2024 (25 years)  
**Expected rows**: ~1,095,000 (219,168 per location)

---

## Models Comparison

### Baseline: NowcastingGPT (Adapted for Time Series)
- **Parameters**: ~22M total (VQ-VAE: ~5M, GPT: ~17M)
- **Architecture**: VQ-VAE + GPT Transformer (two-stage, matching [original paper](https://github.com/Cmeo97/NowcastingGPT))
- **Training**: Stage 1 (VQ-VAE) + Stage 2 (GPT Transformer)
- **Purpose**: Proper implementation of NowcastingGPT adapted for time series
- **Adaptation**: Spatial (radar images) → Temporal (precipitation sequences)

**Our Baseline vs Original NowcastingGPT**:

| Aspect | Original NowcastingGPT | Our NowcastingGPT Baseline |
|--------|------------------------|----------------------------|
| **Architecture** | VQ-VAE + GPT Transformer ✓ | VQ-VAE + GPT Transformer ✓ |
| **Parameters** | ~400M | ~22M (scaled for 2h training) |
| **Input** | Radar images (spatial) | Time series (temporal) |
| **Tokenization** | VQ-VAE discrete tokens ✓ | VQ-VAE discrete tokens ✓ |
| **Encoder** | Spatial convolutions | Temporal (1D) convolutions |
| **Codebook** | Vector Quantization ✓ | Vector Quantization ✓ |
| **Transformer** | Autoregressive GPT ✓ | Autoregressive GPT ✓ |
| **Training** | Two-stage ✓ | Two-stage ✓ |
| **Model Type** | Generative | Generative |

✓ = Core architecture component preserved

---

### Architecture Diagrams

#### Our NowcastingGPT Baseline Architecture (Adapted for Time Series)
```
Input: [24 hours precipitation sequence]
         ↓
┌─────────────────────────────────────────┐
│ STAGE 1: VQ-VAE TOKENIZER (~5M params)  │
├─────────────────────────────────────────┤
│  Encoder (1D Conv layers)               │
│  - 64 → 128 → 256 channels              │
│  - Temporal downsampling                │
│  - Output: Continuous latent (256D)     │
│         ↓                                │
│  Vector Quantizer (Codebook)            │
│  - 512 discrete tokens                  │
│  - Commitment loss                      │
│  - Output: Discrete token indices       │
│         ↓                                │
│  Decoder (1D Transposed Conv)           │
│  - 256 → 128 → 64 channels              │
│  - Temporal upsampling                  │
│  - Reconstructs input                   │
└─────────────────────────────────────────┘
         ↓
   [Discrete Tokens]
         ↓
┌─────────────────────────────────────────┐
│ STAGE 2: GPT TRANSFORMER (~17M params)  │
├─────────────────────────────────────────┤
│  Token + Positional Embedding (256D)    │
│         ↓                                │
│  Transformer Decoder                    │
│  - 8 layers                             │
│  - 8 attention heads                    │
│  - 1024 FFN dimension                   │
│  - Causal masking (autoregressive)      │
│  - Dropout 0.1                          │
│         ↓                                │
│  Output Projection (256D → 512 vocab)   │
└─────────────────────────────────────────┘
         ↓
   [Predicted Future Tokens]
         ↓
   [VQ-VAE Decoder (from Stage 1)]
         ↓
Output: Future precipitation sequence

Training: Two-stage
  1. Train VQ-VAE (reconstruction)
  2. Train GPT (token prediction, VQ-VAE frozen)
```

#### Original NowcastingGPT Architecture (from [GitHub](https://github.com/Cmeo97/NowcastingGPT))
```
Input: [Radar Image Sequence]
         ↓
┌──────────────────────────────┐
│  Stage 1: VQ-VAE Tokenizer   │
│  - Encoder (spatial)         │
│  - Codebook (discrete tokens)│
│  - Decoder (reconstruction)  │
└──────────────────────────────┘
         ↓
   [Discrete Tokens]
         ↓
┌──────────────────────────────┐
│  Stage 2: GPT Transformer    │
│  - Autoregressive generation │
│  - 400M parameters           │
│  - Extreme Value Loss (EVL)  │
└──────────────────────────────┘
         ↓
   [Token Sequence]
         ↓
   [VQ-VAE Decoder]
         ↓
Output: Future radar images
```

**Key Difference**: Original NowcastingGPT processes **spatial data (radar images)** with a two-stage generative model, while our baseline processes **temporal data (time series)** with a single-stage predictive model.

#### Our V5: Enhanced Transformer with Proven Techniques
```
Input: [24 hours precipitation]
         ↓
┌─────────────────────────────────────────┐
│  VQ-VAE Encoder (ENABLED)               │
│  - Loaded from baseline                 │
│  - Discrete tokenization (256D)         │
│  - Frozen weights                       │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Input Projection                        │
│  - Concatenate: raw + VQ-VAE features   │
│  - Linear: (1+256) → 512                 │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Instance Normalization                 │
│  - Normalize each sequence independently│
│  - From foundation models (Chronos, TimesFM)│
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Positional Encoding                    │
│  - Sinusoidal (fixed)                   │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Pre-Norm Transformer (6 layers)         │
│  - ProbSparse Attention (Informer)      │
│    • O(L log L) complexity (not O(L²))   │
│    • Selects top queries efficiently    │
│  - 8 attention heads                     │
│  - d_model: 512, FFN: 2048              │
│  - Stochastic Depth (0.05)               │
│  - Pre-layer normalization              │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Temporal Attention Pooling             │
│  - Learned attention over timesteps      │
│  - Proven from archit_test               │
└─────────────────────────────────────────┘
         ↓
    ┌────────┴────────┐
    ↓                 ↓
[Regression Head]  [Classification Head]
3-layer MLP        3-layer MLP
512→256→128→1      512→256→128→1
    ↓                 ↓
  RMSE, MAE       CSI, POD, FAR, F1
  (mm/h)          (Rain/No-rain)
    └────────┬────────┘
             ↓
   Multi-Task Loss
   (0.7 × MSE + 0.3 × Focal Loss)

Data Augmentation (Training):
  • Mixup (α=0.2)
  • Gaussian noise (σ=0.01)
```

**V5: Key Enhancements (Built on Proven Techniques)**:

1. **ProbSparse Attention** (Informer)
   - Reduces complexity from O(L²) to O(L log L)
   - Selects top queries based on sparsity measurement
   - Efficient for long sequences (24h → future)

2. **Instance Normalization** (Foundation Models)
   - Normalizes each sequence independently
   - Critical for time series (from Chronos, TimesFM)
   - Better than batch normalization for forecasting

3. **Temporal Attention Pooling** (archit_test)
   - Learns which timesteps are most important
   - Proven to work better than simple mean pooling
   - Single-level (simpler than hierarchical)

4. **Pre-Norm Transformer** (archit_test)
   - More stable training than post-norm
   - Better gradient flow
   - 6 layers, 512d (deeper than baseline)

5. **VQ-VAE Integration** (Baseline Strength)
   - Uses baseline's noise reduction
   - Discrete tokenization for robustness
   - Frozen weights (no retraining)

**Additional Features**:
- **Stochastic Depth**: Progressive layer dropout (0.05)
- **Data Augmentation**: Mixup + Gaussian noise
- **Robust Losses**: MSE + Focal loss with label smoothing
- **Advanced Training**: Warmup + cosine annealing, gradient accumulation

**Target Performance**:
- Beat Baseline: RMSE < 0.3798, CSI > 0.5857
- Beat archit_test: RMSE < 0.3613, CSI > 0.6325

---

### V5: Enhanced Transformer with Proven Techniques
- **Architecture**: VQ-VAE Features → Input Projection → Instance Normalization → Pre-Norm Transformer (ProbSparse) → Temporal Attention Pooling → Multi-Task Heads
- **Model Size**: ~8-10M parameters (~35-40MB on disk)
- **Implementation**: `model_v5.py`, `train_v5_improved.py`
- **Model Config** (from `config.yaml` v5 section):
  - `d_model`: 512 (increased for better capacity)
  - `nhead`: 8 (Attention heads)
  - `num_encoder_layers`: 6 (deeper model)
  - `dim_feedforward`: 2048 (larger FFN)
  - `dropout`: 0.2 (balanced)
  - `head_dropout`: 0.3 (balanced)
  - `stochastic_depth`: 0.05 (layer dropout)
  - `use_probsparse`: true (Informer's ProbSparse attention)
  - `batch_size`: 512 (large batch for H100)
  - `gradient_accumulation_steps`: 2 (effective batch 1024)
  - `learning_rate`: 1e-4 (balanced)
  - `weight_decay`: 1e-3 (strong regularization)
  - `warmup_steps`: 1000 (learning rate warmup)
  - `epochs`: 150 (with early stopping on CSI+F1, patience=25)
- **Key Enhancements** (Built on Proven Techniques):
  1. **ProbSparse Attention** (Informer): O(L log L) complexity, selects top queries efficiently
  2. **Instance Normalization** (Foundation Models): Normalizes each sequence independently
  3. **Temporal Attention Pooling** (archit_test): Learns important timesteps
  4. **Pre-Norm Transformer** (archit_test): More stable training, better gradient flow
  5. **VQ-VAE Integration** (Baseline): Uses baseline's noise reduction, frozen weights
- **Additional Features**:
  - **VQ-VAE Integration**: `use_vqvae: true` (ENABLED for noise reduction from baseline)
  - **Pre-Norm Transformer**: More stable training than post-norm (from archit_test)
  - **Data Augmentation**: Mixup (α=0.2), Gaussian noise (σ=0.01)
  - **Stronger Regularization**: Dropout 0.2, weight decay 1e-3, stochastic depth 0.05, label smoothing 0.1
  - **Robust Losses**: Huber (vs MSE), Focal with label smoothing (0.1)
  - **Extreme Reweighting**: 2.0x for 90th percentile events
  - **Advanced Scheduler**: Warmup (1000 steps) + cosine annealing
  - **Enhanced Monitoring**: F1, precision, recall (combined CSI+F1 early stopping)
- **Purpose**: Beat both baselines with novel architecture (Target: RMSE < 0.3613, CSI > 0.6325)
- **Research Contribution**: First adaptive VQ-VAE + Transformer fusion, explicit extreme event pathway, multi-scale + hierarchical architecture
- **Note**: All hyperparameters from `config.yaml` (seed: 202511)

---

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **CSI** | Critical Success Index | **PRIMARY - Higher is better** |
| **POD** | Probability of Detection | Higher is better |
| **FAR** | False Alarm Ratio | Lower is better |
| **Extreme POD** | Detection rate for >90th percentile | Higher is better |

**Rain threshold**: 0.1 mm/h

---

## Results History

### Persistence Baseline (Reference)
```
RMSE: 0.4009 | MAE: 0.0820 | CSI: 0.6145 | POD: 0.7612 | FAR: 0.2388 | Extreme POD: 0.7459
```

### Simple Transformer Baseline (Pending)
**Status**: Awaiting complete data download (all 5 locations)  
**Model**: 2.3M parameters, 4-layer encoder  
**Expected**: CSI 0.62-0.64, RMSE 0.36-0.39  
**Note**: This is NOT the 400M-param NowcastingGPT from literature

### V5 NOVEL: Novel Transformer with 5 Unique Innovations (Current - H100 Optimized)
**Status**: Ready for training  
**Model**: ~20M parameters (d_model=512, 6 encoder + 3 decoder layers)  
**Hardware**: Optimized for H100 GPUs (batch_size=256, lr=2e-4)  
**Target**: Beat simple transformer baseline with:
- Better CSI (>0.65) through focal loss + direct CSI optimization
- Better RMSE (<0.33) through enhanced regression head + larger capacity
- Better extreme event detection through deeper classification head
- Faster convergence with optimized hyperparameters

---

## Output Files

### After running `sbatch run_nowcasting_baseline.sbatch`:
```
nowcasting_baseline_results/
├── metrics.json              # All metrics in JSON
├── training_history.jpg      # 2-panel: loss curves + LR schedule
├── loss_convergence.jpg      # Convergence with best epoch marked
└── best_baseline_model.pth   # Trained model

nowcasting_gpt_data/
├── raw_weather_data.csv      # Downloaded data (~1.1M rows)
└── formatted_data_*.csv      # Train/val/test splits
```

### After running `sbatch run_v5.sbatch`:
```
v5_results/
├── metrics.json              # All metrics + baseline comparison
├── training_history.jpg      # 6-panel: losses, RMSE, CSI, POD/FAR
├── loss_convergence.jpg      # Convergence with best CSI marked
├── best_v5_model.pth         # Trained model (~20MB for 5M params)
├── train_v5.log              # Training log
└── evaluate_v5.log           # Evaluation log
```

**Model Size Comparison**:
- Old V4: 231MB (many features, different architecture)
- Simple Baseline: 8.9MB (2.3M params)
- V5 NOVEL: ~25-30MB (6-7M params, novel architecture with 5 innovations)

---

## Manual Step-by-Step

### Baseline Model
```bash
# 1. Download data (15-20 min, chunked by 5-year periods)
python nowcasting_gpt_data_downloader.py
# Verify: Should show "5/5 locations successful, ~1,095,000 records"

# 2. Format to CSV
python nowcasting_gpt_data_formatter.py

# 3. Train baseline
python nowcasting_gpt_train_baseline.py

# 4. Evaluate baseline
python nowcasting_gpt_evaluate_baseline.py

# 5. Visualize
python visualize_baseline.py
```

### V5 NOVEL Model (after data download)
```bash
# Train V5 NOVEL (uses existing formatted data + VQ-VAE features)
python train_v5_improved.py

# Evaluate V5 NOVEL
python evaluate_v5.py

# Visualize V5 NOVEL
python visualize_v5.py
```

---

## Troubleshooting

### Missing Data Files
**Error**: `FileNotFoundError: nowcasting_gpt_data/formatted_data_train.csv`

**Solution**:
```bash
# Quick fix: Auto-download and format
python check_and_prepare_data.py

# Or use robust pipeline
python run_pipeline.py --pipeline data_only
```

### Incomplete Data (< 1.1M rows)
```bash
rm -rf nowcasting_gpt_data/
python check_and_prepare_data.py --force
```

### CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Edit `train_v5_improved.py` line ~196:
```python
batch_size = 128  # Reduce from 256
# Or reduce model size
d_model = 256  # Reduce from 512
```

### Training Failed Silently
Check if files exist:
```bash
ls -lh v5_results/best_v5_model.pth
ls -lh v5_results/training_history.npy
```

Use pipeline checker:
```bash
python run_pipeline.py --check-only
```

### Environment Issues
```bash
conda env remove -n nowcasting_env
bash setup_nowcasting_env.sh
```

---

## Key Features

- ✅ Single command execution
- ✅ Reproducible (seed 202511)
- ✅ Chunked download (handles API limits)
- ✅ All 6 metrics computed
- ✅ Visualizations auto-generated
- ✅ HPC-ready (SLURM)

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Download & Format Data (run once)                       │
│    → sbatch run_nowcasting_baseline.sbatch                  │
│    → Gets ~1.1M rows from 5 locations (2000-2024)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Train Baseline (NowcastingGPT)                          │
│    → Simple transformer for benchmarking                    │
│    → Expected: CSI ~0.62-0.64, RMSE ~0.36-0.39             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Train V5 NOVEL (Novel Transformer with 5 Innovations)    │
│    → sbatch run_v5.sbatch                                   │
│    → Uses same data, enhanced architecture                  │
│    → Target: Beat baseline on CSI and RMSE                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Compare Results                                          │
│    → V5 metrics.json includes baseline comparison          │
│    → Check improvement percentages                          │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Complete data download**: Fix incomplete data issue (get all 5 locations)
2. **Train baseline**: Establish NowcastingGPT benchmark
3. **Train V5 NOVEL**: Run novel transformer with 5 unique innovations
4. **Compare**: Verify V5 NOVEL beats both baselines (Target: RMSE < 0.3613, CSI > 0.6325)
5. **Iterate**: Tune hyperparameters if needed (learning rate, dropout, etc.)


---

## HPC Commands

### Job Submission

```bash
# Submit training job
sbatch run_train.sbatch
# Output: Submitted batch job JOBID
```

### Monitoring

```bash
# Check job status
squeue -u [USERNAME]  # Replace with your username
squeue -u shhuang
```

#### keep watching / monitoring
```bash
watch -n 1 squeue -u [USERNAME]
watch -n 1 squeue -u shhuang
```
```bash
# Check available GPUs
sinfo -p gpu -t idle -o "%n %G"

# Monitor log in real-time
tail -f slurm-JOBID.out

# Check file sizes
du -h --max-depth=1
```

### Job Management

```bash
# Cancel job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Job details
scontrol show job JOBID
```

### HPC Configuration

Current setup (`run_train.sbatch`):
```bash
#SBATCH --gres=gpu:h100:4      # 4× H100 GPUs
#SBATCH --cpus-per-task=8      # 8 CPU cores
#SBATCH --mem=48G              # 48GB RAM
#SBATCH --time=08:00:00        # 8 hour limit
```

---
