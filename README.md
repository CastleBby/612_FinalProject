# Hourly Precipitation Forecasting with Encoder-Decoder Transformer

**Version 3: Inspired by "Attention is All You Need"**

---

## 🚀 **What's New in V3**

We redesigned the model from scratch using a proper **Encoder-Decoder architecture**:

| Change | V2 (Failed) | V3 (Current) |
|--------|-------------|--------------|
| **Encoder** | Causal masking (only past) | Bi-directional (full 24h context) ✅ |
| **Decoder** | None | With cross-attention to encoder ✅ |
| **Multi-scale** | Removed | Re-added (proven in main branch) ✅ |
| **Series Decomp** | None | Trend/seasonal separation ✅ |
| **Parameters** | 7.2M (overfitted) | 8.5M (better architecture) ✅ |
| **CSI** | 0.5857 ❌ | **TBD** (testing) 🚀 |

**Goal**: Beat main branch baseline (CSI > 0.6181)

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [Architecture](#architecture)
5. [Tools & Technologies](#tools--technologies)
6. [Environment Setup](#environment-setup)
7. [Code Entrypoint](#code-entrypoint)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results & Baseline Comparison](#results--baseline-comparison)
10. [Reproducibility](#reproducibility)
11. [HPC Commands](#hpc-commands)
12. [Limitations & Future Work](#limitations--future-work)

---

## 1. Introduction

### Project Overview

This project implements an **Encoder-Decoder Transformer** for hourly precipitation forecasting across 5 weather stations in Maryland. Inspired by "Attention is All You Need", the model properly separates:
1. **Encoder**: Processes full 24-hour input with bi-directional attention
2. **Decoder**: Generates prediction with cross-attention (no future leakage)

Using 25 years of hourly weather data (2000-2024), V3 incorporates:
- ✅ **Encoder-Decoder architecture** (proper separation of concerns)
- ✅ **Multi-scale attention** (1h, 6h, 24h patterns)
- ✅ **Series decomposition** (Autoformer-style trend/seasonal)
- ✅ **Cross-attention** (decoder learns which past hours matter)
- ✅ **Domain-aware embeddings** (weather + temporal features)

**Goal**: Beat main branch baseline (CSI: 0.6181) by leveraging better architecture

---

## 2. Problem Statement

**Challenge**: Predicting hourly precipitation is difficult because:
1. **Extreme events are rare** (~10% of hours have rain, <1% have heavy rain)
2. **Temporal dependencies** are complex (flash floods vs seasonal patterns)
3. **Limited spatial data** (only 5 stations in Maryland)
4. **Distribution is heavy-tailed** (most hours have 0mm, rare events have >20mm)

**Previous Approaches Failed** because:
- ❌ No causal masking → models saw future data → overfit
- ❌ Geographic attention useless with only 5 stations
- ❌ Too many parameters → overfitting on limited data

---

## 3. Proposed Solution

### Encoder-Decoder Transformer Architecture

**Core Innovation**: Separate encoding and decoding phases

```
┌───────────────────────────────────────────────────┐
│           PAST 24 HOURS (Observed)                │
│  [temp, humidity, precip, pressure, wind, ...]    │
└────────────────┬──────────────────────────────────┘
                 │
                 ▼
        ╔════════════════════╗
        ║   ENCODER          ║
        ║  (Bi-directional)  ║  ← Can see ALL 24h safely
        ║                    ║     (it's all past data!)
        ╚════════╦═══════════╝
                 │ Memory
                 │
        ╔════════╩═══════════╗
        ║   DECODER          ║
        ║  (Cross-Attention) ║  ← Queries encoder for
        ║                    ║     relevant past info
        ╚════════╦═══════════╝
                 │
                 ▼
        ┌────────────────────┐
        │  NEXT 1 HOUR       │
        │  (Precipitation)   │
        └────────────────────┘
```

**Why This Works Better**:
1. **Encoder doesn't need causal masking** - all input is from the past!
2. **Decoder learns what to attend to** - not all hours are equally important
3. **Cross-attention** - explicit mechanism to use encoder context
4. **Better gradient flow** - skip connections across encoder-decoder bridge

### Key Components

1. **Series Decomposition** (Autoformer-inspired)
   - Separates slow trends from seasonal patterns
   - Model focuses on residuals, not noise

2. **Multi-Scale Attention**
   - Different heads learn different temporal scales
   - Some focus on recent hours, others on daily cycles

3. **Domain-Aware Embeddings**
   - Weather features: thermo, hydro, dynamic groups
   - Temporal features: hour, day, month (cyclical)

This reduces false alarms (FAR) while maintaining high detection rate (POD).

### Why These Metrics?

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **RMSE/MAE** | Regression accuracy | Overall prediction quality |
| **CSI** | Hit rate minus false alarms | Balance of detection vs false alarms |
| **POD** | Probability of Detection | How many rain events we catch |
| **FAR** | False Alarm Ratio | How often we falsely predict rain |

For precipitation, CSI/POD/FAR matter because:
- Forecasters care about **event detection** (did it rain?)
- False alarms waste resources (emergency prep, evacuations)
- Missed events can be dangerous (flash floods)

---

## 4. Architecture

### Evolution: V3 - Encoder-Decoder Architecture

**Inspired by "Attention is All You Need" (Vaswani et al., 2017)**

After analyzing performance issues in previous versions, we redesigned the model using a proper **Encoder-Decoder** architecture with:
- ✅ **Bi-directional encoder** (can see full 24h input safely)
- ✅ **Causal decoder** (generates prediction without future leakage)
- ✅ **Cross-attention** (decoder attends to encoder output)
- ✅ **Multi-scale temporal attention** (1h, 6h, 24h patterns)
- ✅ **Series decomposition** (separates trend from seasonal components)

### Model Architecture Diagram

```
                    ┌──────────────────────────────────────┐
                    │      INPUT SEQUENCE (24 hours)       │
                    │   [temperature, humidity, precip,    │
                    │    pressure, wind, temporal_features]│
                    └───────────────┬──────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │   SERIES DECOMPOSITION (Autoformer)  │
                    │   • Moving Average (25-step kernel)  │
                    │   • Separates Trend + Seasonal       │
                    └───────────────┬──────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │    FEATURE EMBEDDING (Domain-Aware)  │
                    │   • Thermo: [temp, humidity]         │
                    │   • Hydro: [precip, pressure]        │
                    │   • Dynamic: [wind]                  │
                    │   • Temporal: [hour, day, month]     │
                    └───────────────┬──────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │  LOCATION + POSITIONAL EMBEDDINGS    │
                    │  • 5 station embeddings (learnable)  │
                    │  • Sinusoidal position encoding      │
                    └───────────────┬──────────────────────┘
                                    │
                                    │
  ╔═════════════════════════════════╩═════════════════════════════════╗
  ║                        ENCODER (Bi-directional)                    ║
  ║                                                                     ║
  ║  ┌────────────────────────────────────────────────────────────┐   ║
  ║  │      🌐 MULTI-SCALE ATTENTION LAYER                        │   ║
  ║  │  • Different heads attend to different time scales:        │   ║
  ║  │    - Short-term: Recent hours (1h resolution)              │   ║
  ║  │    - Medium-term: Convective buildup (6h patterns)         │   ║
  ║  │    - Long-term: Diurnal cycle (24h patterns)               │   ║
  ║  │  • Learnable scale factors per head                        │   ║
  ║  │  • Gated fusion: σ(gate) * cat(scales) + residual          │   ║
  ║  └────────────────────────────────────────────────────────────┘   ║
  ║                             │                                       ║
  ║                             ▼                                       ║
  ║  ┌────────────────────────────────────────────────────────────┐   ║
  ║  │   🔁 TRANSFORMER ENCODER LAYER (×4, Bi-directional)        │   ║
  ║  │                                                             │   ║
  ║  │   • Multi-Head Self-Attention (8 heads, NO causal mask)    │   ║
  ║  │     → Can attend to all 24h positions                      │   ║
  ║  │   • Feed-Forward Network (512 → 2048 → 512)               │   ║
  ║  │   • Pre-LayerNorm + Residual Connections                   │   ║
  ║  │   • GELU activation                                         │   ║
  ║  └────────────────────────────────────────────────────────────┘   ║
  ║                             │                                       ║
  ║                             ▼                                       ║
  ║  ┌────────────────────────────────────────────────────────────┐   ║
  ║  │              LAYER NORM (Final Encoding)                   │   ║
  ║  └────────────────────────────────────────────────────────────┘   ║
  ║                             │                                       ║
  ╚═════════════════════════════╪═══════════════════════════════════════╝
                                │
                                │ Memory (Encoder Output)
                                │
  ╔═════════════════════════════╪═══════════════════════════════════════╗
  ║                        DECODER (Causal)                             ║
  ║                             │                                       ║
  ║  ┌────────────────────────────────────────────────────────────┐   ║
  ║  │     🎯 LEARNABLE DECODER INPUT (Query)                     │   ║
  ║  │  • Shape: (batch, 1, 512)                                  │   ║
  ║  │  • Represents "what to predict" (next 1h precipitation)    │   ║
  ║  └────────────────────────────────────────────────────────────┘   ║
  ║                             │                                       ║
  ║                             ▼                                       ║
  ║  ┌────────────────────────────────────────────────────────────┐   ║
  ║  │    🔒 DECODER LAYER (×2, with Cross-Attention)             │   ║
  ║  │                                                             │   ║
  ║  │  1. Masked Self-Attention (causal, prevents future access) │   ║
  ║  │     └─> Only 1 query, so mask not strictly needed          │   ║
  ║  │                                                             │   ║
  ║  │  2. ⚡ CROSS-ATTENTION to Encoder Memory                   │   ║
  ║  │     • Query: from decoder                                  │   ║
  ║  │     • Key, Value: from encoder output                      │   ║
  ║  │     • Learns which encoder positions are important         │   ║
  ║  │                                                             │   ║
  ║  │  3. Feed-Forward Network (512 → 2048 → 512)               │   ║
  ║  │                                                             │   ║
  ║  └────────────────────────────────────────────────────────────┘   ║
  ║                             │                                       ║
  ║                             ▼                                       ║
  ║  ┌────────────────────────────────────────────────────────────┐   ║
  ║  │              LAYER NORM (Final Decoding)                   │   ║
  ║  └────────────────────────────────────────────────────────────┘   ║
  ╚═════════════════════════════╪═══════════════════════════════════════╝
                                │
                                ▼
                ┌──────────────────────────────────────┐
                │   ✨ IMPROVED OUTPUT HEAD             │
                │                                      │
                │  Layer 1: 512 → 256                 │
                │  Layer 2: 256 → 128 (+ skip from 512)│
                │  Layer 3: 128 → 64  (+ skip from 256)│
                │  Output:  64 → 1                    │
                │                                      │
                │  • Skip connections for gradient flow│
                │  • GELU activations                 │
                │  • LayerNorm + Dropout (0.2)        │
                └──────────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────────┐
                │   PRECIPITATION PREDICTION (mm/h)    │
                └──────────────────────────────────────┘
```

### Why Encoder-Decoder? (vs. Previous Encoder-only)

| Aspect | Encoder-only (V2) | Encoder-Decoder (V3) |
|--------|------------------|----------------------|
| **Input Processing** | Causal masking (only past) | Bi-directional (full 24h) ✅ |
| **Information Flow** | Limited by masking | Full context in encoder ✅ |
| **Prediction** | From last hidden state | Cross-attention to all encoder states ✅ |
| **Future Leakage** | Prevented by mask | Prevented by decoder architecture ✅ |
| **Gradient Flow** | Can be bottlenecked | Better via cross-attention ✅ |

**Key Insight**: For weather prediction, the **input sequence** (past 24h) is fully observed, so we don't need causal masking in the encoder! Causal masking is only needed when generating future predictions autoregressively.

---

## 5. Tools & Technologies

### Core Framework
- **PyTorch 2.0+**: Deep learning framework
- **CUDA 12.6**: GPU acceleration (H100/A100 support)

### Data & Processing
- **Open-Meteo API**: Historical weather data (25 years)
- **NumPy/Pandas**: Data preprocessing
- **Scikit-learn**: Train/test split, metrics

### Visualization
- **Matplotlib**: Loss curves, results
- **Seaborn**: Statistical plots

### HPC
- **SLURM**: Job scheduling
- **Multi-GPU**: 4× H100 (48GB memory)

---

## 6. Environment Setup

### Step 1: Create Virtual Environment

```bash
python3 -m venv 612_FinalProject_env
source 612_FinalProject_env/bin/activate
```

### Step 2: Install PyTorch with CUDA

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Step 3: Install Dependencies

```bash
pip3 install -r requirements.txt
# Installs: numpy, pandas, scikit-learn, PyYAML, requests, tqdm, matplotlib, seaborn
```

### Step 4: Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
```

---

## 7. Code Entrypoint

### HPC Training (SLURM)

```bash
# Submit job (runs train_multitask.py + evaluate_multitask.py)
sbatch run_train.sbatch

# Monitor
squeue -u $USER
tail -f slurm-JOBID.out

# Check results
ls -lh *.pth *.log *.jpg
```

### File Structure

```
612_FinalProject/
├── data_loader.py          # Data fetching & preprocessing
├── train_multitask.py      # Multi-task training (MAIN)
├── evaluate_multitask.py   # Evaluation + loss plots
├── transformer_model.py    # Model architecture
├── config.yaml             # Hyperparameters
├── run_train.sbatch        # HPC job script
├── requirements.txt        # Dependencies
└── README.md               # This file

Generated files:
├── md_weather_data.csv     # Raw data cache
├── processed_data.npz      # Preprocessed sequences
├── best_multitask_model.pth    # Trained model
├── train_multitask.log     # Training history
├── training_history.jpg    # Loss plots (4 subplots)
└── loss_convergence.jpg    # Single convergence plot
```

---

## 8. Evaluation Metrics

### Regression Metrics

- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
  ```
  RMSE = sqrt(mean((y_pred - y_true)²))
  ```

- **MAE** (Mean Absolute Error): Average prediction error
  ```
  MAE = mean(|y_pred - y_true|)
  ```

### Classification Metrics (Event Detection)

- **CSI** (Critical Success Index): Best overall metric
  ```
  CSI = TP / (TP + FP + FN)
  ```
  Balances hits vs false alarms. Higher is better.

- **POD** (Probability of Detection): Hit rate
  ```
  POD = TP / (TP + FN)
  ```
  Fraction of rain events correctly detected.

- **FAR** (False Alarm Ratio): False positive rate
  ```
  FAR = FP / (TP + FP)
  ```
  Fraction of predictions that were false alarms. Lower is better.

- **Extreme POD**: POD for heavy rain events (>90th percentile)

---

## 9. Results & Baseline Comparison

### Model Evolution Timeline

| Version | Architecture | CSI | RMSE | Status |
|---------|-------------|-----|------|--------|
| **Main Branch** | Simple Transformer + Multi-scale Attn | **0.6181** | 0.4132 | Baseline ✅ |
| V2 (Ver 2) | + Causal mask + Temporal conv + Recency | 0.5857 ❌ | 0.3798 | Overfitting |
| **V3 (Current)** | Encoder-Decoder + Multi-scale | **TBD** | **TBD** | Testing 🚀 |

### Performance Comparison with Main Branch

**Main Branch (Baseline)**:
- Architecture: Feature embeddings → Multi-scale attention (1h/6h/24h) → Transformer encoder
- Key innovation: Parallel attention at different resolutions
- Results:
  - ✅ **CSI: 0.6181** (better than persistence 0.6181)
  - RMSE: 0.3798
  - POD: 0.7102
  - Extreme POD: 0.7363

**V2 (Previous Attempt - FAILED)**:
- Problem: Overfitting with causal masking in encoder
  - CSI dropped to 0.5857 ❌
  - Added too many specialized layers
  - Restrictive physics loss

**V3 (Current - Encoder-Decoder)**:
- **Why it should work better**:
  1. ✅ Encoder can see full 24h context (no unnecessary masking)
  2. ✅ Decoder prevents future leakage through architecture
  3. ✅ Cross-attention learns which past hours matter most
  4. ✅ Series decomposition reduces noise
  5. ✅ Multi-scale attention (proven effective in main branch)
  
- **Expected improvements**:
  - Better gradient flow via cross-attention
  - More parameter-efficient (focused architecture)
  - Captures multi-scale patterns like main branch
  - Plus benefits of encoder-decoder structure

---

## 10. Reproducibility

### Random Seed: 202511

All random processes use **seed 202511** for reproducibility:

```python
# PyTorch
torch.manual_seed(202511)
torch.cuda.manual_seed_all(202511)
torch.backends.cudnn.deterministic = True

# NumPy
np.random.seed(202511)

# Python
random.seed(202511)

# Scikit-learn
train_test_split(..., random_state=202511)
```

### Configuration File

```yaml
# config.yaml
reproducibility:
  random_seed: 202511
```

### Reproducibility Checklist

- ✅ Fixed random seed in all files
- ✅ Deterministic CUDA operations
- ✅ No shuffling in temporal data split
- ✅ Fixed train/val/test split
- ✅ Same data preprocessing pipeline
- ✅ Same model initialization

**Result**: Run code twice → get identical results!

---

## 11. HPC Commands

### Job Submission

```bash
# Submit training job
sbatch run_train.sbatch

# Output: Submitted batch job JOBID
```

### Monitoring

```bash
# Check job status
squeue -u shhuang  # Replace with your username

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

**Optimizations enabled**:
- cuDNN benchmarking (faster convolutions)
- TF32 precision (faster on H100/A100)
- Parallel data loading (4 workers)
- Pinned memory (faster GPU transfer)
- Automatic Mixed Precision (AMP)

---

## 12. Architecture Evolution & Design Rationale

### From Main Branch to V3

#### Main Branch (Baseline - CSI: 0.6181)
**Architecture**:
```python
Input → Feature Embeddings → Multi-Scale Attention (1h/6h/24h) → Transformer Encoder → Output
```
**What worked**:
- ✅ Multi-scale attention captured different temporal patterns
- ✅ Simple and effective
- ✅ Beat persistence baseline

**What could improve**:
- Encoder-only architecture might bottleneck information
- Last hidden state pooling loses context

#### V2 (Failed - CSI: 0.5857 ❌)
**Changes from main**:
- Added causal masking to encoder (❌ MISTAKE!)
- Added temporal convolution layers
- Added recency-weighted attention
- Added physics-informed loss (too restrictive)

**Why it failed**:
1. ❌ **Causal masking in encoder was wrong**: Input is ALL past data, doesn't need masking!
2. ❌ **Too many specialized layers**: Overfitted on limited data (5 stations)
3. ❌ **Physics loss too restrictive**: Prevented model from learning patterns
4. ❌ **Lost what worked**: Removed multi-scale attention

**Lesson learned**: Adding complexity != better performance

#### V3 (Current - Testing 🚀)
**Architecture philosophy**: "Don't fix what ain't broke, but use better structure"

**What we kept from main branch**:
- ✅ Multi-scale attention (proven to work)
- ✅ Domain-aware embeddings
- ✅ Feature grouping

**What we improved**:
1. **Encoder-Decoder structure** (from "Attention is All You Need")
   - Encoder: Bi-directional (can see full 24h)
   - Decoder: Cross-attention (learns what matters)
   - Better than encoder-only + pooling

2. **Series Decomposition** (from Autoformer)
   - Separates trend from seasonal
   - Reduces noise in input

3. **Removed harmful components**:
   - ❌ No causal masking in encoder (not needed!)
   - ❌ No physics loss (too restrictive)
   - ❌ No geographic attention (useless with 5 stations)

**Why V3 should work**:
- ✅ Combines proven components (multi-scale from main branch)
- ✅ Better architecture (encoder-decoder > encoder-only)
- ✅ No unnecessary complexity
- ✅ Proper separation of concerns
