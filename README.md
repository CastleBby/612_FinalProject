# Precipitation Nowcasting (CMSC 612 Final Project)

This repository reproduces the end-to-end workflow for the 612 Final Project: fetching historical weather observations for several Maryland locations, training a transformer-based precipitation nowcasting model, and generating evaluation plots and metrics.

## 1. Environment Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda if you do not already have it.
2. Create and activate the project environment:
   ```bash
   conda create -n 612proj python=3.10 -y
   conda activate 612proj
   ```
3. Install the required Python packages:
   ```bash
   # Core dependencies
   conda install -y numpy pandas scikit-learn requests pyyaml tqdm matplotlib

   # PyTorch CPU build (install with pip to avoid libittnotify issues on non-Intel systems)
   pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

   # (Optional) CUDA 12.1 build — Python 3.13 currently has only torch, not torchvision/torchaudio
   pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

The environment now contains all libraries used by the data preparation, training, and analysis scripts.

## 2. Data Preparation

The project does not ship with raw data. Fetch and preprocess the required hourly weather history (2010–2020) from the Open-Meteo archive:

```bash
python data_loader.py
```

This script:

- Downloads the configured variables for five Maryland stations (see `config.yaml`).
- Cleans and interpolates small gaps in the time series.
- Scales features with a shared `MinMaxScaler`.
- Splits the sequences into train/validation/test partitions.
- Saves the artifacts to:
  - `md_weather_data.csv` — concatenated raw hourly observations.
  - `processed_data.npz` — tensors, location indices, and the fitted scaler serialized with NumPy.

Expect the download and preprocessing step to take a couple of minutes on a typical connection.

## 3. Model Training

Train the baseline transformer (single-head regression):
```bash
python train.py
```
Saves `best_model.pth` in the repo root.

Train the improved transformer (two-head regression + classifier, gated inference):
```bash
python improved_model/train.py
```
Saves `improved_model/best_model_deepened.pth`.

Device selection is automatic (CUDA → MPS → CPU). You can shorten runs by lowering `model.epochs` in the corresponding config.

## 4. Evaluation

Benchmark both baseline and improved models on the held-out test split:
```bash
python evaluate.py
```
The unified evaluator loads `best_model.pth` and `improved_model/best_model_deepened.pth`, restores the scaler, and prints:
- RMSE (root-mean-square error; lower is better)
- MAE (mean absolute error; lower is better)
- CSI/POD/FAR for rain events (higher CSI/POD is better; lower FAR is better)
- Extreme-event POD at the configured percentile

The improved model uses its classifier to gate low-confidence rain forecasts at inference, which reduces false alarms. The persistence baseline is no longer printed.

## 5. Visualizations and Analysis

- `analysis_suite.py` reproduces the publication figures (scatter plots, residual histograms, learning-rate ablations, attention maps, etc.) and saves them alongside the script (`fig1_*.png`, ..., `fig7_*.png`).
- `visualize.py` generates an interactive Matplotlib figure for the heaviest test-set event with past 24-hour context and the one-hour forecast.

Both scripts assume `processed_data.npz` and `best_model.pth` already exist.

## 6. Configuration

All tunable parameters live in `config.yaml`. Important sections:

- `data`: station coordinates, variables, sequence length, prediction horizon, and split ratios.
- `model`: transformer width/depth, learning rate, batch size, epochs, and feature-group definitions (used for domain-aware embeddings).
- `eval`: thresholds for rain and extreme-event detection.

Modify the configuration to experiment with different locations, variables, or training schedules. Rerun `data_loader.py` whenever the data block changes to regenerate aligned tensors and scaler state.

## 7. Troubleshooting

- **Conda environment permissions**: If you encounter “No writable envs directories configured,” ensure your Conda install has write access to its `envs/` directory, or export `CONDA_ENVS_PATH` and `CONDA_PKGS_DIRS` that point to writable locations before creating the environment.
- **PyTorch `iJIT_NotifyEvent` errors**: Prefer the official CPU wheel (`pip install ... torch==2.5.1`) instead of the Conda package to avoid missing Intel ITT instrumentation libraries on non-Intel hardware.
- **API limits**: The Open-Meteo archive is free but rate-limited. Re-running `data_loader.py` in quick succession may trigger temporary HTTP errors; retry after a short pause if that happens.

With these steps, you can fully reproduce the dataset preparation, transformer training, and evaluation workflow for the CMSC 612 final project.
