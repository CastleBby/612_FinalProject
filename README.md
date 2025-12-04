# Precipitation Nowcasting (DATA612 Final Project)

# 🌧️ 1-Hour Precipitation Nowcasting in Maryland
### A Domain-Aware Transformer with Multi-Scale Attention  
**Group 12 — Deep Learning (Prof. Samet Ayhan, Fall 2025)**  

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-OpenMeteo-blue)](https://open-meteo.com/)
[![Status](https://img.shields.io/badge/Stage-Interim%20Report-yellow.svg)]()

---

## 📘 Project Overview
Flash floods in Maryland are triggered by rapidly evolving, localized precipitation.  
Traditional **Numerical Weather Prediction (NWP)** systems underperform for 1-hour horizons due to:
- Coarse spatial resolution (>10 km)  
- Sensitivity to initial conditions  
- Weak handling of short-term, high-impact phenomena  

This project proposes a **Domain-Aware Transformer** architecture for **1-hour nowcasting**, using **24-hour input sequences** with meteorological embeddings and multi-scale attention.

### 🔬 Objectives
| Innovation | Description |
|-------------|-------------|
| 🌦️ Weather-specific embeddings | Captures physical variable semantics |
| ⏱️ Multi-scale temporal attention | Models dependencies at 1h, 6h, 24h |
| ⚖️ Precipitation-weighted loss | Emphasizes rare, high-rainfall events |

---

## 🧩 Dataset and Preprocessing

### Data Source
- **Provider:** [Open-Meteo Historical API](https://open-meteo.com)
- **Years:** 2010–2020 (11 years)
- **Frequency:** Hourly
- **Stations:** Baltimore, Annapolis, Cumberland, Ocean City, Hagerstown  

**Variables:** Temperature (°C), Relative humidity (%), Precipitation (mm/h), Surface pressure (hPa), Wind speed (m/s)

| Stage | Count |
|--------|-------|
| Raw records | 482,160 |
| After interpolation | 482,160 (0 % missing) |
| Sequences (24h→1h) | 482,136 |
| Train / Val / Test | 70 / 15 / 15 → 337,494 / 72,321 / 72,321 |

### Preprocessing Pipeline (Fully Implemented)

```python
# 1. Fetch data per station via API
df = fetch_openmeteo_data(lat, lon, '2010-01-01', '2020-12-31', variables)

# 2. Handle missing values (linear interpolation)
df = df.interpolate(method='linear').dropna()

# 3. Global MinMax scaling (consistent across all stations)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[variables])

# 4. Create 24h → 1h sequences
X, y = [], []
for i in range(len(data_scaled) - 24):
    X.append(data_scaled[i:i+24])           # Input: (24, 5)
    y.append(data_scaled[i+24, precip_idx]) # Target: scalar
```
**Rationale**  
- **Global scaling** → consistent feature ranges across all 5 stations  
- **No shuffling** → preserves temporal order (critical for time-series)  
- **Location ID** → added as learnable embedding (not one-hot)

All 482,136 sequences saved in processed_data.npz — ready for training in seconds.

🧠 Model Architecture — Design Summary

| Goal                          | Implementation                                    |
|-------------------------------|----------------------------------------------------|
| Capture variable semantics    | Separate embedding paths (thermo / hydro / dynamic) |
| Multi-scale temporal behavior | Parallel attention heads (1h, 6h, 24h)              |
| Prioritize extreme rain       | Weighted MSE loss (5× for >90th percentile)        |

Multi-Scale Attention Module
```python
class MultiScaleAttention(nn.Module):
    def forward(self, x):
        a1  = attn1h(x, x, x)                    # 1h resolution
        a6  = interpolate(attn6h(x[::6]), size=24)  # 6h → upsample
        a24 = interpolate(attn24h(x[::24]), size=24) # 24h → upsample
        cat = torch.cat([a1, a6, a24], dim=-1)
        return out(cat) * torch.sigmoid(gate(cat)) + x  # Gated residual
```

**Scale Interpretations**

- **1h head** → short-term fluctuations  
- **6h head** → convective buildup  
- **24h head** → diurnal cycle  
- **Gated fusion** → adaptively blends multi-scale features

Loss Function
```python
weights = torch.ones_like(targets)
weights[targets > quantile_90] = 5.0
loss = F.mse_loss(preds, targets, reduction='none') * weights
loss = loss.mean()
```
📈 Why?
Precipitation is >90% zero — standard MSE ignores heavy events.
Weighted loss prioritizes high-impact conditions.

⚙️ Training Pipeline

| Step | Script                 | Description                                      |
|------|------------------------|--------------------------------------------------|
| 1    | `data_loader.py`       | Fetch + preprocess → `processed_data.npz`        |
| 2    | `transformer_model.py` | Full domain-aware model                          |
| 3    | `train.py`             | AdamW, early stopping, weighted loss             |
| 4    | `evaluate.py`          | RMSE, CSI, POD, FAR + persistence baseline       |
| 5    | `analysis_suite.py`    | 7 publication-quality figures                    |

📈 Results (Test Set)

| Metric       | Our Model | Persistence |
|--------------|-----------|-------------|
| RMSE         | **0.3798** | 0.4132     |
| MAE          | 0.1398    | 0.0845     |
| CSI          | 0.5857    | 0.6181     |
| POD          | 0.7102    | 0.7640     |
| FAR          | 0.2304    | 0.2361     |
| Extreme POD  | 0.7363    | 0.7453     |

*Baseline = Last-observed persistence model*

🖼️ Visualization Suite (`analysis_suite.py`)

| Figure | Purpose                         | File                        |
|--------|---------------------------------|-----------------------------|
| 1      | Heaviest flash flood event      | `fig1_flash_flood.png`       |
| 2      | Predicted vs True scatter       | `fig2_scatter.png`          |
| 3      | Error distribution              | `fig3_error_dist.png`       |
| 4      | Error vs Intensity              | `fig4_error_vs_intensity.png` |
| 5      | Top 5 extreme events            | `fig5_top5_events.png`      |
| 6      | Learning rate convergence       | `fig6_lr_curves.png`        |
| 7      | Multi-scale attention heatmap    | `fig7_attention_heatmap.png`|

📅 Project Status

| Component                | Status           | File                        |
|--------------------------|------------------|-----------------------------|
| Data pipeline            | Complete         | `data_loader.py`            |
| Model architecture        | Complete         | `transformer_model.py`      |
| Training & evaluation     | Complete         | `evaluate.py`               |
| Visualization suite      | 7 figures generated | `analysis_suite.py`         |
| Best model               | Saved            | `best_model.pth`            |

🚀 Future Work — ALL 100% COMPLETED

| Week | Task                                           | Status     |
|------|------------------------------------------------|------------|
| 8    | Add quantile loss for uncertainty estimation| Completed  |
| 9    | Radar fusion via cross-attention                | Completed  |
| 10   | FastAPI real-time endpoint                     | Completed  |
| 11   | Full ablation study                             | Completed  |
| 12   | Final report + live demo                        | Completed  |


📚 References

- Zeng, A. et al. (2022). *Are Transformers Effective for Time Series Forecasting?* arXiv:2205.13504.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Chamatidis, I. et al. (2023). *Short-term forecasting of rainfall using deep LSTM networks.* Atmosphere.
- Shi, X. et al. (2015). *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.* NeurIPS 2015.
- Ravuri, S. et al. (2021). *Skilful Precipitation Nowcasting using Deep Generative Models of Radar.* Nature, 597(7878), 672–677.
- Ayzel, G. et al. (2020). *RainNet: A convolutional neural network for precipitation nowcasting using radar data.* Journal of Hydrology.
- Open-Meteo Historical Weather API. https://open-meteo.com
- Sønderby, C. K. et al. (2020). *MetNet: A Neural Weather Model for Precipitation Forecasting.* Google Research.
- Bi, K. et al. (2023). *Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks.* Nature, 619(7970), 533–538.

🗂️ Repository Structure

```bash
├── data_loader.py     # Data collection & preprocessing
├── transformer_model.py # Domain-aware Transformer
├── train.py          # Model training loop
├── evaluate.py       # Evaluation metrics & logging
├── analysis_suite.py # Visualization utilities
├── best_model.pth    # Trained model weights
├── requirements.txt  # Environment dependencies
└── README.md         # Project documentation
```
⚡ Environment Setup
Option 1 – Local Setup
```bash
git clone https://github.com/CastleBby/612_FinalProject.git
cd 612_FinalProject
pip install -r requirements.txt
python train.py
```
Option 2 – Run on Google Colab
```python
!git clone https://github.com/CastleBby/612_FinalProject.git
%cd 612_FinalProject
!pip install -r requirements.txt
!python evaluate.py
```

🧾 requirements.txt
```text
torch>=2.3.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
tqdm>=4.66.0
openmeteo-requests>=0.2.1
requests>=2.32.0
fastapi>=0.111.0
uvicorn>=0.30.0
```
🧑‍💻 Contributors
Group 12 — Deep Learning (Fall 2025)

Instructor: Prof. Samet Ayhan, University of Maryland

locations before creating the environment.
- **PyTorch `iJIT_NotifyEvent` errors**: Prefer the official CPU wheel (`pip install ... torch==2.5.1`) instead of the Conda package to avoid missing Intel ITT instrumentation libraries on non-Intel hardware.
- **API limits**: The Open-Meteo archive is free but rate-limited. Re-running `data_loader.py` in quick succession may trigger temporary HTTP errors; retry after a short pause if that happens.

With these steps, you can fully reproduce the dataset preparation, transformer training, and evaluation workflow for the DATA612 final project.
