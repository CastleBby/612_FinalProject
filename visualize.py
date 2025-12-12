# visualize.py
import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml
import os
from transformer_model import get_model


# ------------------------------------------------------------------
# 1. Load config & processed test data
# ------------------------------------------------------------------
config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
cfg = yaml.safe_load(open(config_path))
data = np.load('processed_data.npz', allow_pickle=True)

X_test   = data['X_test']      # (N, 24, 5)  scaled
y_test   = data['y_test']      # (N,)       scaled
loc_test = data['loc_test']    # (N,)       int
scaler   = data['scaler'].item()   # MinMaxScaler

precip_idx = cfg['data']['variables'].index('precipitation')


# ------------------------------------------------------------------
# 2. Un-scale a *single* precipitation value (scalar → mm)
# ------------------------------------------------------------------
def unscale_scalar(scaled_val, scaler, precip_idx):
    dummy = np.zeros(5)                     # 5 features
    dummy[precip_idx] = scaled_val
    return scaler.inverse_transform([dummy])[0, precip_idx]


# ------------------------------------------------------------------
# 3. Un-scale an *entire feature row* (used for past 24 h)
# ------------------------------------------------------------------
def unscale_row(row, scaler, precip_idx):
    """row: (5,) scaled → returns precipitation in mm"""
    return unscale_scalar(row[precip_idx], scaler, precip_idx)


# ------------------------------------------------------------------
# 4. Find the heaviest rain hour in the test set
# ------------------------------------------------------------------
idx = np.argmax(y_test)                     # index of max precip
x_sample = torch.FloatTensor(X_test[idx:idx+1])
loc_sample = torch.LongTensor([loc_test[idx]])


# ------------------------------------------------------------------
# 5. Load the trained model
# ------------------------------------------------------------------
model = get_model(
    'transformer',
    input_dim=X_test.shape[-1],
    seq_len=cfg['data']['seq_len'],
    d_model=cfg['model']['d_model'],
    nhead=cfg['model']['nhead'],
    num_encoder_layers=cfg['model'].get('num_encoder_layers', cfg['model']['num_layers']),
    num_decoder_layers=cfg['model'].get('num_decoder_layers', 2),
    embed_dim=cfg['model']['embed_dim'],
    dropout=cfg['model']['dropout'],
    num_locations=len(cfg['data']['locations']),
    feature_groups=cfg['model']['feature_groups'],
    use_series_decomposition=cfg['model'].get('use_series_decomposition', False)
)
checkpoint = torch.load('best_model.pth', map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    model.load_state_dict(checkpoint)
model.eval()


# ------------------------------------------------------------------
# 6. Predict (scaled → mm)
# ------------------------------------------------------------------
with torch.no_grad():
    pred_scaled = model(x_sample, loc_sample).item()

pred_mm = unscale_scalar(pred_scaled, scaler, precip_idx)
true_mm = unscale_scalar(y_test[idx], scaler, precip_idx)


# ------------------------------------------------------------------
# 7. Un-scale the past 24 hours (only precipitation)
# ------------------------------------------------------------------
past_scaled = X_test[idx]                     # (24, 5)
past_mm = [unscale_row(row, scaler, precip_idx) for row in past_scaled]


# ------------------------------------------------------------------
# 8. Plot
# ------------------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(range(-23, 1), past_mm,
         label='Past 24 h (mm)', marker='o', color='steelblue')
plt.axhline(pred_mm, color='red',   linestyle='--', linewidth=2,
            label=f'Predicted: {pred_mm:.2f} mm')
plt.axhline(true_mm, color='green', linestyle='--', linewidth=2,
            label=f'True: {true_mm:.2f} mm')
plt.title('1-Hour Flash-Flood Nowcasting (Heaviest Test Event)')
plt.xlabel('Hours Before Forecast')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
