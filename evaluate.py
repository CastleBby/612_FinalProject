# evaluate.py
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml
from transformer_model import get_model


# --------------------------------------------------------------
# 1. Helper: load config
# --------------------------------------------------------------
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# 2. Un-scale precipitation (MinMax → real mm)
# --------------------------------------------------------------
def unscale(y_scaled, scaler, precip_idx):
    """y_scaled: 1-D array of scaled precipitation values."""
    dummy = np.zeros((len(y_scaled), scaler.scale_.size))
    dummy[:, precip_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, precip_idx]


# --------------------------------------------------------------
# 3. Classification metrics (rain / extreme)
# --------------------------------------------------------------
def classification_metrics(y_true, y_pred, threshold):
    yb_t = (y_true > threshold).astype(int)
    yb_p = (y_pred > threshold).astype(int)
    tp = np.sum(yb_t & yb_p)
    fp = np.sum(~yb_t & yb_p)
    fn = np.sum(yb_t & ~yb_p)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return pod, far, csi


# --------------------------------------------------------------
# 4. Main evaluation
# --------------------------------------------------------------
if __name__ == '__main__':
    cfg = load_config()

    # ----- device (CUDA, MPS, or CPU) -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----- load processed test data -----
    data = np.load('processed_data.npz', allow_pickle=True)
    X_test   = data['X_test']
    y_test   = data['y_test']
    loc_test = data['loc_test']
    scaler   = data['scaler'].item()                     # MinMaxScaler object

    precip_idx = cfg['data']['variables'].index('precipitation')

    # ----- build & load model -----
    model = get_model(
        'transformer',
        input_dim=X_test.shape[-1],
        seq_len=cfg['data']['seq_len'],
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=cfg['model']['num_layers'],
        embed_dim=cfg['model']['embed_dim'],
        dropout=cfg['model']['dropout'],
        num_locations=len(cfg['data']['locations']),
        feature_groups=cfg['model']['feature_groups']
    ).to(device)

    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ----- inference in batches to avoid OOM -----
    batch_size = cfg['model'].get('batch_size', 64)  # Use training batch size or default
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            batch_loc = torch.LongTensor(loc_test[i:i+batch_size]).to(device)
            batch_pred = model(batch_X, batch_loc).cpu().numpy()
            predictions.append(batch_pred)

    pred_scaled = np.concatenate(predictions, axis=0).flatten()

    # ----- un-scale -----
    y_pred = unscale(pred_scaled, scaler, precip_idx)
    y_true = unscale(y_test, scaler, precip_idx)

    # ----- apply prediction threshold (reduce false alarms) -----
    pred_threshold = cfg['eval'].get('prediction_threshold', 0.0)
    if pred_threshold > 0:
        y_pred = np.where(y_pred < pred_threshold, 0.0, y_pred)
        print(f"Applied prediction threshold: {pred_threshold} mm")

    # ----- regression metrics -----
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # ----- rain-event metrics (threshold = 0.1 mm) -----
    pod, far, csi = classification_metrics(y_true, y_pred, cfg['eval']['rain_threshold'])

    # ----- extreme-event POD (90th percentile) -----
    extreme_thresh = np.percentile(y_true, cfg['eval']['extreme_threshold'] * 100)
    pod_ext, _, _ = classification_metrics(y_true, y_pred, extreme_thresh)

    # ----- print results -----
    print("\n=== TRANSFORMER RESULTS (mm/h) ===")
    print(f"RMSE : {rmse:6.4f}")
    print(f"MAE  : {mae:6.4f}")
    print(f"CSI  : {csi:6.4f} | POD: {pod:6.4f} | FAR: {far:6.4f}")
    print(f"Extreme POD ({cfg['eval']['extreme_threshold']*100:.0f}th %): {pod_ext:6.4f}")

    # ----- persistence baseline (last observed = forecast) -----
    persist_scaled = X_test[:, -1, precip_idx]
    persist_pred   = unscale(persist_scaled, scaler, precip_idx)

    rmse_p = np.sqrt(mean_squared_error(y_true, persist_pred))
    mae_p  = mean_absolute_error(y_true, persist_pred)
    pod_p, far_p, csi_p = classification_metrics(y_true, persist_pred, cfg['eval']['rain_threshold'])
    pod_ext_p, _, _ = classification_metrics(y_true, persist_pred, extreme_thresh)

    print("\n=== PERSISTENCE BASELINE ===")
    print(f"RMSE : {rmse_p:6.4f}")
    print(f"MAE  : {mae_p:6.4f}")
    print(f"CSI  : {csi_p:6.4f} | POD: {pod_p:6.4f} | FAR: {far_p:6.4f}")
    print(f"Extreme POD: {pod_ext_p:6.4f}")
