# data_loader.py
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def fetch_openmeteo_data(lat, lon, start_date, end_date, variables):
    """Fetch hourly data from Open-Meteo API for a single location."""
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(variables),
        'timezone': 'America/New_York'
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"API error for lat={lat}, lon={lon}: {response.text}")
    data = response.json()['hourly']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['location'] = f"{lat:.2f}_{lon:.2f}"
    return df.set_index('time')


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 1. Load config & fetch raw data
    # ------------------------------------------------------------------
    cfg = load_config()
    all_raw_dfs = []
    print("Fetching data for MD locations...")
    for loc in tqdm(cfg['data']['locations']):
        df = fetch_openmeteo_data(
            loc['lat'], loc['lon'],
            cfg['data']['start_date'],
            cfg['data']['end_date'],
            cfg['data']['variables']
        )
        df = df.interpolate(method='linear').dropna()
        all_raw_dfs.append(df)

    # Save the concatenated raw CSV (optional, for inspection)
    full_raw = pd.concat(all_raw_dfs).sort_index()
    full_raw.to_csv('md_weather_data.csv')
    print(f"Raw data saved: {len(full_raw)} rows across {len(cfg['data']['locations'])} locations.")

    # ------------------------------------------------------------------
    # 2. Build per-location sequences + location IDs
    # ------------------------------------------------------------------
    variables = cfg['data']['variables']
    seq_len   = cfg['data']['seq_len']
    horizon   = cfg['data']['pred_horizon']
    precip_idx = variables.index('precipitation')

    # map (lat_lon string) → integer id
    loc_map = {f"{loc['lat']:.2f}_{loc['lon']:.2f}": i
               for i, loc in enumerate(cfg['data']['locations'])}

    all_X, all_y, all_loc = [], [], []
    scaler = None                     # one global scaler

    for df in all_raw_dfs:
        loc_key = df['location'].iloc[0]
        loc_id  = loc_map[loc_key]

        # keep only the numeric columns we need
        data = df[variables].values

        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(data)          # fit on the first station (distributions are similar)

        data_scaled = scaler.transform(data)

        X_loc, y_loc = [], []
        for i in range(len(data_scaled) - seq_len - horizon + 1):
            X_loc.append(data_scaled[i:i + seq_len])
            # target = precipitation at t+1 (or next horizon steps)
            y_loc.append(data_scaled[i + seq_len:
                                    i + seq_len + horizon,
                                    precip_idx])

        X_loc = np.array(X_loc)
        y_loc = np.array(y_loc)
        if horizon == 1:
            y_loc = y_loc.squeeze(-1)   # (samples,)

        all_X.append(X_loc)
        all_y.append(y_loc)
        all_loc.append(np.full(len(X_loc), loc_id, dtype=np.int32))

    X       = np.concatenate(all_X, axis=0)
    y       = np.concatenate(all_y, axis=0)
    loc_idx = np.concatenate(all_loc, axis=0)

    # ------------------------------------------------------------------
    # 3. Train / Val / Test split (temporal → shuffle=False)
    # ------------------------------------------------------------------
    X_temp, X_test, y_temp, y_test, loc_temp, loc_test = train_test_split(
        X, y, loc_idx,
        test_size=cfg['data']['test_split'],
        random_state=42,
        shuffle=False
    )

    val_ratio = cfg['data']['val_split'] / (1.0 - cfg['data']['test_split'])
    X_train, X_val, y_train, y_val, loc_train, loc_val = train_test_split(
        X_temp, y_temp, loc_temp,
        test_size=val_ratio,
        random_state=42,
        shuffle=False
    )

    # ------------------------------------------------------------------
    # 4. Save everything
    # ------------------------------------------------------------------
    np.savez('processed_data.npz',
             X_train=X_train, y_train=y_train,
             X_val=X_val,   y_val=y_val,
             X_test=X_test, y_test=y_test,
             loc_train=loc_train, loc_val=loc_val, loc_test=loc_test,
             scaler=scaler)

    print(f"Sequences ready → Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
