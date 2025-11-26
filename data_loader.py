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


def add_temporal_features(df):
    """Add temporal features for better pattern recognition."""
    # Cyclical encoding for hour (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    # Cyclical encoding for day of year (1-365)
    day_of_year = df.index.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    # Month as cyclical feature
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    return df


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 1. Load config & fetch raw data
    # ------------------------------------------------------------------
    import time
    cfg = load_config()
    all_raw_dfs = []
    print("Fetching data for MD locations...")
    for i, loc in enumerate(tqdm(cfg['data']['locations'])):
        if i > 0:
            # Add delay to avoid API rate limit
            time.sleep(70)  # Wait 70 seconds between requests
        df = fetch_openmeteo_data(
            loc['lat'], loc['lon'],
            cfg['data']['start_date'],
            cfg['data']['end_date'],
            cfg['data']['variables']
        )
        # Interpolate missing values
        df = df.interpolate(method='linear').bfill().ffill()

        # Add temporal features
        df = add_temporal_features(df)

        # Drop any remaining NaN
        df = df.dropna()

        all_raw_dfs.append(df)

    # Save the concatenated raw CSV (optional, for inspection)
    full_raw = pd.concat(all_raw_dfs).sort_index()
    full_raw.to_csv('md_weather_data.csv')
    print(f"Raw data saved: {len(full_raw)} rows across {len(cfg['data']['locations'])} locations.")

    # Print data quality statistics
    print("\n=== Data Quality Report ===")
    for var in cfg['data']['variables']:
        print(f"{var:25s} - Mean: {full_raw[var].mean():8.2f}, Std: {full_raw[var].std():8.2f}, "
              f"Min: {full_raw[var].min():8.2f}, Max: {full_raw[var].max():8.2f}")
    print(f"Precipitation > 0.1mm:    {(full_raw['precipitation'] > 0.1).sum()} hours ({(full_raw['precipitation'] > 0.1).mean()*100:.1f}%)")
    print(f"Extreme events (>90th):   {(full_raw['precipitation'] > full_raw['precipitation'].quantile(0.9)).sum()} hours")

    # ------------------------------------------------------------------
    # 2. Build per-location sequences + location IDs
    # ------------------------------------------------------------------
    variables = cfg['data']['variables']
    temporal_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    all_features = variables + temporal_features

    seq_len   = cfg['data']['seq_len']
    horizon   = cfg['data']['pred_horizon']
    precip_idx = variables.index('precipitation')

    # map (lat_lon string) → integer id
    loc_map = {f"{loc['lat']:.2f}_{loc['lon']:.2f}": i
               for i, loc in enumerate(cfg['data']['locations'])}

    # Collect all data first to fit scaler properly
    print("\n=== Building sequences with temporal features ===")
    all_data_for_scaler = []
    for df in all_raw_dfs:
        all_data_for_scaler.append(df[all_features].values)

    # Fit scaler on ALL data (not just first station!)
    all_data_combined = np.vstack(all_data_for_scaler)
    scaler = MinMaxScaler()
    scaler.fit(all_data_combined)
    print(f"Scaler fitted on {len(all_data_combined):,} total samples")

    # Now build sequences per location
    all_X, all_y, all_loc = [], [], []

    for df in all_raw_dfs:
        loc_key = df['location'].iloc[0]
        loc_id  = loc_map[loc_key]

        # Get all features (weather + temporal)
        data = df[all_features].values
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

    print(f"Total sequences: {len(X):,}")
    print(f"Input shape: {X.shape} (includes {len(temporal_features)} temporal features)")

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

    print(f"Sequences ready -> Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
