"""
NowcastingGPT Data Formatter
Converts raw weather data to NowcastingGPT CSV format: timestamp,precipitation

Format: YYYYMMDDHHmm,precipitation_value
Example: 201306201200,28.53037296037299

Usage:
    python nowcasting_gpt_data_formatter.py
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime

# Reproducibility
RANDOM_SEED = 202511
np.random.seed(RANDOM_SEED)


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(data_dir='nowcasting_gpt_data'):
    """Load raw weather data"""
    pkl_path = f'{data_dir}/raw_weather_data.pkl'
    if Path(pkl_path).exists():
        print(f"Loading raw data from: {pkl_path}")
        df = pd.read_pickle(pkl_path)
    else:
        csv_path = f'{data_dir}/raw_weather_data.csv'
        print(f"Loading raw data from: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['time'])
    
    print(f"Loaded {len(df):,} records")
    return df


def format_timestamp(dt):
    """
    Format datetime to NowcastingGPT format: YYYYMMDDHHmm
    
    Args:
        dt: pandas datetime
    
    Returns:
        String timestamp like '201306201200'
    """
    return dt.strftime('%Y%m%d%H%M')


def create_nowcasting_format_csv(df, location_name, output_path):
    """
    Create NowcastingGPT format CSV for a single location.
    
    Format: timestamp,precipitation_value
    
    Args:
        df: DataFrame for single location
        location_name: Name of location
        output_path: Path to save CSV
    """
    # Sort by time
    df = df.sort_values('time').copy()
    
    # Format timestamp
    df['timestamp'] = df['time'].apply(format_timestamp)
    
    # Create output DataFrame with only timestamp and precipitation
    output_df = df[['timestamp', 'precipitation']].copy()
    
    # Save without headers (NowcastingGPT format)
    output_df.to_csv(output_path, index=False, header=False)
    
    print(f"  ✓ {location_name}: {len(output_df):,} records → {output_path}")
    
    # Print sample
    print(f"    Sample (first 3 rows):")
    for _, row in output_df.head(3).iterrows():
        print(f"      {row['timestamp']},{row['precipitation']}")
    
    return output_df


def split_train_val_test(df, train_split=0.7, val_split=0.15, test_split=0.15, seed=RANDOM_SEED):
    """
    Split data into train/val/test sets temporally.
    
    Args:
        df: DataFrame with time column
        train_split: Fraction for training (default 0.7)
        val_split: Fraction for validation (default 0.15)
        test_split: Fraction for test (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate split indices (temporal split, not random)
    n = len(df)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\nTemporal split:")
    print(f"  Training:   {len(train_df):,} records ({train_df['time'].min()} to {train_df['time'].max()})")
    print(f"  Validation: {len(val_df):,} records ({val_df['time'].min()} to {val_df['time'].max()})")
    print(f"  Test:       {len(test_df):,} records ({test_df['time'].min()} to {test_df['time'].max()})")
    
    return train_df, val_df, test_df


def format_all_locations(cfg, data_dir='nowcasting_gpt_data'):
    """
    Format data for all locations and create train/val/test splits.
    
    Args:
        cfg: Configuration dictionary
        data_dir: Directory containing raw data
    """
    # Load raw data
    df = load_raw_data(data_dir)
    
    # Create output directory
    output_dir = f'{data_dir}/formatted_csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("FORMATTING DATA TO NOWCASTINGGPT FORMAT")
    print("="*80)
    
    # Get split ratios from config or use defaults
    train_split = 1.0 - cfg['data']['val_split'] - cfg['data']['test_split']
    val_split = cfg['data']['val_split']
    test_split = cfg['data']['test_split']
    
    print(f"Split ratios: Train={train_split:.1%}, Val={val_split:.1%}, Test={test_split:.1%}")
    print()
    
    # Process each location
    all_train_dfs = []
    all_val_dfs = []
    all_test_dfs = []
    
    for location in cfg['data']['locations']:
        location_name = location['name']
        print(f"Processing {location_name}...")
        
        # Filter data for this location
        loc_df = df[df['location'] == location_name].copy()
        
        if len(loc_df) == 0:
            print(f"  ⚠️  No data for {location_name}, skipping")
            continue
        
        # Split data temporally
        train_df, val_df, test_df = split_train_val_test(
            loc_df, train_split, val_split, test_split, seed=RANDOM_SEED
        )
        
        # Create formatted CSVs
        train_path = f'{output_dir}/training_{location_name.replace(" ", "")}.csv'
        val_path = f'{output_dir}/validation_{location_name.replace(" ", "")}.csv'
        test_path = f'{output_dir}/testing_{location_name.replace(" ", "")}.csv'
        
        train_formatted = create_nowcasting_format_csv(train_df, f"{location_name} (train)", train_path)
        val_formatted = create_nowcasting_format_csv(val_df, f"{location_name} (val)", val_path)
        test_formatted = create_nowcasting_format_csv(test_df, f"{location_name} (test)", test_path)
        
        all_train_dfs.append(train_formatted)
        all_val_dfs.append(val_formatted)
        all_test_dfs.append(test_formatted)
        
        print()
    
    # Create combined CSVs (all locations)
    print("Creating combined CSVs (all locations)...")
    
    combined_train = pd.concat(all_train_dfs, ignore_index=True)
    combined_val = pd.concat(all_val_dfs, ignore_index=True)
    combined_test = pd.concat(all_test_dfs, ignore_index=True)
    
    # Sort by timestamp
    combined_train = combined_train.sort_values('timestamp')
    combined_val = combined_val.sort_values('timestamp')
    combined_test = combined_test.sort_values('timestamp')
    
    # Save combined CSVs (both formats for compatibility)
    combined_train.to_csv(f'{output_dir}/training_AllLocations.csv', index=False, header=False)
    combined_val.to_csv(f'{output_dir}/validation_AllLocations.csv', index=False, header=False)
    combined_test.to_csv(f'{output_dir}/testing_AllLocations.csv', index=False, header=False)
    
    # Also save in the expected format for V5 pipeline (directly in nowcasting_gpt_data/)
    data_dir = 'nowcasting_gpt_data'
    combined_train.to_csv(f'{data_dir}/formatted_data_train.csv', index=False, header=False)
    combined_val.to_csv(f'{data_dir}/formatted_data_val.csv', index=False, header=False)
    combined_test.to_csv(f'{data_dir}/formatted_data_test.csv', index=False, header=False)
    
    print(f"  ✓ Combined training: {len(combined_train):,} records")
    print(f"  ✓ Combined validation: {len(combined_val):,} records")
    print(f"  ✓ Combined test: {len(combined_test):,} records")
    print(f"  ✓ V5 pipeline files: formatted_data_train/val/test.csv")
    
    # Save metadata
    metadata = {
        'format_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'format': 'NowcastingGPT (timestamp,precipitation)',
        'timestamp_format': 'YYYYMMDDHHmm',
        'num_locations': len(cfg['data']['locations']),
        'locations': [loc['name'] for loc in cfg['data']['locations']],
        'split_ratios': {
            'train': train_split,
            'validation': val_split,
            'test': test_split
        },
        'split_method': 'temporal (chronological)',
        'train_records': len(combined_train),
        'val_records': len(combined_val),
        'test_records': len(combined_test),
        'random_seed': RANDOM_SEED,
        'precipitation_stats': {
            'train': {
                'mean': float(combined_train['precipitation'].mean()),
                'std': float(combined_train['precipitation'].std()),
                'min': float(combined_train['precipitation'].min()),
                'max': float(combined_train['precipitation'].max()),
                'rain_percentage': float((combined_train['precipitation'] > 0.1).mean() * 100)
            },
            'test': {
                'mean': float(combined_test['precipitation'].mean()),
                'std': float(combined_test['precipitation'].std()),
                'min': float(combined_test['precipitation'].min()),
                'max': float(combined_test['precipitation'].max()),
                'rain_percentage': float((combined_test['precipitation'] > 0.1).mean() * 100)
            }
        }
    }
    
    with open(f'{output_dir}/format_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to: {output_dir}/format_metadata.json")
    
    # Print summary
    print("\n" + "="*80)
    print("FORMATTING COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - training_{{location}}.csv (per location)")
    print(f"  - validation_{{location}}.csv (per location)")
    print(f"  - testing_{{location}}.csv (per location)")
    print(f"  - training_AllLocations.csv (combined)")
    print(f"  - validation_AllLocations.csv (combined)")
    print(f"  - testing_AllLocations.csv (combined)")
    print(f"\nPrecipitation statistics (training set):")
    print(f"  Mean: {metadata['precipitation_stats']['train']['mean']:.4f} mm/h")
    print(f"  Std:  {metadata['precipitation_stats']['train']['std']:.4f} mm/h")
    print(f"  Max:  {metadata['precipitation_stats']['train']['max']:.4f} mm/h")
    print(f"  Rain events (>0.1mm): {metadata['precipitation_stats']['train']['rain_percentage']:.2f}%")
    print("="*80)


if __name__ == '__main__':
    print("="*80)
    print("NowcastingGPT Data Formatter")
    print("="*80)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load configuration
    cfg = load_config()
    
    # Format data
    format_all_locations(cfg)
    
    print("\n✅ Data formatting complete!")
    print("\nFiles created:")
    print("  1. Detailed files: nowcasting_gpt_data/formatted_csv/")
    print("  2. V5 pipeline files: nowcasting_gpt_data/formatted_data_*.csv")
    print("\nNext steps:")
    print("  - Train baseline: sbatch run_nowcasting_baseline.sbatch")
    print("  - Train V5: sbatch run_v5.sbatch")

