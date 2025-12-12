"""
NowcastingGPT Data Downloader for Maryland Weather Data
Downloads historical weather data from Open-Meteo API and saves as raw data.

Usage:
    python nowcasting_gpt_data_downloader.py
"""

import requests
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

# Reproducibility
RANDOM_SEED = 202511
np.random.seed(RANDOM_SEED)


def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def download_location_data_chunked(lat, lon, start_date, end_date, location_name, chunk_years=5, max_retries=3):
    """
    Download weather data for a single location in chunks (to avoid API timeouts).
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        location_name: Name of location
        chunk_years: Download in chunks of N years
        max_retries: Maximum retry attempts per chunk
    
    Returns:
        DataFrame with weather data
    """
    from datetime import datetime, timedelta
    
    print(f"\nDownloading data for {location_name} ({lat}, {lon})...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Strategy: Chunked download ({chunk_years} years per request)")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_chunks = []
    current = start
    
    while current < end:
        # Calculate chunk end date (chunk_years later or end_date, whichever is earlier)
        chunk_end = min(current + timedelta(days=365*chunk_years), end)
        
        chunk_start_str = current.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        
        print(f"  Downloading chunk: {chunk_start_str} to {chunk_end_str}")
        
        # Try downloading this chunk with retries
        success = False
        for attempt in range(max_retries):
            try:
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": chunk_start_str,
                    "end_date": chunk_end_str,
                    "hourly": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m",
                    "timezone": "America/New_York"
                }
                
                response = requests.get(url, params=params, timeout=120)
                response.raise_for_status()
                data = response.json()
                
                # Check if data contains expected fields
                if 'hourly' not in data:
                    raise ValueError(f"No hourly data in response")
                
                hourly = data['hourly']
                chunk_df = pd.DataFrame({
                    'time': pd.to_datetime(hourly['time']),
                    'temperature_2m': hourly['temperature_2m'],
                    'relative_humidity_2m': hourly['relative_humidity_2m'],
                    'precipitation': hourly['precipitation'],
                    'pressure_msl': hourly['pressure_msl'],
                    'wind_speed_10m': hourly['wind_speed_10m'],
                    'location': location_name,
                    'latitude': lat,
                    'longitude': lon
                })
                
                all_chunks.append(chunk_df)
                print(f"    ✓ Downloaded {len(chunk_df)} hours")
                success = True
                
                # Rate limiting: wait between requests to avoid API limits
                time.sleep(3)
                break
                
            except Exception as e:
                print(f"    Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Longer wait for retry
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"    ✗ Failed to download chunk after {max_retries} attempts")
                    print(f"    ⚠️  Check API status or try again later")
                    return None
        
        if not success:
            return None
        
        # Move to next chunk
        current = chunk_end + timedelta(days=1)
    
    # Combine all chunks
    if all_chunks:
        combined_df = pd.concat(all_chunks, ignore_index=True)
        print(f"  ✓ Total downloaded for {location_name}: {len(combined_df)} hours")
        return combined_df
    else:
        print(f"  ✗ No data downloaded for {location_name}")
        return None


def download_location_data(lat, lon, start_date, end_date, location_name):
    """
    Wrapper for chunked download (for backward compatibility).
    """
    return download_location_data_chunked(lat, lon, start_date, end_date, location_name, chunk_years=5)


def download_all_locations(cfg):
    """
    Download data for all locations specified in config.
    
    Returns:
        DataFrame with all location data combined
    """
    all_data = []
    failed_locations = []
    
    print("\n" + "="*80)
    print(f"Downloading data for {len(cfg['data']['locations'])} locations")
    print("="*80)
    
    for i, location in enumerate(cfg['data']['locations'], 1):
        print(f"\n[{i}/{len(cfg['data']['locations'])}] Processing {location['name']}...")
        
        # Add delay between locations to avoid rate limiting
        if i > 1:
            print(f"  Waiting 5 seconds before next location (API rate limit)...")
            time.sleep(5)
        
        df = download_location_data(
            lat=location['lat'],
            lon=location['lon'],
            start_date=cfg['data']['start_date'],
            end_date=cfg['data']['end_date'],
            location_name=location['name']
        )
        
        if df is not None:
            all_data.append(df)
            print(f"✓ {location['name']} complete: {len(df):,} records")
        else:
            failed_locations.append(location['name'])
            print(f"✗ {location['name']} FAILED - API issue or rate limit")
            print(f"  Try running script again later, or reduce date range")
    
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"Successful: {len(all_data)}/{len(cfg['data']['locations'])} locations")
    if failed_locations:
        print(f"Failed: {', '.join(failed_locations)}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(combined_df):,}")
        print(f"Expected records: ~{25*365*24*len(cfg['data']['locations']):,} ({len(cfg['data']['locations'])} locations × 25 years × 365 days × 24 hours)")
        
        coverage = len(combined_df) / (25*365*24*len(cfg['data']['locations'])) * 100
        print(f"Data coverage: {coverage:.1f}%")
        
        if coverage < 95:
            print("\n⚠️  WARNING: Data coverage < 95%")
            print("   Some data may be missing. Check for API errors above.")
        
        return combined_df
    else:
        print("\n❌ ERROR: No data downloaded!")
        return None


def save_raw_data(df, output_dir='nowcasting_gpt_data'):
    """
    Save raw downloaded data.
    
    Args:
        df: DataFrame with weather data
        output_dir: Directory to save data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = f'{output_dir}/raw_weather_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Raw data saved to: {csv_path}")
    
    # Save as pickle for faster loading
    pkl_path = f'{output_dir}/raw_weather_data.pkl'
    df.to_pickle(pkl_path)
    print(f"✓ Raw data saved to: {pkl_path}")
    
    # Save metadata
    metadata = {
        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_records': len(df),
        'num_locations': df['location'].nunique(),
        'date_range': f"{df['time'].min()} to {df['time'].max()}",
        'variables': ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                     'pressure_msl', 'wind_speed_10m'],
        'locations': df['location'].unique().tolist(),
        'random_seed': RANDOM_SEED
    }
    
    import json
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved to: {output_dir}/metadata.json")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Locations: {df['location'].nunique()}")
    print(f"\nRecords per location:")
    print(df.groupby('location').size())
    print(f"\nPrecipitation statistics (mm/h):")
    print(df.groupby('location')['precipitation'].describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print("="*80)


def verify_data_quality(df):
    """
    Verify data quality and print warnings.
    
    Args:
        df: DataFrame with weather data
    """
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)
    
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values detected:\n{missing[missing > 0]}")
    
    # Check for duplicate timestamps per location
    duplicates = df.groupby(['location', 'time']).size()
    if (duplicates > 1).any():
        issues.append(f"Duplicate timestamps detected: {(duplicates > 1).sum()} cases")
    
    # Check for negative precipitation
    if (df['precipitation'] < 0).any():
        issues.append(f"Negative precipitation values: {(df['precipitation'] < 0).sum()} cases")
    
    # Check for unreasonable temperatures
    if (df['temperature_2m'] < -50).any() or (df['temperature_2m'] > 60).any():
        issues.append(f"Extreme temperature values detected")
    
    # Check for gaps in time series
    for location in df['location'].unique():
        loc_df = df[df['location'] == location].sort_values('time')
        time_diffs = loc_df['time'].diff()
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs > expected_diff]
        if len(gaps) > 0:
            issues.append(f"{location}: {len(gaps)} time gaps detected")
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ No data quality issues detected")
    
    print("="*80)


if __name__ == '__main__':
    print("="*80)
    print("NowcastingGPT Data Downloader - Maryland Weather Data")
    print("="*80)
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load configuration
    cfg = load_config()
    print(f"Configuration loaded from config.yaml")
    print(f"  Locations: {len(cfg['data']['locations'])}")
    print(f"  Date range: {cfg['data']['start_date']} to {cfg['data']['end_date']}")
    print(f"  Variables: {cfg['data']['variables']}")
    print()
    
    # Download data
    print("Starting data download from Open-Meteo API...")
    print("(This may take several minutes depending on date range)")
    print()
    
    df = download_all_locations(cfg)
    
    if df is not None:
        # Verify data quality
        verify_data_quality(df)
        
        # Save raw data
        save_raw_data(df)
        
        print("\n✅ Data download complete!")
        print("\nNext steps:")
        print("  1. Run: python nowcasting_gpt_data_formatter.py")
        print("  2. Then: sbatch nowcasting_gpt_pipeline.sbatch")
    else:
        print("\n❌ Data download failed!")
        exit(1)

