#!/usr/bin/env python3
"""
Quick data checker and preparer.
Ensures all required data files exist before training.

Usage:
    python check_and_prepare_data.py [--force]
"""

import os
import sys
import subprocess
import argparse


def check_file(filepath, min_size_mb=0):
    """Check if file exists and meets minimum size."""
    if not os.path.exists(filepath):
        return False, f"Missing: {filepath}"
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb < min_size_mb:
        return False, f"Too small: {filepath} ({size_mb:.1f}MB < {min_size_mb}MB)"
    
    return True, f"OK: {filepath} ({size_mb:.1f}MB)"


def main():
    parser = argparse.ArgumentParser(description='Check and prepare data for training')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    args = parser.parse_args()
    
    print("="*80)
    print("Data Preparation Check")
    print("="*80)
    
    # Required files
    raw_data = "nowcasting_gpt_data/raw_weather_data.csv"
    train_data = "nowcasting_gpt_data/formatted_data_train.csv"
    val_data = "nowcasting_gpt_data/formatted_data_val.csv"
    test_data = "nowcasting_gpt_data/formatted_data_test.csv"
    
    all_exist = True
    
    # Check raw data
    print("\n1. Checking raw data...")
    exists, msg = check_file(raw_data, min_size_mb=10)
    print(f"   {msg}")
    if not exists or args.force:
        all_exist = False
        print("\n   → Downloading raw data from Open-Meteo API...")
        try:
            subprocess.run(['python', 'nowcasting_gpt_data_downloader.py'], check=True)
            print("   ✓ Download completed")
        except subprocess.CalledProcessError:
            print("   ✗ Download failed!")
            sys.exit(1)
    
    # Check formatted data
    print("\n2. Checking formatted data...")
    formatted_exist = True
    for filepath in [train_data, val_data, test_data]:
        exists, msg = check_file(filepath, min_size_mb=1)
        print(f"   {msg}")
        if not exists:
            formatted_exist = False
    
    if not formatted_exist or args.force:
        print("\n   → Formatting data into train/val/test splits...")
        try:
            subprocess.run(['python', 'nowcasting_gpt_data_formatter.py'], check=True)
            print("   ✓ Formatting completed")
            
            # Verify the expected files were created
            for filepath in [train_data, val_data, test_data]:
                if not os.path.exists(filepath):
                    print(f"   ✗ Expected file not created: {filepath}")
                    sys.exit(1)
        except subprocess.CalledProcessError:
            print("   ✗ Formatting failed!")
            sys.exit(1)
    
    # Final check
    print("\n" + "="*80)
    print("Data Preparation Summary")
    print("="*80)
    
    all_ready = True
    for filepath in [raw_data, train_data, val_data, test_data]:
        exists, msg = check_file(filepath)
        status = "✓" if exists else "✗"
        print(f"{status} {msg}")
        if not exists:
            all_ready = False
    
    print("="*80)
    
    if all_ready:
        print("\n✅ All data files are ready for training!")
        print("\nYou can now run:")
        print("  python train_v5.py")
        print("  or")
        print("  sbatch run_v5_robust.sbatch")
        return 0
    else:
        print("\n❌ Some data files are missing or invalid")
        print("\nTry running with --force to re-download:")
        print("  python check_and_prepare_data.py --force")
        return 1


if __name__ == '__main__':
    sys.exit(main())

