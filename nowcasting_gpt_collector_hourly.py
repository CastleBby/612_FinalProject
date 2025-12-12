"""
NowcastingGPT Collector Adapted for Hourly Weather Sequences

Original NowcastingGPT uses radar images (3D tensors).
This version adapts it to use hourly weather sequences (1D time series).

Key changes:
- Instead of loading radar images, load precipitation time series
- Create sequences of length (obs_time + pred_time) hours
- Format: [obs_time hours of past data] → [pred_time hours to predict]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path


class HourlyWeatherDataset(Dataset):
    """
    Dataset for hourly weather sequences.
    
    Instead of loading images like original NowcastingGPT,
    we load time series windows from CSV files.
    
    Args:
        csv_path: Path to CSV file (format: timestamp,precipitation)
        obs_time: Number of hours to observe (input sequence length)
        pred_time: Number of hours to predict (output sequence length)
        transform: Optional transform (not used for now)
    """
    def __init__(self, csv_path, obs_time=24, pred_time=1, transform=None):
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.transform = transform
        self.sequence_length = obs_time + pred_time
        
        # Load CSV data
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path, header=None, names=['timestamp', 'precipitation'])
        
        # Sort by timestamp to ensure temporal order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract precipitation values
        self.precipitation = df['precipitation'].values
        self.timestamps = df['timestamp'].values
        
        # Calculate number of valid sequences
        # We need at least sequence_length consecutive hours
        self.num_sequences = len(self.precipitation) - self.sequence_length + 1
        
        if self.num_sequences <= 0:
            raise ValueError(f"Not enough data! Need at least {self.sequence_length} hours, got {len(self.precipitation)}")
        
        print(f"  Loaded {len(self.precipitation)} hours")
        print(f"  Created {self.num_sequences} sequences (obs={obs_time}h, pred={pred_time}h)")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get a sequence window.
        
        Returns:
            Tensor of shape (sequence_length,) containing:
            - First obs_time values: input (past observations)
            - Last pred_time values: target (future to predict)
        """
        # Extract sequence window
        sequence = self.precipitation[idx:idx + self.sequence_length]
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        # Optionally apply transform (normalization, etc.)
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor


class HourlyWeatherCollector:
    """
    Data collector adapted for hourly weather sequences.
    
    Compatible with NowcastingGPT training pipeline but uses
    hourly time series instead of radar images.
    """
    
    def __init__(self):
        self.training_data = []
        self.testing_data = []
        self.validation_data = []
        self.testing_data_ext = []  # For extreme events (optional)
    
    def collect_training_data(self, csv_paths, obs_time, pred_time, batch_size, shuffle=True):
        """
        Collect training data from multiple CSV files.
        
        Args:
            csv_paths: List of CSV file paths or single path
            obs_time: Number of hours to observe
            pred_time: Number of hours to predict
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
        
        Returns:
            DataLoader, total_length
        """
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        # Create dataset for each CSV
        datasets = []
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                try:
                    dataset = HourlyWeatherDataset(csv_path, obs_time, pred_time)
                    datasets.append(dataset)
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
        
        if not datasets:
            raise ValueError("No valid datasets loaded!")
        
        # Concatenate datasets
        if len(datasets) > 1:
            combined_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"\nCombined {len(datasets)} datasets into training set")
        else:
            combined_dataset = datasets[0]
        
        # Create DataLoader
        self.training_data = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        length = len(combined_dataset)
        print(f"Training DataLoader created: {length} sequences, batch_size={batch_size}")
        
        return self.training_data, length
    
    def collect_testing_data(self, csv_paths, obs_time, pred_time, batch_size, shuffle=False):
        """Collect testing data."""
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        datasets = []
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                try:
                    dataset = HourlyWeatherDataset(csv_path, obs_time, pred_time)
                    datasets.append(dataset)
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
        
        if not datasets:
            raise ValueError("No valid datasets loaded!")
        
        if len(datasets) > 1:
            combined_dataset = torch.utils.data.ConcatDataset(datasets)
        else:
            combined_dataset = datasets[0]
        
        self.testing_data = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        length = len(combined_dataset)
        print(f"Testing DataLoader created: {length} sequences")
        
        return self.testing_data, length
    
    def collect_validation_data(self, csv_paths, obs_time, pred_time, batch_size, shuffle=False):
        """Collect validation data."""
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        datasets = []
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                try:
                    dataset = HourlyWeatherDataset(csv_path, obs_time, pred_time)
                    datasets.append(dataset)
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
        
        if not datasets:
            raise ValueError("No valid datasets loaded!")
        
        if len(datasets) > 1:
            combined_dataset = torch.utils.data.ConcatDataset(datasets)
        else:
            combined_dataset = datasets[0]
        
        self.validation_data = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        length = len(combined_dataset)
        print(f"Validation DataLoader created: {length} sequences")
        
        return self.validation_data, length


if __name__ == '__main__':
    """Test the hourly weather collector"""
    print("="*80)
    print("Testing HourlyWeatherCollector")
    print("="*80)
    
    # Test with a sample CSV
    test_csv = "nowcasting_gpt_data/formatted_csv/training_AllLocations.csv"
    
    if Path(test_csv).exists():
        print(f"\nTesting with: {test_csv}")
        
        # Create collector
        collector = HourlyWeatherCollector()
        
        # Collect training data
        train_loader, train_len = collector.collect_training_data(
            csv_paths=[test_csv],
            obs_time=24,  # 24 hours of past data
            pred_time=1,   # Predict next 1 hour
            batch_size=32,
            shuffle=True
        )
        
        # Test batch loading
        print("\nTesting batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1}: shape={batch.shape}, min={batch.min():.4f}, max={batch.max():.4f}")
            if i >= 2:  # Just show first 3 batches
                break
        
        print("\n✓ HourlyWeatherCollector test passed!")
    else:
        print(f"\n⚠️  Test file not found: {test_csv}")
        print("Run nowcasting_gpt_data_formatter.py first!")

