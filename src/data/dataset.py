"""Data pipeline for dance generation system."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split


class SyntheticDanceDataset(Dataset):
    """Synthetic dance dataset for demonstration purposes.
    
    This dataset generates random dance sequences that follow basic patterns
    like sinusoidal movements, circular motions, and random walks.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 50,
        feature_dim: int = 10,
        pattern_type: str = "mixed",
        noise_level: float = 0.1,
        normalize: bool = True
    ):
        """Initialize synthetic dance dataset.
        
        Args:
            num_samples: Number of samples to generate.
            sequence_length: Length of each dance sequence.
            feature_dim: Number of features per timestep.
            pattern_type: Type of movement pattern ('sinusoidal', 'circular', 'random', 'mixed').
            noise_level: Amount of noise to add.
            normalize: Whether to normalize the data.
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.pattern_type = pattern_type
        self.noise_level = noise_level
        self.normalize = normalize
        
        self.data = self._generate_data()
        
        if self.normalize:
            self.data = self._normalize_data()
    
    def _generate_data(self) -> np.ndarray:
        """Generate synthetic dance data."""
        data = np.zeros((self.num_samples, self.sequence_length, self.feature_dim))
        
        for i in range(self.num_samples):
            if self.pattern_type == "sinusoidal":
                data[i] = self._generate_sinusoidal_pattern()
            elif self.pattern_type == "circular":
                data[i] = self._generate_circular_pattern()
            elif self.pattern_type == "random":
                data[i] = self._generate_random_pattern()
            elif self.pattern_type == "mixed":
                pattern_types = ["sinusoidal", "circular", "random"]
                pattern = np.random.choice(pattern_types)
                if pattern == "sinusoidal":
                    data[i] = self._generate_sinusoidal_pattern()
                elif pattern == "circular":
                    data[i] = self._generate_circular_pattern()
                else:
                    data[i] = self._generate_random_pattern()
            
            # Add noise
            noise = np.random.normal(0, self.noise_level, data[i].shape)
            data[i] += noise
        
        return data
    
    def _generate_sinusoidal_pattern(self) -> np.ndarray:
        """Generate sinusoidal movement pattern."""
        t = np.linspace(0, 4 * np.pi, self.sequence_length)
        pattern = np.zeros((self.sequence_length, self.feature_dim))
        
        for j in range(self.feature_dim):
            frequency = 0.5 + j * 0.1
            amplitude = 1.0 + j * 0.2
            phase = j * np.pi / 4
            pattern[:, j] = amplitude * np.sin(frequency * t + phase)
        
        return pattern
    
    def _generate_circular_pattern(self) -> np.ndarray:
        """Generate circular movement pattern."""
        t = np.linspace(0, 2 * np.pi, self.sequence_length)
        pattern = np.zeros((self.sequence_length, self.feature_dim))
        
        for j in range(0, self.feature_dim, 2):
            if j + 1 < self.feature_dim:
                radius = 1.0 + j * 0.1
                pattern[:, j] = radius * np.cos(t)
                pattern[:, j + 1] = radius * np.sin(t)
        
        return pattern
    
    def _generate_random_pattern(self) -> np.ndarray:
        """Generate random walk pattern."""
        pattern = np.zeros((self.sequence_length, self.feature_dim))
        
        for j in range(self.feature_dim):
            # Random walk
            steps = np.random.normal(0, 0.1, self.sequence_length)
            pattern[:, j] = np.cumsum(steps)
        
        return pattern
    
    def _normalize_data(self) -> np.ndarray:
        """Normalize the data."""
        # Normalize each feature independently
        normalized_data = np.zeros_like(self.data)
        for j in range(self.feature_dim):
            feature_data = self.data[:, :, j]
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            normalized_data[:, :, j] = (feature_data - mean) / (std + 1e-8)
        
        return normalized_data
    
    def __len__(self) -> int:
        """Return dataset length."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        sequence = self.data[idx]
        
        # For autoregressive generation, input is sequence[:-1], target is sequence[1:]
        input_seq = torch.FloatTensor(sequence[:-1])
        target_seq = torch.FloatTensor(sequence[1:])
        
        return input_seq, target_seq


class DanceDataModule:
    """Data module for dance generation system."""
    
    def __init__(
        self,
        dataset_name: str = "synthetic",
        data_path: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
        sequence_length: int = 50,
        feature_dim: int = 10,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        **dataset_kwargs
    ):
        """Initialize data module.
        
        Args:
            dataset_name: Name of the dataset to use.
            data_path: Path to data directory.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes.
            sequence_length: Length of dance sequences.
            feature_dim: Number of features per timestep.
            train_split: Fraction of data for training.
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing.
            **dataset_kwargs: Additional dataset parameters.
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.dataset_kwargs = dataset_kwargs
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self._setup_datasets()
    
    def _setup_datasets(self) -> None:
        """Setup train, validation, and test datasets."""
        if self.dataset_name == "synthetic":
            self._setup_synthetic_datasets()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _setup_synthetic_datasets(self) -> None:
        """Setup synthetic datasets."""
        # Generate full dataset
        full_dataset = SyntheticDanceDataset(
            sequence_length=self.sequence_length,
            feature_dim=self.feature_dim,
            **self.dataset_kwargs
        )
        
        # Split into train/val/test
        num_samples = len(full_dataset)
        train_size = int(num_samples * self.train_split)
        val_size = int(num_samples * self.val_split)
        test_size = num_samples - train_size - val_size
        
        # Create indices for splitting
        indices = np.arange(num_samples)
        train_indices, temp_indices = train_test_split(
            indices, test_size=val_size + test_size, random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=test_size, random_state=42
        )
        
        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        logging.info(f"Dataset splits - Train: {len(self.train_dataset)}, "
                    f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_dance_data_module(config) -> DanceDataModule:
    """Create dance data module from configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        DanceDataModule instance.
    """
    return DanceDataModule(
        dataset_name=config.data.dataset_name,
        data_path=config.data.data_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        sequence_length=config.data.sequence_length,
        feature_dim=config.data.feature_dim,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split
    )
