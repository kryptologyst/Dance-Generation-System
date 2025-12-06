"""Unit tests for dance generation system."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.configs.config import get_default_config, Config
from src.models.models import create_model, DanceLSTM, DanceTransformer, DanceVAE, count_parameters
from src.data.dataset import SyntheticDanceDataset, DanceDataModule
from src.utils.utils import (
    set_seed, get_device, normalize_sequence, denormalize_sequence,
    calculate_diversity_metrics, smooth_sequence, interpolate_sequence,
    validate_sequence
)
from src.utils.sampling import DanceSampler, DanceVisualizer


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        assert isinstance(config, Config)
        assert config.model.model_type == "lstm"
        assert config.training.epochs == 100
        assert config.data.batch_size == 32
    
    def test_config_update(self):
        """Test configuration updates."""
        config = get_default_config()
        config.update({"training": {"epochs": 200}})
        assert config.training.epochs == 200


class TestModels:
    """Test model creation and functionality."""
    
    def test_lstm_model(self):
        """Test LSTM model creation and forward pass."""
        model = DanceLSTM(input_size=10, output_size=10, hidden_units=64)
        
        # Test forward pass
        x = torch.randn(2, 20, 10)
        output, hidden = model(x)
        
        assert output.shape == (2, 20, 10)
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2
    
    def test_transformer_model(self):
        """Test Transformer model creation and forward pass."""
        model = DanceTransformer(input_size=10, output_size=10, d_model=64)
        
        # Test forward pass
        x = torch.randn(2, 20, 10)
        output = model(x)
        
        assert output.shape == (2, 20, 10)
    
    def test_vae_model(self):
        """Test VAE model creation and forward pass."""
        model = DanceVAE(input_size=10, sequence_length=20, latent_dim=16)
        
        # Test forward pass
        x = torch.randn(2, 20, 10)
        recon_x, mu, logvar = model(x)
        
        assert recon_x.shape == (2, 20, 10)
        assert mu.shape == (2, 16)
        assert logvar.shape == (2, 16)
    
    def test_model_creation_from_config(self):
        """Test model creation from configuration."""
        config = get_default_config()
        
        for model_type in ["lstm", "transformer", "vae"]:
            config.model.model_type = model_type
            model = create_model(config)
            assert model is not None
    
    def test_parameter_counting(self):
        """Test parameter counting."""
        model = DanceLSTM(input_size=10, output_size=10, hidden_units=64)
        param_count = count_parameters(model)
        assert param_count > 0


class TestDataModule:
    """Test data module functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticDanceDataset(
            num_samples=100,
            sequence_length=20,
            feature_dim=5
        )
        
        assert len(dataset) == 100
        
        # Test data loading
        input_seq, target_seq = dataset[0]
        assert input_seq.shape == (19, 5)  # sequence_length - 1
        assert target_seq.shape == (19, 5)
    
    def test_data_module(self):
        """Test data module creation."""
        data_module = DanceDataModule(
            dataset_name="synthetic",
            batch_size=16,
            sequence_length=20,
            feature_dim=5
        )
        
        # Test data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        rand1 = np.random.rand()
        
        set_seed(42)
        rand2 = np.random.rand()
        
        assert rand1 == rand2
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_normalization(self):
        """Test sequence normalization."""
        sequence = np.random.rand(50, 10)
        
        # Test minmax normalization
        normalized = normalize_sequence(sequence, method="minmax")
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        
        # Test denormalization
        denormalized = denormalize_sequence(normalized, sequence, method="minmax")
        np.testing.assert_allclose(denormalized, sequence, rtol=1e-5)
    
    def test_diversity_metrics(self):
        """Test diversity metrics calculation."""
        sequences = np.random.rand(5, 20, 10)
        metrics = calculate_diversity_metrics(sequences)
        
        assert "intra_diversity" in metrics
        assert "inter_diversity" in metrics
        assert "total_diversity" in metrics
        
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_sequence_smoothing(self):
        """Test sequence smoothing."""
        sequence = np.random.rand(20, 5)
        smoothed = smooth_sequence(sequence, window_size=3)
        
        assert smoothed.shape == sequence.shape
    
    def test_sequence_interpolation(self):
        """Test sequence interpolation."""
        start_pose = np.array([1, 2, 3])
        end_pose = np.array([4, 5, 6])
        
        interpolated = interpolate_sequence(start_pose, end_pose, num_steps=5)
        
        assert interpolated.shape == (5, 3)
        np.testing.assert_array_equal(interpolated[0], start_pose)
        np.testing.assert_array_equal(interpolated[-1], end_pose)
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        # Valid sequence
        valid_seq = np.random.rand(20, 5) * 0.1
        assert validate_sequence(valid_seq)
        
        # Invalid sequence (too fast)
        invalid_seq = np.random.rand(20, 5) * 100
        assert not validate_sequence(invalid_seq, max_joint_velocity=1.0)


class TestSampling:
    """Test sampling functionality."""
    
    def test_visualizer_creation(self):
        """Test visualizer creation."""
        visualizer = DanceVisualizer()
        assert visualizer is not None
    
    def test_sampler_creation(self):
        """Test sampler creation."""
        model = DanceLSTM(input_size=10, output_size=10, hidden_units=64)
        device = torch.device("cpu")
        sampler = DanceSampler(model, device)
        assert sampler is not None


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training process."""
        config = get_default_config()
        config.training.epochs = 1  # Quick test
        config.data.batch_size = 4
        
        # Create data module
        data_module = DanceDataModule(
            dataset_name="synthetic",
            batch_size=config.data.batch_size,
            sequence_length=config.data.sequence_length,
            feature_dim=config.data.feature_dim
        )
        
        # Create model
        model = create_model(config)
        
        # Test forward pass
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        input_seq, target_seq = batch
        
        if config.model.model_type == "vae":
            recon_x, mu, logvar = model(input_seq)
            assert recon_x.shape == target_seq.shape
        else:
            output = model(input_seq)
            assert output.shape == target_seq.shape


if __name__ == "__main__":
    pytest.main([__file__])
