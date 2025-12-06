"""Training and evaluation modules for dance generation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..models.models import create_model, count_parameters
from ..utils.utils import calculate_diversity_metrics, get_device


class DanceGenerationModule(pl.LightningModule):
    """PyTorch Lightning module for dance generation."""
    
    def __init__(self, config):
        """Initialize the module.
        
        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = create_model(config)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # For VAE models, we need additional KL loss
        self.is_vae = config.model.model_type.lower() == "vae"
        
        logging.info(f"Model created with {count_parameters(self.model):,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.is_vae:
            recon_x, mu, logvar = self.model(x)
            return recon_x, mu, logvar
        else:
            return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_seq, target_seq = batch
        
        if self.is_vae:
            recon_x, mu, logvar = self.forward(input_seq)
            
            # Reconstruction loss
            recon_loss = self.criterion(recon_x, target_seq)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / input_seq.size(0)  # Normalize by batch size
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss
            
            self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True)
            self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True)
        else:
            output = self.forward(input_seq)
            loss = self.criterion(output, target_seq)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_seq, target_seq = batch
        
        if self.is_vae:
            recon_x, mu, logvar = self.forward(input_seq)
            
            # Reconstruction loss
            recon_loss = self.criterion(recon_x, target_seq)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / input_seq.size(0)
            
            loss = recon_loss + 0.1 * kl_loss
            
            self.log('val/recon_loss', recon_loss, on_step=False, on_epoch=True)
            self.log('val/kl_loss', kl_loss, on_step=False, on_epoch=True)
        else:
            output = self.forward(input_seq)
            loss = self.criterion(output, target_seq)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with evaluation metrics."""
        input_seq, target_seq = batch
        
        if self.is_vae:
            recon_x, mu, logvar = self.forward(input_seq)
            output = recon_x
        else:
            output = self.forward(input_seq)
        
        # Calculate metrics
        mse_loss = self.criterion(output, target_seq)
        mae_loss = nn.L1Loss()(output, target_seq)
        
        # Calculate diversity metrics
        output_np = output.detach().cpu().numpy()
        diversity_metrics = calculate_diversity_metrics(output_np)
        
        metrics = {
            'test/mse': mse_loss,
            'test/mae': mae_loss,
            'test/intra_diversity': diversity_metrics['intra_diversity'],
            'test/inter_diversity': diversity_metrics['inter_diversity'],
            'test/total_diversity': diversity_metrics['total_diversity']
        }
        
        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True)
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }
    
    def generate_samples(
        self,
        num_samples: int = 10,
        sequence_length: int = 50,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate dance samples.
        
        Args:
            num_samples: Number of samples to generate.
            sequence_length: Length of generated sequences.
            temperature: Sampling temperature.
            
        Returns:
            Generated sequences as numpy array.
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            if self.is_vae:
                # VAE generation
                generated = self.model.generate(num_samples, device)
                generated = generated.cpu().numpy()
            else:
                # Autoregressive generation
                generated_samples = []
                
                for _ in range(num_samples):
                    # Random initial input
                    initial_input = torch.randn(1, 1, self.config.model.input_size, device=device)
                    
                    if hasattr(self.model, 'generate'):
                        sample = self.model.generate(initial_input, sequence_length, temperature)
                    else:
                        # Fallback for models without generate method
                        sample = self._autoregressive_generate(
                            initial_input, sequence_length, temperature
                        )
                    
                    generated_samples.append(sample.cpu().numpy())
                
                generated = np.concatenate(generated_samples, axis=0)
        
        return generated
    
    def _autoregressive_generate(
        self,
        initial_input: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Fallback autoregressive generation."""
        generated_sequence = []
        current_input = initial_input
        
        for _ in range(num_steps):
            output = self.forward(current_input)
            
            if temperature != 1.0:
                output = output / temperature
            
            next_input = output[:, -1:, :]
            generated_sequence.append(next_input)
            current_input = torch.cat([current_input, next_input], dim=1)
        
        return torch.cat(generated_sequence, dim=1)


class DanceTrainer:
    """Trainer class for dance generation models."""
    
    def __init__(self, config):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.model_module = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
    
    def setup_training(self, data_module) -> None:
        """Setup training components.
        
        Args:
            data_module: Data module for training.
        """
        # Create model module
        self.model_module = DanceGenerationModule(self.config)
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=str(self.config.logs_dir),
            name="dance_generation"
        )
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.epochs,
            devices=self.config.training.devices,
            accelerator=self.config.training.accelerator,
            precision=self.config.training.precision,
            deterministic=self.config.training.deterministic,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches
        )
        
        self.logger.info("Training setup completed")
    
    def _setup_callbacks(self) -> list:
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoints_dir,
            filename='best-{epoch:02d}-{val/loss:.2f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
        
        return callbacks
    
    def train(self, data_module) -> None:
        """Train the model.
        
        Args:
            data_module: Data module for training.
        """
        if self.trainer is None:
            self.setup_training(data_module)
        
        self.logger.info("Starting training...")
        
        self.trainer.fit(
            self.model_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader()
        )
        
        self.logger.info("Training completed")
    
    def test(self, data_module) -> Dict[str, float]:
        """Test the model.
        
        Args:
            data_module: Data module for testing.
            
        Returns:
            Test metrics.
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_training first.")
        
        self.logger.info("Starting testing...")
        
        test_results = self.trainer.test(
            self.model_module,
            dataloaders=data_module.test_dataloader()
        )
        
        self.logger.info("Testing completed")
        
        return test_results[0] if test_results else {}
    
    def generate_and_save_samples(
        self,
        num_samples: int = 100,
        output_dir: Optional[str] = None
    ) -> np.ndarray:
        """Generate and save samples.
        
        Args:
            num_samples: Number of samples to generate.
            output_dir: Output directory for samples.
            
        Returns:
            Generated samples.
        """
        if self.model_module is None:
            raise ValueError("Model not trained. Call train first.")
        
        if output_dir is None:
            output_dir = self.config.samples_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating {num_samples} samples...")
        
        # Generate samples
        samples = self.model_module.generate_samples(
            num_samples=num_samples,
            sequence_length=self.config.data.sequence_length
        )
        
        # Save samples
        np.save(output_dir / f"samples_{num_samples}.npy", samples)
        
        self.logger.info(f"Samples saved to {output_dir}")
        
        return samples


def train_model(config, data_module) -> DanceTrainer:
    """Train a dance generation model.
    
    Args:
        config: Configuration object.
        data_module: Data module.
        
    Returns:
        Trained trainer instance.
    """
    trainer = DanceTrainer(config)
    trainer.train(data_module)
    return trainer


def evaluate_model(trainer: DanceTrainer, data_module) -> Dict[str, float]:
    """Evaluate a trained model.
    
    Args:
        trainer: Trained trainer instance.
        data_module: Data module.
        
    Returns:
        Evaluation metrics.
    """
    return trainer.test(data_module)
