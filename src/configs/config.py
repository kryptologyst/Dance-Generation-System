"""Configuration management for dance generation system."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for dance generation models."""
    model_type: str = "lstm"  # lstm, transformer, vae, gan
    input_size: int = 10
    output_size: int = 10
    hidden_units: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 50
    latent_dim: int = 64  # for VAE/GAN models


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset_name: str = "synthetic"
    data_path: str = "data/"
    batch_size: int = 32
    num_workers: int = 4
    sequence_length: int = 50
    feature_dim: int = 10
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"  # 16-mixed, 32, bf16-mixed
    devices: int = 1
    accelerator: str = "auto"  # auto, gpu, cpu, mps
    deterministic: bool = True
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = field(default_factory=lambda: ["mse", "mae", "diversity"])
    save_samples: bool = True
    num_samples: int = 100
    sample_frequency: int = 10  # epochs


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    project_root: str = "."
    assets_dir: str = "assets"
    checkpoints_dir: str = "assets/checkpoints"
    samples_dir: str = "assets/samples"
    logs_dir: str = "assets/logs"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_wandb: bool = False
    wandb_project: str = "dance-generation"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Convert relative paths to absolute
        self.project_root = Path(self.project_root).resolve()
        self.assets_dir = self.project_root / self.assets_dir
        self.checkpoints_dir = self.project_root / self.checkpoints_dir
        self.samples_dir = self.project_root / self.samples_dir
        self.logs_dir = self.project_root / self.logs_dir
        
        # Create directories
        for dir_path in [self.assets_dir, self.checkpoints_dir, self.samples_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle Path objects
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return get_default_config()
