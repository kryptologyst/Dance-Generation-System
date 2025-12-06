#!/usr/bin/env python3
"""Main training script for dance generation system."""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.configs.config import load_config, get_default_config
from src.data.dataset import create_dance_data_module
from src.models.training import train_model, evaluate_model
from src.utils.utils import set_seed, setup_logging, get_device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train dance generation model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "transformer", "vae"],
        default="lstm",
        help="Type of model to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate samples after training"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file="assets/logs/training.log"
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    config.model.model_type = args.model_type
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.seed = args.seed
    
    if args.device != "auto":
        config.training.accelerator = args.device
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data module
    logger.info("Creating data module...")
    data_module = create_dance_data_module(config)
    
    # Train model
    logger.info(f"Training {config.model.model_type} model...")
    trainer = train_model(config, data_module)
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_metrics = evaluate_model(trainer, data_module)
    
    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate samples if requested
    if args.generate_samples:
        logger.info(f"Generating {args.num_samples} samples...")
        samples = trainer.generate_and_save_samples(
            num_samples=args.num_samples
        )
        logger.info(f"Generated samples shape: {samples.shape}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
