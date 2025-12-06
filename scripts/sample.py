#!/usr/bin/env python3
"""Sampling script for dance generation system."""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.configs.config import load_config, get_default_config
from src.models.models import create_model
from src.utils.utils import set_seed, setup_logging, get_device
from src.utils.sampling import create_sampler, DanceVisualizer, save_samples


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate dance samples")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Length of generated sequences"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/samples",
        help="Output directory for samples"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize generated samples"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Create interactive visualizations"
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file="assets/logs/sampling.log"
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model state dict and config if available
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Create model
    model = create_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sampler
    sampler = create_sampler(model, device)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    samples = sampler.sample_multiple(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    logger.info(f"Generated {len(samples)} samples")
    
    # Save samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_samples(samples, str(output_dir), prefix="generated")
    
    # Also save as single numpy array
    samples_array = np.array(samples)
    np.save(output_dir / "all_samples.npy", samples_array)
    
    logger.info(f"Samples saved to {output_dir}")
    
    # Visualize samples if requested
    if args.visualize:
        logger.info("Creating visualizations...")
        visualizer = DanceVisualizer()
        
        # Plot first few samples
        num_plots = min(5, len(samples))
        for i in range(num_plots):
            visualizer.plot_sequence_2d(
                samples[i],
                title=f"Generated Dance Sequence {i+1}",
                save_path=output_dir / f"sequence_{i+1:03d}.png"
            )
        
        # Create comparison plot
        visualizer.plot_multiple_sequences(
            samples[:num_plots],
            titles=[f"Sequence {i+1}" for i in range(num_plots)],
            save_path=output_dir / "comparison.png"
        )
        
        logger.info("Visualizations saved")
    
    # Create interactive visualizations if requested
    if args.interactive:
        logger.info("Creating interactive visualizations...")
        visualizer = DanceVisualizer()
        
        # Create interactive plot for first sample
        if samples:
            fig = visualizer.plot_interactive_sequence(
                samples[0],
                title="Interactive Dance Sequence"
            )
            
            # Save interactive plot
            fig.write_html(str(output_dir / "interactive_sequence.html"))
            
            logger.info("Interactive visualization saved")
    
    # Print sample statistics
    logger.info("Sample statistics:")
    logger.info(f"  Shape: {samples_array.shape}")
    logger.info(f"  Mean: {np.mean(samples_array):.4f}")
    logger.info(f"  Std: {np.std(samples_array):.4f}")
    logger.info(f"  Min: {np.min(samples_array):.4f}")
    logger.info(f"  Max: {np.max(samples_array):.4f}")
    
    logger.info("Sampling completed successfully!")


if __name__ == "__main__":
    main()
