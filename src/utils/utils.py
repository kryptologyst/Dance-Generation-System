"""Utility functions for dance generation system."""

import random
import numpy as np
import torch
from typing import Tuple, Optional, Union, List
import logging
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
        log_format: Optional custom log format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def normalize_sequence(sequence: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize dance sequence data.
    
    Args:
        sequence: Input sequence data.
        method: Normalization method ('minmax', 'zscore', 'unit').
        
    Returns:
        Normalized sequence.
    """
    if method == "minmax":
        min_val = sequence.min(axis=0, keepdims=True)
        max_val = sequence.max(axis=0, keepdims=True)
        return (sequence - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "zscore":
        mean = sequence.mean(axis=0, keepdims=True)
        std = sequence.std(axis=0, keepdims=True)
        return (sequence - mean) / (std + 1e-8)
    
    elif method == "unit":
        norm = np.linalg.norm(sequence, axis=-1, keepdims=True)
        return sequence / (norm + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_sequence(
    normalized_sequence: np.ndarray,
    original_sequence: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """Denormalize dance sequence data.
    
    Args:
        normalized_sequence: Normalized sequence data.
        original_sequence: Original sequence for reference statistics.
        method: Normalization method used.
        
    Returns:
        Denormalized sequence.
    """
    if method == "minmax":
        min_val = original_sequence.min(axis=0, keepdims=True)
        max_val = original_sequence.max(axis=0, keepdims=True)
        return normalized_sequence * (max_val - min_val) + min_val
    
    elif method == "zscore":
        mean = original_sequence.mean(axis=0, keepdims=True)
        std = original_sequence.std(axis=0, keepdims=True)
        return normalized_sequence * std + mean
    
    elif method == "unit":
        norm = np.linalg.norm(original_sequence, axis=-1, keepdims=True)
        return normalized_sequence * norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_diversity_metrics(sequences: np.ndarray) -> dict:
    """Calculate diversity metrics for generated sequences.
    
    Args:
        sequences: Array of generated sequences.
        
    Returns:
        Dictionary containing diversity metrics.
    """
    if len(sequences) < 2:
        return {"intra_diversity": 0.0, "inter_diversity": 0.0}
    
    # Intra-sequence diversity (within each sequence)
    intra_diversities = []
    for seq in sequences:
        if len(seq) > 1:
            pairwise_distances = np.linalg.norm(
                seq[1:] - seq[:-1], axis=-1
            )
            intra_diversities.append(np.mean(pairwise_distances))
    
    intra_diversity = np.mean(intra_diversities) if intra_diversities else 0.0
    
    # Inter-sequence diversity (between sequences)
    if len(sequences) > 1:
        pairwise_distances = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                # Use mean sequence as representative
                seq_i_mean = np.mean(sequences[i], axis=0)
                seq_j_mean = np.mean(sequences[j], axis=0)
                distance = np.linalg.norm(seq_i_mean - seq_j_mean)
                pairwise_distances.append(distance)
        
        inter_diversity = np.mean(pairwise_distances) if pairwise_distances else 0.0
    else:
        inter_diversity = 0.0
    
    return {
        "intra_diversity": float(intra_diversity),
        "inter_diversity": float(inter_diversity),
        "total_diversity": float(intra_diversity + inter_diversity)
    }


def smooth_sequence(sequence: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply smoothing to dance sequence.
    
    Args:
        sequence: Input sequence.
        window_size: Size of smoothing window.
        
    Returns:
        Smoothed sequence.
    """
    if window_size <= 1:
        return sequence
    
    smoothed = np.zeros_like(sequence)
    half_window = window_size // 2
    
    for i in range(len(sequence)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(sequence), i + half_window + 1)
        smoothed[i] = np.mean(sequence[start_idx:end_idx], axis=0)
    
    return smoothed


def interpolate_sequence(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_steps: int
) -> np.ndarray:
    """Interpolate between two poses.
    
    Args:
        start_pose: Starting pose.
        end_pose: Ending pose.
        num_steps: Number of interpolation steps.
        
    Returns:
        Interpolated sequence.
    """
    if num_steps <= 1:
        return np.array([start_pose])
    
    t = np.linspace(0, 1, num_steps).reshape(-1, 1)
    interpolated = start_pose + t * (end_pose - start_pose)
    
    return interpolated


def validate_sequence(sequence: np.ndarray, max_joint_velocity: float = 10.0) -> bool:
    """Validate dance sequence for realistic constraints.
    
    Args:
        sequence: Input sequence.
        max_joint_velocity: Maximum allowed joint velocity.
        
    Returns:
        True if sequence is valid, False otherwise.
    """
    if len(sequence) < 2:
        return True
    
    # Check for excessive velocities
    velocities = np.linalg.norm(sequence[1:] - sequence[:-1], axis=-1)
    if np.any(velocities > max_joint_velocity):
        return False
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(sequence)):
        return False
    
    return True


def save_sequence(
    sequence: np.ndarray,
    filepath: Union[str, Path],
    format: str = "npy"
) -> None:
    """Save dance sequence to file.
    
    Args:
        sequence: Dance sequence to save.
        filepath: Output file path.
        format: File format ('npy', 'csv', 'json').
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "npy":
        np.save(filepath, sequence)
    elif format == "csv":
        np.savetxt(filepath, sequence.reshape(-1, sequence.shape[-1]), delimiter=",")
    elif format == "json":
        import json
        with open(filepath, 'w') as f:
            json.dump(sequence.tolist(), f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_sequence(
    filepath: Union[str, Path],
    format: str = "npy"
) -> np.ndarray:
    """Load dance sequence from file.
    
    Args:
        filepath: Input file path.
        format: File format ('npy', 'csv', 'json').
        
    Returns:
        Loaded dance sequence.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == "npy":
        return np.load(filepath)
    elif format == "csv":
        return np.loadtxt(filepath, delimiter=",")
    elif format == "json":
        import json
        with open(filepath, 'r') as f:
            return np.array(json.load(f))
    else:
        raise ValueError(f"Unsupported format: {format}")
