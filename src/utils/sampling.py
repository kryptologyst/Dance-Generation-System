"""Sampling and visualization utilities for dance generation."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from ..utils.utils import smooth_sequence, interpolate_sequence, validate_sequence


class DanceVisualizer:
    """Visualizer for dance sequences."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots.
        """
        self.figsize = figsize
    
    def plot_sequence_2d(
        self,
        sequence: np.ndarray,
        title: str = "Dance Sequence",
        save_path: Optional[str] = None
    ) -> None:
        """Plot 2D dance sequence.
        
        Args:
            sequence: Dance sequence of shape (seq_len, feature_dim).
            title: Plot title.
            save_path: Optional path to save the plot.
        """
        plt.figure(figsize=self.figsize)
        
        # Plot each feature dimension
        for i in range(min(sequence.shape[1], 6)):  # Limit to 6 features for readability
            plt.plot(sequence[:, i], label=f'Feature {i}', alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sequence_3d(
        self,
        sequence: np.ndarray,
        title: str = "3D Dance Sequence",
        save_path: Optional[str] = None
    ) -> None:
        """Plot 3D dance sequence.
        
        Args:
            sequence: Dance sequence of shape (seq_len, feature_dim).
            title: Plot title.
            save_path: Optional path to save the plot.
        """
        if sequence.shape[1] < 3:
            logging.warning("Sequence has less than 3 features, cannot create 3D plot")
            return
        
        fig = go.Figure()
        
        # Extract x, y, z coordinates (assuming first 3 features are spatial)
        x = sequence[:, 0]
        y = sequence[:, 1]
        z = sequence[:, 2]
        
        # Create 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(
                size=3,
                color=np.arange(len(sequence)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time Step")
            ),
            line=dict(width=4),
            name="Dance Path"
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_interactive_sequence(
        self,
        sequence: np.ndarray,
        title: str = "Interactive Dance Sequence"
    ) -> go.Figure:
        """Create interactive plotly visualization.
        
        Args:
            sequence: Dance sequence of shape (seq_len, feature_dim).
            title: Plot title.
            
        Returns:
            Plotly figure object.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position Over Time', 'Velocity Over Time', 
                          'Acceleration Over Time', 'Feature Heatmap'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Position plot
        for i in range(min(3, sequence.shape[1])):
            fig.add_trace(
                go.Scatter(
                    y=sequence[:, i],
                    mode='lines',
                    name=f'Feature {i}',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Velocity plot
        if len(sequence) > 1:
            velocity = np.diff(sequence, axis=0)
            for i in range(min(3, velocity.shape[1])):
                fig.add_trace(
                    go.Scatter(
                        y=velocity[:, i],
                        mode='lines',
                        name=f'Velocity {i}',
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
        
        # Acceleration plot
        if len(sequence) > 2:
            acceleration = np.diff(velocity, axis=0)
            for i in range(min(3, acceleration.shape[1])):
                fig.add_trace(
                    go.Scatter(
                        y=acceleration[:, i],
                        mode='lines',
                        name=f'Acceleration {i}',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        # Feature heatmap
        fig.add_trace(
            go.Heatmap(
                z=sequence.T,
                colorscale='Viridis',
                showscale=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_multiple_sequences(
        self,
        sequences: List[np.ndarray],
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot multiple dance sequences for comparison.
        
        Args:
            sequences: List of dance sequences.
            titles: Optional titles for each sequence.
            save_path: Optional path to save the plot.
        """
        num_sequences = len(sequences)
        fig, axes = plt.subplots(num_sequences, 1, figsize=(self.figsize[0], self.figsize[1] * num_sequences))
        
        if num_sequences == 1:
            axes = [axes]
        
        for i, sequence in enumerate(sequences):
            title = titles[i] if titles and i < len(titles) else f"Sequence {i+1}"
            
            # Plot each feature dimension
            for j in range(min(sequence.shape[1], 6)):
                axes[i].plot(sequence[:, j], label=f'Feature {j}', alpha=0.7)
            
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class DanceSampler:
    """Sampler for generating dance sequences."""
    
    def __init__(self, model, device: torch.device):
        """Initialize sampler.
        
        Args:
            model: Trained dance generation model.
            device: Device to run inference on.
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def sample_sequence(
        self,
        initial_input: Optional[torch.Tensor] = None,
        sequence_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> np.ndarray:
        """Sample a single dance sequence.
        
        Args:
            initial_input: Optional initial input tensor.
            sequence_length: Length of sequence to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k sampling.
            top_p: Optional nucleus sampling.
            
        Returns:
            Generated sequence as numpy array.
        """
        with torch.no_grad():
            if initial_input is None:
                # Generate random initial input
                initial_input = torch.randn(1, 1, self.model.input_size, device=self.device)
            
            if hasattr(self.model, 'generate'):
                # Use model's generate method
                generated = self.model.generate(
                    initial_input, sequence_length, temperature
                )
            else:
                # Fallback autoregressive generation
                generated = self._autoregressive_sample(
                    initial_input, sequence_length, temperature, top_k, top_p
                )
            
            return generated.cpu().numpy().squeeze(0)
    
    def _autoregressive_sample(
        self,
        initial_input: torch.Tensor,
        sequence_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Fallback autoregressive sampling."""
        generated_sequence = []
        current_input = initial_input
        
        for _ in range(sequence_length):
            output = self.model(current_input)
            
            if temperature != 1.0:
                output = output / temperature
            
            # Apply top-k or top-p sampling if specified
            if top_k is not None:
                output = self._top_k_sampling(output, top_k)
            elif top_p is not None:
                output = self._top_p_sampling(output, top_p)
            
            next_input = output[:, -1:, :]
            generated_sequence.append(next_input)
            current_input = torch.cat([current_input, next_input], dim=1)
        
        return torch.cat(generated_sequence, dim=1)
    
    def _top_k_sampling(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k sampling."""
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, 1)
        return torch.gather(logits, -1, sampled_indices)
    
    def _top_p_sampling(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def sample_multiple(
        self,
        num_samples: int = 10,
        sequence_length: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> List[np.ndarray]:
        """Sample multiple dance sequences.
        
        Args:
            num_samples: Number of sequences to generate.
            sequence_length: Length of each sequence.
            temperature: Sampling temperature.
            **kwargs: Additional sampling parameters.
            
        Returns:
            List of generated sequences.
        """
        samples = []
        
        for _ in range(num_samples):
            sample = self.sample_sequence(
                sequence_length=sequence_length,
                temperature=temperature,
                **kwargs
            )
            samples.append(sample)
        
        return samples
    
    def interpolate_between_sequences(
        self,
        start_sequence: np.ndarray,
        end_sequence: np.ndarray,
        num_steps: int = 10
    ) -> List[np.ndarray]:
        """Interpolate between two dance sequences.
        
        Args:
            start_sequence: Starting sequence.
            end_sequence: Ending sequence.
            num_steps: Number of interpolation steps.
            
        Returns:
            List of interpolated sequences.
        """
        interpolated_sequences = []
        
        for i in range(num_steps):
            t = i / (num_steps - 1)
            interpolated = (1 - t) * start_sequence + t * end_sequence
            interpolated_sequences.append(interpolated)
        
        return interpolated_sequences
    
    def generate_with_conditioning(
        self,
        condition: torch.Tensor,
        sequence_length: int = 50,
        temperature: float = 1.0
    ) -> np.ndarray:
        """Generate sequence with conditioning (for conditional models).
        
        Args:
            condition: Conditioning tensor.
            sequence_length: Length of sequence to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated sequence.
        """
        # This is a placeholder for conditional generation
        # Implementation depends on the specific model architecture
        logging.warning("Conditional generation not implemented for this model")
        return self.sample_sequence(sequence_length=sequence_length, temperature=temperature)


def create_sampler(model, device: torch.device) -> DanceSampler:
    """Create a dance sampler.
    
    Args:
        model: Trained model.
        device: Device to run on.
        
    Returns:
        DanceSampler instance.
    """
    return DanceSampler(model, device)


def save_samples(
    samples: List[np.ndarray],
    output_dir: str,
    prefix: str = "sample"
) -> None:
    """Save generated samples to files.
    
    Args:
        samples: List of generated sequences.
        output_dir: Output directory.
        prefix: Filename prefix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples):
        filename = f"{prefix}_{i:03d}.npy"
        np.save(output_dir / filename, sample)
    
    logging.info(f"Saved {len(samples)} samples to {output_dir}")


def load_samples(samples_dir: str, prefix: str = "sample") -> List[np.ndarray]:
    """Load samples from directory.
    
    Args:
        samples_dir: Directory containing samples.
        prefix: Filename prefix.
        
    Returns:
        List of loaded sequences.
    """
    samples_dir = Path(samples_dir)
    samples = []
    
    for file_path in sorted(samples_dir.glob(f"{prefix}_*.npy")):
        sample = np.load(file_path)
        samples.append(sample)
    
    logging.info(f"Loaded {len(samples)} samples from {samples_dir}")
    return samples
