"""Dance generation models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import logging


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]


class DanceLSTM(nn.Module):
    """LSTM-based dance generation model."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """Initialize LSTM model.
        
        Args:
            input_size: Input feature dimension.
            output_size: Output feature dimension.
            hidden_units: Number of hidden units.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_units * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(lstm_output_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size).
            hidden: Optional hidden state.
            
        Returns:
            Output sequence and hidden state.
        """
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.output_projection(self.dropout(lstm_out))
        
        return output, hidden
    
    def generate(
        self,
        initial_input: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate dance sequence autoregressively.
        
        Args:
            initial_input: Initial input of shape (batch_size, 1, input_size).
            num_steps: Number of steps to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated sequence of shape (batch_size, num_steps, output_size).
        """
        self.eval()
        device = initial_input.device
        batch_size = initial_input.size(0)
        
        generated_sequence = []
        current_input = initial_input
        hidden = None
        
        with torch.no_grad():
            for _ in range(num_steps):
                output, hidden = self.forward(current_input, hidden)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    output = output / temperature
                
                # Use the last timestep as next input
                next_input = output[:, -1:, :]
                generated_sequence.append(next_input)
                
                # Update current input for next iteration
                current_input = next_input
        
        return torch.cat(generated_sequence, dim=1)


class DanceTransformer(nn.Module):
    """Transformer-based dance generation model."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        """Initialize transformer model.
        
        Args:
            input_size: Input feature dimension.
            output_size: Output feature dimension.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_feedforward: Feedforward dimension.
            dropout: Dropout rate.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size).
            mask: Optional attention mask.
            
        Returns:
            Output sequence of shape (batch_size, seq_len, output_size).
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to output dimension
        output = self.output_projection(self.dropout(x))
        
        return output
    
    def generate(
        self,
        initial_input: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate dance sequence autoregressively.
        
        Args:
            initial_input: Initial input of shape (batch_size, 1, input_size).
            num_steps: Number of steps to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated sequence of shape (batch_size, num_steps, output_size).
        """
        self.eval()
        device = initial_input.device
        batch_size = initial_input.size(0)
        
        generated_sequence = []
        current_input = initial_input
        
        with torch.no_grad():
            for _ in range(num_steps):
                output = self.forward(current_input)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    output = output / temperature
                
                # Use the last timestep as next input
                next_input = output[:, -1:, :]
                generated_sequence.append(next_input)
                
                # Update current input for next iteration
                current_input = torch.cat([current_input, next_input], dim=1)
        
        return torch.cat(generated_sequence, dim=1)


class DanceVAE(nn.Module):
    """Variational Autoencoder for dance generation."""
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """Initialize VAE model.
        
        Args:
            input_size: Input feature dimension.
            sequence_length: Length of dance sequences.
            latent_dim: Latent space dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of layers in encoder/decoder.
        """
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        current_dim = input_size * sequence_length
        
        for i in range(num_layers):
            next_dim = hidden_dim // (2 ** i)
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim // (2 ** (num_layers - 1 - i))
            decoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        decoder_layers.append(nn.Linear(current_dim, input_size * sequence_length))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size).
            
        Returns:
            Mean and log variance of latent distribution.
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.
        
        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to sequence.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            
        Returns:
            Reconstructed sequence of shape (batch_size, seq_len, input_size).
        """
        batch_size = z.size(0)
        x_flat = self.decoder(z)
        x = x_flat.view(batch_size, self.sequence_length, self.input_size)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size).
            
        Returns:
            Reconstructed sequence, mean, and log variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new dance sequences.
        
        Args:
            num_samples: Number of samples to generate.
            device: Device to generate on.
            
        Returns:
            Generated sequences of shape (num_samples, seq_len, input_size).
        """
        self.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            generated = self.decode(z)
        
        return generated


def create_model(config) -> nn.Module:
    """Create model from configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Model instance.
    """
    model_type = config.model.model_type.lower()
    
    if model_type == "lstm":
        return DanceLSTM(
            input_size=config.model.input_size,
            output_size=config.model.output_size,
            hidden_units=config.model.hidden_units,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    
    elif model_type == "transformer":
        return DanceTransformer(
            input_size=config.model.input_size,
            output_size=config.model.output_size,
            d_model=config.model.hidden_units,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    
    elif model_type == "vae":
        return DanceVAE(
            input_size=config.model.input_size,
            sequence_length=config.model.sequence_length,
            latent_dim=config.model.latent_dim
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
