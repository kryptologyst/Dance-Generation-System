# Dance Generation System

A production-ready dance generation system using deep learning models including LSTM, Transformer, and VAE architectures. This project generates coherent and expressive dance sequences from synthetic motion data.

## Features

- **Multiple Model Architectures**: LSTM, Transformer, and VAE-based dance generation
- **Comprehensive Evaluation**: MSE, MAE, and diversity metrics
- **Interactive Visualization**: 2D/3D plots and interactive Streamlit demo
- **Production Ready**: Proper configuration management, logging, and testing
- **Device Support**: Automatic CUDA/MPS/CPU detection
- **Reproducible**: Deterministic seeding and structured experiments

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Dance-Generation-System.git
cd Dance-Generation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Training a Model

Train an LSTM model with default settings:
```bash
python scripts/train.py --model-type lstm --epochs 50
```

Train a Transformer model with custom configuration:
```bash
python scripts/train.py --config configs/transformer.yaml --epochs 100
```

Train a VAE model:
```bash
python scripts/train.py --config configs/vae.yaml --epochs 200
```

### Generating Samples

Generate samples from a trained model:
```bash
python scripts/sample.py --checkpoint assets/checkpoints/best-model.pth --num-samples 20 --visualize
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
0394_Dance_generation_system/
├── src/                          # Source code
│   ├── configs/                  # Configuration management
│   │   └── config.py
│   ├── data/                    # Data pipeline
│   │   └── dataset.py
│   ├── models/                  # Model definitions
│   │   ├── models.py
│   │   └── training.py
│   └── utils/                   # Utilities
│       ├── utils.py
│       └── sampling.py
├── scripts/                     # Training and sampling scripts
│   ├── train.py
│   └── sample.py
├── configs/                     # Configuration files
│   ├── default.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   └── vae.yaml
├── demo/                        # Interactive demos
│   └── streamlit_app.py
├── tests/                       # Unit tests
│   └── test_dance_generation.py
├── assets/                      # Generated assets
│   ├── checkpoints/            # Model checkpoints
│   ├── samples/                # Generated samples
│   └── logs/                   # Training logs
├── requirements.txt             # Dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Model Architectures

### LSTM Model
- Bidirectional LSTM layers for temporal modeling
- Dropout for regularization
- Autoregressive generation capability
- Best for: Sequential pattern learning

### Transformer Model
- Multi-head self-attention mechanism
- Positional encoding for sequence understanding
- Parallel processing capability
- Best for: Long-range dependencies

### VAE Model
- Encoder-decoder architecture with latent space
- Reparameterization trick for training
- Probabilistic generation
- Best for: Diverse and smooth generation

## Configuration

The system uses YAML configuration files for easy experimentation:

```yaml
model:
  model_type: "lstm"          # lstm, transformer, vae
  input_size: 10
  output_size: 10
  hidden_units: 128
  num_layers: 2
  dropout: 0.1

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  precision: "16-mixed"       # Mixed precision training
```

## Data Pipeline

The system includes a synthetic dance dataset generator that creates:
- Sinusoidal movement patterns
- Circular motion patterns
- Random walk patterns
- Mixed pattern combinations

Each sequence represents a dance movement with configurable:
- Sequence length (default: 50 timesteps)
- Feature dimensions (default: 10 features)
- Pattern types and noise levels

## Evaluation Metrics

### Quality Metrics
- **MSE (Mean Squared Error)**: Reconstruction quality
- **MAE (Mean Absolute Error)**: Average deviation
- **KL Divergence**: Latent space quality (VAE only)

### Diversity Metrics
- **Intra-sequence Diversity**: Movement variation within sequences
- **Inter-sequence Diversity**: Variation between different sequences
- **Total Diversity**: Combined diversity measure

## Sampling and Visualization

### Sampling Methods
- **Temperature Sampling**: Control randomness (0.1-2.0)
- **Top-k Sampling**: Limit vocabulary size
- **Nucleus Sampling**: Dynamic vocabulary selection

### Visualization Options
- 2D feature plots over time
- 3D trajectory visualization
- Interactive Plotly charts
- Velocity and acceleration analysis
- Feature correlation heatmaps

## Training Commands

### Basic Training
```bash
# LSTM model
python scripts/train.py --model-type lstm --epochs 100

# Transformer model
python scripts/train.py --model-type transformer --epochs 200

# VAE model
python scripts/train.py --model-type vae --epochs 300
```

### Advanced Training
```bash
# Custom configuration
python scripts/train.py --config configs/transformer.yaml

# Multi-GPU training
python scripts/train.py --model-type lstm --devices 2

# Mixed precision
python scripts/train.py --model-type transformer --precision 16-mixed

# Generate samples after training
python scripts/train.py --model-type lstm --generate-samples --num-samples 100
```

## Sampling Commands

### Basic Sampling
```bash
# Generate samples
python scripts/sample.py --checkpoint assets/checkpoints/best-model.pth

# Generate with visualization
python scripts/sample.py --checkpoint best-model.pth --visualize --interactive

# Custom parameters
python scripts/sample.py --checkpoint best-model.pth --num-samples 50 --temperature 0.8
```

### Advanced Sampling
```bash
# Top-k sampling
python scripts/sample.py --checkpoint best-model.pth --top-k 10

# Nucleus sampling
python scripts/sample.py --checkpoint best-model.pth --top-p 0.9

# Long sequences
python scripts/sample.py --checkpoint best-model.pth --sequence-length 100
```

## Interactive Demo

The Streamlit demo provides:
- Model upload and selection
- Real-time parameter adjustment
- Multiple visualization modes
- Sample download options
- Statistical analysis

Launch with:
```bash
streamlit run demo/streamlit_app.py
```

## Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_dance_generation.py::TestModels
pytest tests/test_dance_generation.py::TestDataModule
```

## Configuration Files

### Default Configuration (`configs/default.yaml`)
- Balanced settings for general use
- LSTM model with moderate complexity
- Standard training parameters

### LSTM Configuration (`configs/lstm.yaml`)
- Optimized for LSTM models
- Larger hidden units and more layers
- Higher batch size for efficiency

### Transformer Configuration (`configs/transformer.yaml`)
- Optimized for Transformer models
- Lower learning rate for stability
- Gradient accumulation for larger effective batch size

### VAE Configuration (`configs/vae.yaml`)
- Optimized for VAE models
- Smaller latent dimension
- Longer training for convergence

## Performance Tips

### Training Optimization
- Use mixed precision training (`--precision 16-mixed`)
- Enable gradient accumulation for larger effective batch sizes
- Use appropriate learning rates (0.001 for LSTM, 0.0001 for Transformer)
- Monitor validation loss for early stopping

### Generation Quality
- Lower temperature (0.5-0.8) for more coherent sequences
- Higher temperature (1.2-1.5) for more diverse sequences
- Use top-k or nucleus sampling for better quality
- Validate generated sequences for realistic constraints

### Memory Management
- Reduce batch size if running out of memory
- Use gradient checkpointing for large models
- Enable CPU offloading for very large models

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

**Training Instability**
- Lower learning rate
- Increase gradient clipping
- Check data normalization

**Poor Generation Quality**
- Increase model capacity
- Train for more epochs
- Adjust sampling temperature
- Use different sampling strategies

### Debug Mode
Enable debug logging:
```bash
python scripts/train.py --log-level DEBUG
```

## Model Cards

### LSTM Model
- **Architecture**: Bidirectional LSTM with dropout
- **Parameters**: ~50K-200K depending on configuration
- **Training Time**: 1-2 hours on modern GPU
- **Best Use Case**: Sequential pattern learning, quick prototyping
- **Limitations**: Limited long-range dependencies

### Transformer Model
- **Architecture**: Multi-head self-attention with positional encoding
- **Parameters**: ~100K-500K depending on configuration
- **Training Time**: 2-4 hours on modern GPU
- **Best Use Case**: Long sequences, complex patterns
- **Limitations**: Higher computational requirements

### VAE Model
- **Architecture**: Encoder-decoder with latent space
- **Parameters**: ~30K-150K depending on configuration
- **Training Time**: 2-3 hours on modern GPU
- **Best Use Case**: Diverse generation, smooth interpolation
- **Limitations**: Potential mode collapse, blurry outputs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dance_generation_system,
  title={Dance Generation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Dance-Generation-System}
}
```

## Acknowledgments

- PyTorch Lightning for training infrastructure
- Streamlit for interactive demos
- Plotly for visualization capabilities
- The dance and motion capture research community
# Dance-Generation-System
