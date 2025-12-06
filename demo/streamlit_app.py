"""Streamlit demo for dance generation system."""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.configs.config import get_default_config
from src.models.models import create_model
from src.utils.utils import set_seed, get_device
from src.utils.sampling import create_sampler, DanceVisualizer


def load_model(checkpoint_path: str, config):
    """Load model from checkpoint."""
    device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Create and load model
    model = create_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device


def create_3d_dance_plot(sequence):
    """Create 3D plot of dance sequence."""
    if sequence.shape[1] < 3:
        return None
    
    fig = go.Figure()
    
    # Extract x, y, z coordinates
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
        title="3D Dance Sequence",
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position'
        ),
        width=800,
        height=600
    )
    
    return fig


def create_feature_plot(sequence):
    """Create feature plot over time."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Position Over Time', 'Velocity Over Time'),
        vertical_spacing=0.1
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
                row=2, col=1
            )
    
    fig.update_layout(
        title="Dance Sequence Analysis",
        height=600,
        showlegend=True
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Dance Generation Demo",
        page_icon="💃",
        layout="wide"
    )
    
    st.title("💃 Dance Generation System")
    st.markdown("Generate and visualize dance sequences using AI models")
    
    # Sidebar for controls
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["lstm", "transformer", "vae"],
        index=0
    )
    
    # Checkpoint upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=['pth', 'pt'],
        help="Upload a trained model checkpoint"
    )
    
    if uploaded_file is None:
        st.warning("Please upload a model checkpoint to generate dance sequences.")
        st.stop()
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    num_samples = st.sidebar.slider("Number of Samples", 1, 20, 5)
    sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        top_k = st.number_input("Top-k Sampling", value=None, min_value=1, help="Set to None to disable")
        top_p = st.slider("Top-p (Nucleus) Sampling", 0.0, 1.0, 1.0, 0.1, help="Set to 1.0 to disable")
    
    # Load model
    try:
        # Save uploaded file temporarily
        checkpoint_path = f"temp_checkpoint_{model_type}.pth"
        with open(checkpoint_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load configuration
        config = get_default_config()
        config.model.model_type = model_type
        
        # Set seed
        set_seed(seed)
        
        # Load model
        with st.spinner("Loading model..."):
            model, device = load_model(checkpoint_path, config)
        
        st.success(f"Model loaded successfully! ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        # Create sampler
        sampler = create_sampler(model, device)
        
        # Generate samples
        if st.button("Generate Dance Sequences", type="primary"):
            with st.spinner("Generating dance sequences..."):
                samples = sampler.sample_multiple(
                    num_samples=num_samples,
                    sequence_length=sequence_length,
                    temperature=temperature,
                    top_k=top_k if top_k else None,
                    top_p=top_p if top_p < 1.0 else None
                )
            
            st.success(f"Generated {len(samples)} dance sequences!")
            
            # Display samples
            st.header("Generated Dance Sequences")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["2D Plots", "3D Visualization", "Analysis"])
            
            with tab1:
                st.subheader("2D Feature Plots")
                
                # Plot first few samples
                for i, sample in enumerate(samples[:5]):
                    st.subheader(f"Sample {i+1}")
                    
                    fig = px.line(
                        x=range(len(sample)),
                        y=sample,
                        title=f"Dance Sequence {i+1}",
                        labels={'x': 'Time Step', 'y': 'Value'}
                    )
                    
                    # Add traces for each feature
                    for j in range(sample.shape[1]):
                        fig.add_scatter(
                            x=list(range(len(sample))),
                            y=sample[:, j],
                            mode='lines',
                            name=f'Feature {j}',
                            line=dict(width=2)
                        )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("3D Dance Visualization")
                
                # Show 3D plot for first sample
                if samples and samples[0].shape[1] >= 3:
                    fig_3d = create_3d_dance_plot(samples[0])
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.warning("Need at least 3 features for 3D visualization")
                else:
                    st.warning("Need at least 3 features for 3D visualization")
            
            with tab3:
                st.subheader("Sequence Analysis")
                
                # Statistics
                all_samples = np.array(samples)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Value", f"{np.mean(all_samples):.4f}")
                with col2:
                    st.metric("Std Deviation", f"{np.std(all_samples):.4f}")
                with col3:
                    st.metric("Min Value", f"{np.min(all_samples):.4f}")
                with col4:
                    st.metric("Max Value", f"{np.max(all_samples):.4f}")
                
                # Feature correlation heatmap
                if len(samples) > 1:
                    st.subheader("Feature Correlation")
                    
                    # Calculate correlation matrix
                    flat_samples = all_samples.reshape(-1, all_samples.shape[-1])
                    corr_matrix = np.corrcoef(flat_samples.T)
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Feature Correlation Matrix",
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Velocity analysis
                st.subheader("Velocity Analysis")
                
                velocities = []
                for sample in samples:
                    if len(sample) > 1:
                        vel = np.linalg.norm(np.diff(sample, axis=0), axis=1)
                        velocities.extend(vel)
                
                if velocities:
                    fig_vel = px.histogram(
                        x=velocities,
                        title="Velocity Distribution",
                        labels={'x': 'Velocity', 'y': 'Count'}
                    )
                    st.plotly_chart(fig_vel, use_container_width=True)
            
            # Download samples
            st.header("Download Samples")
            
            # Convert samples to downloadable format
            samples_array = np.array(samples)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download as NumPy (.npy)",
                    data=samples_array.tobytes(),
                    file_name="dance_samples.npy",
                    mime="application/octet-stream"
                )
            
            with col2:
                # Convert to CSV
                csv_data = samples_array.reshape(-1, samples_array.shape[-1])
                csv_string = np.savetxt("temp.csv", csv_data, delimiter=",")
                
                with open("temp.csv", "r") as f:
                    csv_content = f.read()
                
                st.download_button(
                    label="Download as CSV",
                    data=csv_content,
                    file_name="dance_samples.csv",
                    mime="text/csv"
                )
        
        # Clean up temporary files
        Path(checkpoint_path).unlink(missing_ok=True)
        Path("temp.csv").unlink(missing_ok=True)
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
