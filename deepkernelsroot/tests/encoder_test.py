import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from unittest.mock import patch, MagicMock


from deepkernels.models.encoder import VAEConfig, ConvolutionalLoopEncoder, ConvolutionalNetwork1D

# --- Fixtures ---
@pytest.fixture
def base_config():
    return VAEConfig(
        input_dim=30,
        latent_dim=16,
        k_atoms=30,
        M=128
    )

@pytest.fixture
def mock_base_methods():
    """
    Mocks the methods inherited from BaseGenerativeModel.
    Notice the updated target string for patch.multiple!
    """
    # 2. Update the patch path to match where the class actually lives
    with patch.multiple(
        'deepkernels.models.encoder.ConvolutionalLoopEncoder', 
        reparameterise=MagicMock(side_effect=lambda mu, logvar: mu + torch.exp(0.5 * logvar) * 0.1),
        lowrankmultivariatenorm=MagicMock(return_value=torch.randn(2, 30))
    ) as mocks:
        yield mocks

@pytest.fixture
def encoder(base_config, mock_base_methods):
    return ConvolutionalLoopEncoder(config=base_config)

# --- Tests for VAEConfig ---

def test_vae_config_defaults():
    """Test that VAEConfig initializes with correct default values."""
    config = VAEConfig()
    assert config.input_dim == 30
    assert config.latent_dim == 16
    assert config.k_atoms == 30
    assert config.M == 128

def test_vae_config_validation():
    """Test that VAEConfig properly enforces Pydantic constraints."""
    with pytest.raises(ValidationError):
        # latent_dim has a minimum (ge=8)
        VAEConfig(latent_dim=4)
        
    with pytest.raises(ValidationError):
        # M has a maximum (le=512)
        VAEConfig(M=1024)

# --- Tests for ConvolutionalNetwork1D ---

def test_conv1d_network_shape():
    """Test the 1D ResNet block shape preservation and transformation."""
    batch_size = 4
    in_channels = 32
    out_channels = 64
    seq_len = 10
    
    model = ConvolutionalNetwork1D(in_channels, out_channels, stride=2)
    x = torch.randn(batch_size, in_channels, seq_len)
    out = model(x)
    
    # Stride of 2 should halve the sequence length
    expected_seq_len = seq_len // 2
    assert out.shape == (batch_size, out_channels, expected_seq_len)

# --- Tests for ConvolutionalLoopEncoder ---

def test_run_convolutional_layers_2d_input(encoder):
    """Test the convolutional stack with [Batch, Features] input."""
    batch_size = 2
    x = torch.randn(batch_size, encoder.input_dim)
    
    mu, logvar = encoder.run_convolutional_layers(x)
    
    assert mu.shape == (batch_size, encoder.bottleneck_dim)
    assert logvar.shape == (batch_size, encoder.bottleneck_dim)
    
    # Check logvar clamping (max is 4.0, min is -10.0 in your code)
    assert torch.all(logvar <= 4.0)
    assert torch.all(logvar >= -10.0)

def test_run_convolutional_layers_3d_input(encoder):
    """Test the convolutional stack with [Batch, Seq_Len, Features] input."""
    batch_size = 2
    seq_len = 15
    x = torch.randn(batch_size, seq_len, encoder.input_dim)
    
    mu, logvar = encoder.run_convolutional_layers(x)
    
    # AdaptiveAvgPool1d ensures the sequence dimension is pooled to 1, then squeezed
    assert mu.shape == (batch_size, encoder.bottleneck_dim)
    assert logvar.shape == (batch_size, encoder.bottleneck_dim)

def test_encoder_forward_no_vae_out(encoder):
    """Test the full forward pass when generating defaults for missing state."""
    batch_size = 2
    x = torch.randn(batch_size, encoder.input_dim)
    
    out = encoder(x, vae_out=None)
    
    # Use hasattr to check for NamedTuple attributes instead of 'in'
    assert hasattr(out, "z")
    assert hasattr(out, "mu_z")
    assert hasattr(out, "logvar_z")
    assert hasattr(out, "pi")
    assert hasattr(out, "log_pi")
    assert hasattr(out, "alpha")
    
    # Use dot notation for accessing values
    assert out.z.shape == (batch_size, encoder.latent_dim)
    assert out.pi.shape == (batch_size, encoder.k_atoms)

def test_encoder_forward_with_vae_out(encoder):
    """Test the forward pass when previous state (vae_out) is provided."""
    batch_size = 2
    x = torch.randn(batch_size, encoder.input_dim)
    
    # Note: mock_vae_out is still a dict because your forward pass expects 
    # the input vae_out to be a dict (or at least acts like one with vae_out['recon']).
    mock_vae_out = {
        "recon": torch.randn_like(x),
        "alpha": torch.randn(batch_size, encoder.k_atoms),
        "alpha_mu": torch.randn(batch_size, encoder.k_atoms),
        "alpha_factor": torch.randn(batch_size, encoder.k_atoms, encoder.rank),
        "alpha_diag": torch.ones(batch_size, encoder.k_atoms),
        "pi": torch.softmax(torch.randn(batch_size, encoder.k_atoms), dim=-1),
        "bottleneck": torch.randn(batch_size, encoder.bottleneck_dim)
    }
    
    out = encoder(x, vae_out=mock_vae_out)
    
    assert hasattr(out, "z")
    assert out.z.shape == (batch_size, encoder.latent_dim)
    
    # Use dot notation for the output, but keep bracket notation for the mock dict!
    assert torch.allclose(out.pi, mock_vae_out["pi"])

def test_encoder_fusion_layer_dimensions(encoder):
    """Ensure the inputs to the fusion layer sum up to the expected 158 dims."""
    batch_size = 2
    
    # conv_bottleneck (64) + log_pi (k_atoms=30) + prev_bottleneck (bottleneck_dim=64) = 158
    conv_bottleneck = torch.randn(batch_size, 64)
    log_pi = torch.randn(batch_size, 30)
    prev_bottleneck = torch.randn(batch_size, 64)
    
    concat_input = torch.cat([conv_bottleneck, log_pi, prev_bottleneck], dim=-1)
    
    assert concat_input.shape[-1] == 158
    
    fusion_out = encoder.fusion_net(concat_input)
    assert fusion_out.shape == (batch_size, 64)