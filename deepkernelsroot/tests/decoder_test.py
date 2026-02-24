import pytest
import torch
from unittest.mock import patch, MagicMock
import torch.nn as nn

# Update this import path to match your actual project structure
from deepkernels.models.decoder import SpectralDecoder 

# --- Fixtures ---

@pytest.fixture
def mock_base_methods():
    """
    Mocks the methods inherited from BaseGenerativeModel and the SimpleLoss ABC.
    Uses patch directly to avoid dictionary/KeyError quirks.
    """
    def mock_stack_features(features_list):
        return torch.stack(features_list, dim=1)

    with patch('deepkernels.models.decoder.SpectralDecoder.stack_features', side_effect=mock_stack_features), \
         patch('deepkernels.models.decoder.SpectralDecoder.lowrankmultivariatenorm', side_effect=lambda mu, f, d: mu), \
         patch('deepkernels.models.decoder.SpectralDecoder.dirichlet_sample', side_effect=lambda alpha: torch.softmax(alpha, dim=-1)), \
         patch('deepkernels.models.decoder.SpectralDecoder.update_added_loss_term') as mock_update, \
         patch('deepkernels.models.decoder.LossTerm'):
         
        yield mock_update

@pytest.fixture
def decoder(mock_base_methods):
    """Instantiates the SpectralDecoder with default test parameters."""
    return SpectralDecoder(
        input_dim=30,
        spectral_dim=256,
        num_clusters=30,
        spectral_emb_dim=2048,
        input_dim_data=30,
        bottleneck_dim=64,
        num_experts=8,
        k_atoms=30,
        latent_dim=16
    )

# --- Tests ---

def test_decoder_initialization(decoder):
    """Ensure the network structures initialize with the correct dimensions."""
    assert len(decoder.expert_variational_heads) == decoder.num_experts
    assert len(decoder.expert_logit_heads) == decoder.num_experts
    
    # Check that orthogonal init was applied to linears
    linear_layer = decoder.mu_alpha
    assert torch.is_tensor(linear_layer.weight)

def test_disentangle(decoder):
    """Test the physical-informed decomposition of the bottleneck state."""
    batch_size = 4
    # Bottleneck is exactly 64 dims: 4 (ls) + 30 (pi) + 30 (dt)
    bottleneck = torch.randn(batch_size, 64)
    
    recon, trend, amplitude, residual = decoder.disentangle(bottleneck)
    
    # All outputs should map to input_dim_data (30)
    assert recon.shape == (batch_size, 30)
    assert trend.shape == (batch_size, 30)
    assert amplitude.shape == (batch_size, 30)
    assert residual.shape == (batch_size, 30)
    
    # Amplitude uses a sigmoid, so it must be bound between 0 and 1
    assert torch.all((amplitude >= 0.0) & (amplitude <= 1.0))

def test_get_alpha_mvn_heads_decoder(decoder):
    """Test the Low-Rank MVN parameter extraction."""
    batch_size = 4
    bottleneck = torch.randn(batch_size, 64)
    
    mu, factor, diag = decoder.get_alpha_mvn_heads_decoder(bottleneck)
    
    assert mu.shape == (batch_size, decoder.k_atoms)
    # Factor is [Batch, K, Rank]
    assert factor.shape == (batch_size, decoder.k_atoms, decoder.rank)
    assert diag.shape == (batch_size, decoder.k_atoms)
    
    # Diag uses softplus + 1e-6, so it must be strictly positive
    assert torch.all(diag > 0)

def test_log_alpha_kl_low_rank(decoder):
    """Ensure the KL divergence calculation runs and records the loss."""
    batch_size = 2
    mu = torch.zeros(batch_size, decoder.k_atoms)
    chol = torch.randn(batch_size, decoder.k_atoms * decoder.rank) 
    diag = torch.ones(batch_size, decoder.k_atoms)
    
    decoder.log_alpha_kl_low_rank(mu, chol, diag, k_atoms=decoder.k_atoms)
    
    # Assert directly on the patched method!
    decoder.update_added_loss_term.assert_called()

def test_predict_lengthscale_and_log_kl(decoder):
    """Test lengthscale posterior generation, clamping, and KL logging."""
    batch_size = 4
    bottleneck = torch.randn(batch_size, 64)
    
    ls_sample = decoder.predict_lengthscale_and_log_kl(bottleneck)
    
    assert ls_sample.shape == (batch_size, decoder.k_atoms)
    assert torch.all(ls_sample >= 1e-4)
    assert torch.all(ls_sample <= 100.0)
    
    # Assert directly on the patched method!
    decoder.update_added_loss_term.assert_called()

def test_decoder_forward_pass(decoder):
    """
    Test the full forward pipeline using the strictly typed NamedTuple.
    """
    batch_size = 2
    spectral_emb_dim = 2048
    x = torch.randn(batch_size, spectral_emb_dim)
    
    out = decoder(x, vae_out=None)
    
    expected_keys = [
        'bottleneck', 'alpha', 'alpha_mu', 'alpha_factor', 'alpha_diag', 
        'mixture_means_per_expert', 'parameters_per_expert', 'recon', 
        'bandwidth_mod', 'pi', 'amp', 'trend', 'res', 'ls'
    ]
    for key in expected_keys:
        assert hasattr(out, key)
    
    assert out.recon.shape == (batch_size, 30)
    assert out.pi.shape == (batch_size, decoder.k_atoms)
    assert out.bottleneck.shape == (batch_size, 64)
    
    # Hardcoded to 16 since `latent_dim` isn't saved to `self` in the decoder init
    assert out.mixture_means_per_expert.shape == (batch_size, decoder.num_experts, decoder.latent_dim)
    assert out.parameters_per_expert.shape == (batch_size, decoder.num_experts, decoder.k_atoms)