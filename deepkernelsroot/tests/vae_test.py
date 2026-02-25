import pytest
import torch
from unittest.mock import MagicMock, patch
from typing import NamedTuple

# Update this import path to match your project structure
from deepkernels.models.spectral_vae import SpectralVAE, StateSpaceOutput, HistoryOutput, DecoderSpaceOutput

# --- Dummy NamedTuples to simulate module outputs ---
class DummyEncoderOut(NamedTuple):
    alpha: torch.Tensor
    z: torch.Tensor
    ls: torch.Tensor

class DummyDirichletOut(NamedTuple):
    predicted_lengthscale: torch.Tensor
    ls_logvar: torch.Tensor
    features: torch.Tensor
    pi: torch.Tensor
    omega: torch.Tensor
    gated_weights: torch.Tensor

class DummyDecoderOut(NamedTuple):
    recon: torch.Tensor
    bottleneck: torch.Tensor
    parameters_per_expert: torch.Tensor
    mixture_means_per_expert: torch.Tensor
    trend: torch.Tensor
    ls: torch.Tensor
    bandwidth_mod: torch.Tensor

# --- Fixtures ---

@pytest.fixture
def mock_submodules():
    """Mocks the submodules so we can test the loop logic in isolation."""
    batch_size, k_atoms, latent_dim = 2, 30, 16
    
    # Pre-build dummy tensors to return
    enc_out = DummyEncoderOut(
        alpha=torch.ones(batch_size, k_atoms),
        z=torch.zeros(batch_size, latent_dim),
        ls=torch.empty(0) # Test the empty tensor logic!
    )
    
    dir_out = DummyDirichletOut(
        predicted_lengthscale=torch.ones(batch_size, k_atoms),
        ls_logvar=torch.zeros(batch_size, k_atoms),
        features=torch.zeros(batch_size, k_atoms, 2),
        pi=torch.ones(batch_size, k_atoms) / k_atoms,
        omega=torch.zeros(batch_size, k_atoms),
        gated_weights=torch.ones(batch_size, 8)
    )
    
    dec_out = DummyDecoderOut(
        recon=torch.zeros(batch_size, 10), # Assuming input feature dim is 10
        bottleneck=torch.zeros(batch_size, 64),
        parameters_per_expert=torch.zeros(batch_size, 8, k_atoms),
        mixture_means_per_expert=torch.zeros(batch_size, 8, latent_dim),
        trend=torch.zeros(batch_size, 10),
        ls=torch.ones(batch_size, k_atoms),
        bandwidth_mod=torch.ones(batch_size, k_atoms)
    )

    with patch('deepkernels.models.spectral_vae.ConvolutionalLoopEncoder') as MockEnc, \
         patch('deepkernels.models.spectral_vae.AmortisedDirichlet') as MockDir, \
         patch('deepkernels.models.spectral_vae.SpectralDecoder') as MockDec, \
         patch('deepkernels.models.spectral_vae.SpectralVAE.dirichlet_sample', return_value=dir_out.pi):
        
        # Configure the mocked instances to return our NamedTuples
        MockEnc.return_value.return_value = enc_out
        MockDir.return_value.return_value = dir_out
        MockDec.return_value.return_value = dec_out
        
        yield MockEnc, MockDir, MockDec

@pytest.fixture
def vae_model(mock_submodules):
    """Instantiates the SpectralVAE with mocked internal organs."""
    return SpectralVAE()

# --- Tests ---

def test_spectral_vae_forward_shapes(vae_model):
    """Tests if the loop correctly executes and stacks history into [Batch, SeqLen, ...]"""
    batch_size, seq_len, features = 2, 5, 10
    x = torch.randn(batch_size, seq_len, features)
    
    out = vae_model(x, vae_out=None, steps=1)
    
    # 1. Check Output Types
    assert isinstance(out, StateSpaceOutput)
    assert isinstance(out.history, HistoryOutput)
    assert isinstance(out.current_state, DummyDecoderOut)
    
    # 2. Check Stacking Shapes (Should match seq_len, NOT seq_len * steps)
    assert out.history.recons.shape == (batch_size, seq_len, features)
    assert out.history.latents.shape == (batch_size, seq_len, 16)
    assert out.history.pis.shape == (batch_size, seq_len, 30)
    assert out.history.bottlenecks.shape == (batch_size, seq_len, 64)

def test_spectral_vae_refinement_steps(vae_model, mock_submodules):
    """Ensures the inner refinement loop runs the correct number of times."""
    batch_size, seq_len, features = 2, 4, 10
    x = torch.randn(batch_size, seq_len, features)
    steps = 3
    
    MockEnc, MockDir, MockDec = mock_submodules
    
    out = vae_model(x, vae_out=None, steps=steps)
    
    # If seq_len=4 and steps=3, the submodules should be called exactly 12 times
    expected_calls = seq_len * steps
    assert MockEnc.return_value.call_count == expected_calls
    assert MockDir.return_value.call_count == expected_calls
    assert MockDec.return_value.call_count == expected_calls
    
    # History should still only have a sequence length of 4 (one save per timestep)
    assert out.history.recons.shape[1] == seq_len

def test_spectral_vae_generative_mode(vae_model, mock_submodules):
    """Tests if generative_mode routes the previous reconstruction as the new input."""
    batch_size, seq_len, features = 2, 3, 10
    x = torch.randn(batch_size, seq_len, features)
    
    MockEnc, _, _ = mock_submodules
    
    vae_model(x, vae_out=None, steps=1, generative_mode=True)
    
    # Grab the arguments passed to the encoder on the final timestep
    final_call_args = MockEnc.return_value.call_args[0]
    x_input_to_encoder = final_call_args[0]
    
    # Because generative_mode=True, the final input should be generated by the mock decoder 
    # (which outputs zeros in our fixture), NOT the original random `x`.
    assert torch.all(x_input_to_encoder == 0)
