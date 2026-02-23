import pytest
import torch
import math
from typing import NamedTuple

from deepkernels.models.dirichlet import AmortisedDirichlet, HDPConfig

def test_amortised_dirichlet_forward_and_backward():
    # --- 1. Setup Dimensions ---
    batch_size = 4
    k_atoms = 30
    latent_dim = 16
    fourier_dim = 128
    
    # --- 2. Initialize Model ---
    # (Assuming the inheritance fix is applied)
    model = AmortisedDirichlet(
        k_atoms=k_atoms, 
        fourier_dim=fourier_dim, 
        latent_dim=latent_dim
    )
    
    # --- 3. Mock the VAE Inputs ---
    # This perfectly mimics what your ResNetFeedbackEncoder will output
    x_in = torch.randn(batch_size, latent_dim)
    
    mock_vae_out = {
        'pi': torch.full((batch_size, k_atoms), 1.0 / k_atoms), 
        'mu_alpha': torch.randn(batch_size, k_atoms),
        'factor_alpha': torch.randn(batch_size, k_atoms, 3),
        'diag_alpha': torch.ones(batch_size, k_atoms),
        'ls': torch.rand(batch_size, k_atoms) * 2.0
    }
    # --- 4. Forward Pass Test ---
    # Wrap in try-except to catch broadcasting or dictionary errors cleanly
    #try:
    out = model(x=x_in, vae_out=mock_vae_out)
    #except Exception as e:
    #    pytest.fail(f"Forward pass crashed with exception: {e}")
        
    # --- 5. Shape and NaNs Assertions ---
    assert isinstance(out, tuple), "Model must return a namedTuple"
    
    # Check the Pi Simplex
    pi = out.pi
    assert pi.shape == (batch_size, k_atoms), f"Expected pi shape {(batch_size, k_atoms)}, got {pi.shape}"
    assert not torch.isnan(pi).any(), "NaNs detected in the Dirichlet pi simplex!"
    assert torch.allclose(pi.sum(dim=-1), torch.ones(batch_size)), "Pi simplex does not sum to 1.0!"
    
    # Check the Frequencies (Omega)
    omega = out.frequencies
    assert omega.shape == (batch_size, k_atoms, fourier_dim, latent_dim), "Omega broadcasting failed!"
    
    # Check Gated Features
    features = out.features
    assert features.dim() == 2, "Gated features should be flattened for the downstream network"
    assert not torch.isnan(features).any(), "NaNs detected in gated features!"

    # --- 6. Backward Pass Test (The Moment of Truth) ---
    # We create a dummy loss by summing the gated features to ensure gradients flow 
    # all the way backward through the rsample() and stick-breaking logic.
    dummy_loss = features.sum() + pi.sum()
    
    try:
        dummy_loss.backward()
    except Exception as e:
        pytest.fail(f"Backward pass crashed. PyTorch rsample gradients failed: {e}")
        
    # Verify gradients populated on key generative parameters
    assert model.raw_gamma.grad is not None, "Gradients failed to reach raw_gamma (Global Stick Breaking broken)"
    assert model.h_log_sigma.grad is not None, "Gradients failed to reach lengthscales"
    # Grab the first raw parameter dynamically to bypass Spectral Norm wrappers
    nkn_param = next(model.kernel_network.parameters())
    assert nkn_param.grad is not None, "Gradients failed to reach the internal KernelNetwork"

    print("All AmortisedDirichlet tests passed successfully!")