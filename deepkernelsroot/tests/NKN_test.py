import pytest
import torch
from deepkernels.models.kernel_network import KernelNetwork # Update path if needed

@pytest.fixture
def bottleneck_dim():
    return 64

@pytest.fixture
def network(bottleneck_dim):
    return KernelNetwork(
        bottleneck_dim=bottleneck_dim,
        spectral_emb_dim=2048,
        gp_dim=1,
        spectral_micro_mixtures=4
    )

def test_nkn_2d_input(network, bottleneck_dim):
    """Test standard 2D batch inputs [Batch, Features]"""
    batch_size = 16
    x = torch.randn(batch_size, bottleneck_dim)
    
    out = network(x)
    
    # Check Dirichlet feature dimension
    assert out['features_large'].shape == (batch_size, 2048)
    
    # Check KeOps GP parameter formatting
    gp_params = out['gp_params']
    assert gp_params['gates'].shape == (batch_size, 1, 1, 16)
    assert gp_params['ls_rbf'].shape == (batch_size, 1, 1, 1)
    assert gp_params['w_sm'].shape == (batch_size, 1, 1, 4)

def test_nkn_3d_input(network, bottleneck_dim):
    """Test temporal/expert 3D batch inputs [Batch, SeqLen, Features]"""
    batch_size = 8
    seq_len = 10
    x = torch.randn(batch_size, seq_len, bottleneck_dim)
    
    out = network(x)
    
    # If the dynamic routing works, this should output [Batch, SeqLen, ...] safely
    assert out['features_large'].shape == (batch_size, seq_len, 2048)
    
    gp_params = out['gp_params']
    # Ensures .view() didn't crush the sequence dimension!
    assert gp_params['gates'].shape == (batch_size, seq_len, 1, 1, 16)
    assert gp_params['ls_rbf'].shape == (batch_size, seq_len, 1, 1, 1)

def test_nkn_positivity_constraint(network, bottleneck_dim):
    """Ensure all KeOps GP params are strictly positive (lengthscales/variances)"""
    x = torch.randn(4, bottleneck_dim)
    out = network(x)
    gp_params = out['gp_params']
    
    for key, val in gp_params.items():
        if key != 'gates': # Gates are softplus too, but checking core params
            assert torch.all(val > 0), f"Parameter {key} must be strictly positive!"

def test_nkn_combinatorics_logic(network, bottleneck_dim):
    """Test the log-space einsum interaction logic specifically"""
    x = torch.randn(2, bottleneck_dim)
    lin, per, rbf, rat = network.compute_primitives(x)
    
    # Should output [Batch, 128] compressed interactions
    interactions = network.compute_kernel_interactions(lin, per, rbf, rat)
    assert interactions.shape == (2, 128)
    
    # Ensure no NaNs from the log-space transform
    assert not torch.isnan(interactions).any(), "Log-space product resulted in NaNs!"