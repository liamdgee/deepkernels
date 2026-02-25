import pytest
import torch
from deepkernels.models.variationalautoencoder import SpectralVAE, StateSpaceOutput
from deepkernels.models.NKN import GPParams

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 4

@pytest.fixture
def input_dim():
    return 30  # Assuming your raw data features are 30

@pytest.fixture
def model():
    # Instantiate the model. 
    # (Assuming default dimensions in your sub-modules map correctly to input_dim=30)
    vae = SpectralVAE()
    vae.eval() # Set to eval mode for deterministic testing
    return vae

@pytest.fixture
def dummy_input(batch_size, seq_len, input_dim):
    # [Batch, SeqLen, Features]
    return torch.randn(batch_size, seq_len, input_dim)


def test_spectral_vae_initialization(model):
    """Ensure the VAE and its sub-modules initialize correctly."""
    assert hasattr(model, 'encoder'), "Missing Encoder"
    assert hasattr(model, 'dirichlet'), "Missing Dirichlet"
    assert hasattr(model, 'decoder'), "Missing Decoder"

def test_spectral_vae_forward_pass_types(model, dummy_input):
    """Test that the forward pass returns the correct NamedTuples without crashing."""
    out = model(dummy_input, steps=2)
    
    # Check top-level output type
    assert isinstance(out, StateSpaceOutput), f"Expected StateSpaceOutput, got {type(out)}"
    
    # Check nested output types
    assert hasattr(out, 'state'), "Missing 'state' in output"
    assert hasattr(out, 'history'), "Missing 'history' in output"
    assert isinstance(out.history.gp_params, GPParams), "gp_params not correctly repacked into GPParams!"

def test_spectral_vae_temporal_stacking_shapes(model, dummy_input, batch_size, seq_len, input_dim):
    """
    CRITICAL TEST: Ensure the loop correctly stacks over the sequence dimension (dim=1).
    If this fails, the GP will receive mashed sequence steps.
    """
    out = model(dummy_input, steps=2)
    history = out.history
    
    # 1. Check physical reconstruction shapes [Batch, SeqLen, Features]
    assert history.recons.shape == (batch_size, seq_len, input_dim), \
        f"Recons shape mismatch. Expected {(batch_size, seq_len, input_dim)}, got {history.recons.shape}"
    
    # 2. Check latent routing shapes (Assuming 30 k_atoms)
    assert history.pis.shape[:2] == (batch_size, seq_len), "Routing probabilities missing temporal dim"
    
    # 3. Check GP Features (Assuming 8 experts, 16 latent dim)
    assert history.gp_features.shape[:2] == (batch_size, seq_len), "GP Features missing temporal dim"

def test_gp_params_namedtuple_repacking(model, dummy_input, batch_size, seq_len):
    """
    Tests the bugfix we made: ensure all KeOps GP params were extracted 
    from the list of tuples and stacked correctly into tensors.
    """
    out = model(dummy_input, steps=1)
    gp_params = out.history.gp_params
    
    # Every single tensor inside GPParams should have the shape [Batch, SeqLen, ...]
    assert gp_params.ls_rbf.shape[:2] == (batch_size, seq_len), "RBF Lengthscale missing temporal dim"
    assert gp_params.gates.shape[:2] == (batch_size, seq_len), "Gates missing temporal dim"
    assert gp_params.w_sm.shape[:2] == (batch_size, seq_len), "Spectral Mixture weights missing temporal dim"

def test_generative_mode_autoregression(model, batch_size, seq_len, input_dim):
    """
    Tests the generative mode feedback loop where x_t = current_state.recon
    """
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Run with generative_mode=True
    current_state, history = model.refinement_loop(
        x=dummy_input, 
        steps=2, 
        generative_mode=True
    )
    
    # In generative mode, it should still produce valid stacked histories
    assert history.recons.shape == (batch_size, seq_len, input_dim)
    assert not torch.isnan(history.recons).any(), "Generative loop produced NaNs!"