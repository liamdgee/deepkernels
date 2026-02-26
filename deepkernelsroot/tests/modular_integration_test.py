import pytest
import torch
import gpytorch

from deepkernels.models.model import StateSpaceKernelProcess

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def seq_len():
    return 64

@pytest.fixture
def input_dim():
    return 30

def test_end_to_end_orchestrator(batch_size, seq_len, input_dim):
    """
    Tests the complete pipeline from raw data -> VAE -> GP -> Loss -> Backprop
    """
    # 1. Initialize the grand orchestrator
    model = StateSpaceKernelProcess()
    model.train() # Put into training mode
    
    # 2. Create dummy raw physical signal data
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    
    # 3. Forward Pass: The Assembly Line
    current_state, gp_output, gp_input, history = model(x)
    
    # --- SHAPE ASSERTIONS ---
    # Ensure GP input features stacked perfectly: [Batch, Experts, SeqLen, Dim]
    assert gp_input.dim() == 4 
    assert gp_input.size(2) == seq_len 
    
    # Ensure GP returned a valid GPyTorch MultivariateNormal distribution
    assert isinstance(gp_output, gpytorch.distributions.MultivariateNormal)
    
    # 4. The Loss Function (Dummy target for Marginal Log Likelihood)
    # The GP output matches the shape of the inputs, so we create a dummy target of the same shape.
    target_shape = gp_output.event_shape
    dummy_target = torch.randn(*target_shape)
    
    # Calculate a simple loss (e.g., negative log probability of the dummy target)
    # Note: In your real training loop, you'll use gpytorch.mlls.ExactMarginalLogLikelihood
    loss = -gp_output.log_prob(dummy_target).sum()
    
    # 5. Backward Pass: The Gradient Flow Test
    loss.backward()
    
    # --- GRADIENT ASSERTIONS ---
    # Prove that the GP learned
    assert model.gp.covar_module.raw_outputscale.grad is not None
    
    # Prove that gradients flowed all the way backward into the VAE Encoder!
    # (Checking the very first layer of the network ensures the chain is unbroken)
    encoder_first_layer_weights = list(model.vae.encoder.parameters())[0]
    assert encoder_first_layer_weights.grad is not None

    print("\nEnd-to-End Execution Successful! Gradients flowed perfectly.")