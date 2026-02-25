import pytest
import torch
from linear_operator.operators import KeOpsLinearOperator
from deepkernels.kernels.keops import GenerativeKernel

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def num_latents():
    return 8

@pytest.fixture
def kernel(batch_size, num_latents):
    """Initializes the kernel with the LMC batch shape."""
    batch_shape = torch.Size([num_latents])
    return GenerativeKernel(batch_shape=batch_shape)

@pytest.fixture
def mock_hyperparams(batch_size, num_latents):
    """
    Creates mock outputs mimicking the VAE Decoder's parameter projections.
    All tensors require gradients to test backpropagation.
    """
    Q = 4 # Number of spectral mixture components
    
    def make_param(*shape):
        return torch.rand(*shape, requires_grad=True)

    # Base shapes: [Batch, num_latents]
    return {
        'ls_rbf': make_param(batch_size, num_latents),
        'ls_per': make_param(batch_size, num_latents),
        'p_per':  make_param(batch_size, num_latents),
        'ls_mat': make_param(batch_size, num_latents),
        
        # Spectral Mixture shapes: [Batch, num_latents, Q]
        'w_sm':   make_param(batch_size, num_latents, Q),
        'mu_sm':  make_param(batch_size, num_latents, Q),
        'v_sm':   make_param(batch_size, num_latents, Q),
        
        # Gates shape: [Batch, num_latents, 16] (Since kernels_out = 16)
        'gates':  torch.softmax(make_param(batch_size, num_latents, 16), dim=-1)
    }

# --- Tests ---

def test_kernel_initialization(kernel, num_latents):
    """Test that the batch_shape applied correctly to parameters."""
    assert kernel.batch_shape == torch.Size([num_latents])
    assert kernel.raw_outputscale.shape == torch.Size([num_latents])
    assert kernel.kernels_out == 16

def test_kernel_full_covariance(kernel, mock_hyperparams, batch_size, num_latents):
    """Test the dense N x M KeOps matrix construction."""
    N, M = 10, 5
    D = 3 # Feature dimension
    
    x1 = torch.randn(batch_size, num_latents, N, D)
    x2 = torch.randn(batch_size, num_latents, M, D)
    
    covar = kernel(x1, x2, diag=False, gp_params=mock_hyperparams)
    
    # 1. Output must be a KeOps lazy operator to save memory
    assert isinstance(covar, KeOpsLinearOperator)
    
    # 2. Shape must be [Batch, num_latents, N, M]
    assert covar.size() == torch.Size([batch_size, num_latents, N, M])

def test_kernel_diag_fallback(kernel, mock_hyperparams, batch_size, num_latents):
    """Test the diag=True fallback (used for marginal variance)."""
    N = 10
    D = 3
    
    # For diagonal, x1 and x2 are the same
    x1 = torch.randn(batch_size, num_latents, N, D)
    
    covar_diag = kernel(x1, x1, diag=True, gp_params=mock_hyperparams)
    
    # 1. Output must be a standard materialized tensor, NOT KeOps
    assert isinstance(covar_diag, torch.Tensor)
    
    # 2. Shape must be [Batch, num_latents, N] 
    assert covar_diag.shape == torch.Size([batch_size, num_latents, N])

def test_kernel_gradient_flow(kernel, mock_hyperparams, batch_size, num_latents):
    """Ensure backpropagation flows through the combinatorial tree to the VAE."""
    N = 4
    D = 3
    x1 = torch.randn(batch_size, num_latents, N, D)
    
    # Run forward pass (dense)
    covar = kernel(x1, x1, diag=False, gp_params=mock_hyperparams)
    
    # Create a dummy loss (e.g., sum of all kernel elements)
    # Note: covar is a KeOpsLinearOperator, so we must evaluate it to a dense tensor for the dummy loss
    loss = covar.to_dense().sum()
    loss.backward()
    
    # 1. Check that the kernel's native parameters received gradients
    assert kernel.raw_outputscale.grad is not None
    
    # 2. Check that the VAE's predicted hyperparameters received gradients
    assert mock_hyperparams['ls_rbf'].grad is not None
    assert mock_hyperparams['w_sm'].grad is not None
    assert mock_hyperparams['gates'].grad is not None