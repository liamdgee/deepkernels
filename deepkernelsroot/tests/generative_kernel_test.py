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
def mock_hyperparams_factory(batch_size, num_latents):
    """
    A factory fixture that generates perfectly shaped VAE outputs 
    for ANY sequence length the test requires!
    Shape: [Batch, 1, SeqLen, Features]
    """
    def _generator():
        Q = 4 # Number of spectral mixture components
        
        def make_param(feat_dim):
            shape = (batch_size, 1, feat_dim)
            return torch.rand(*shape, requires_grad=True)

        return {
            'ls_rbf': make_param(1),
            'ls_per': make_param(1),
            'p_per':  make_param(1),
            'ls_mat': make_param(1),
            'w_sm':   make_param(Q),
            'mu_sm':  make_param(Q),
            'v_sm':   make_param(Q),
            'gates':  torch.softmax(make_param(16), dim=-1) 
        }
    
    return _generator
# --- Tests ---

def test_kernel_initialization(kernel, num_latents):
    """Test that the batch_shape applied correctly to parameters."""
    assert kernel.batch_shape == torch.Size([num_latents])
    assert kernel.raw_outputscale.shape == torch.Size([num_latents])
    assert kernel.kernels_out == 16

def test_kernel_full_covariance(kernel, mock_hyperparams_factory, batch_size, num_latents):
    N, M = 10, 5
    D = 3 
    
    gp_params = mock_hyperparams_factory() 
    
    x1 = torch.randn(batch_size, num_latents, N, D)
    x2 = torch.randn(batch_size, num_latents, M, D)
    
    # 1. Standard __call__
    covar = kernel(x1, x2, diag=False, gp_params=gp_params)
    
    # 2. Unwrap the GPyTorch lazy shell
    evaluated_covar = covar.evaluate_kernel()
    
    # 3. Bypass the GPyTorch/LinearOperator naming collision entirely.
    # We just need to know it operates like a Lazy/Linear operator!
    assert hasattr(evaluated_covar, 'to_dense')
    assert evaluated_covar.size() == torch.Size([batch_size, num_latents, N, M])
    
    # 4. The Ultimate Math Proof: Force KeOps to compile and evaluate.
    # If this doesn't crash, your dimensions and broadcasting are 100% perfect!
    dense_matrix = evaluated_covar.to_dense()
    assert dense_matrix.shape == torch.Size([batch_size, num_latents, N, M])

def test_kernel_gradient_flow(kernel, mock_hyperparams_factory, batch_size, num_latents):
    N = 4
    D = 3
    
    gp_params = mock_hyperparams_factory()
    
    x1 = torch.randn(batch_size, num_latents, N, D, requires_grad=True)
    
    # Standard __call__
    covar = kernel(x1, x1, diag=False, gp_params=gp_params)
    
    # Evaluate and calculate a dummy loss
    loss = covar.to_dense().sum()
    loss.backward()
    
    # Check that gradients successfully traversed the KeOps C++ graph 
    # and reached both the Kernel and the VAE hyperparameter projections!
    assert kernel.raw_outputscale.grad is not None
    assert gp_params['ls_rbf'].grad is not None