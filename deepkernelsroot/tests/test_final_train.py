import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock
import gpytorch
# Import your real classes
from deepkernels.models.model import StateSpaceKernelProcess
from deepkernels.models.exactgp import Simple
from deepkernels.train.langevin_trainer import LangevinTrainer

@pytest.fixture
def tiny_dataset():
    """Creates a miniature dataset to bypass VRAM limits during testing."""
    N = 128
    seq_len = 32
    features = 30
    k_atoms = 30
    
    # Create x_data with a slight trend so points are distinct in space
    # This prevents the "Not Positive Definite" error
    x_data = torch.randn(N, seq_len, features) + torch.linspace(0, 1, N).view(N, 1, 1)
    
    y_data = torch.randn(N, k_atoms)
    dataset = TensorDataset(x_data, y_data)
    
    # Ensure the DataLoader doesn't drop the last batch if it's smaller
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    full_x = x_data
    full_y = y_data.t().contiguous() 
    
    return train_loader, full_x, full_y, N

@pytest.fixture
def mock_trainer(tiny_dataset):
    """Initializes the real trainer with tiny dimensions."""
    train_loader, full_x, full_y, N = tiny_dataset
    
    gp = Simple(train_x=full_x, train_y=full_y, num_latents=8)
    
    model = StateSpaceKernelProcess(gp=gp)
    
    trainer = LangevinTrainer(
        model=model,
        device='cpu', 
        n_data=N,
        langevin_temp=1e-5
    )
    return trainer

def test_parameter_routing(mock_trainer):
    """Ensures the orchestrator correctly separated the parameters."""
    assert len(mock_trainer.adam_params) > 0, "GP Adam optimizer is empty!"
    assert len(mock_trainer.adamw_params) > 0, "VAE AdamW optimizer is empty!"
    assert len(mock_trainer.langevin_params) > 0, "Dirichlet SGLD optimizer is empty!"

def test_vae_step_forward_backward(mock_trainer, tiny_dataset):
    """Tests if Stage 1 (VAE Mini-batch) can execute a full backward pass."""
    train_loader, _, _, _ = tiny_dataset
    x_batch, y_batch = next(iter(train_loader))
    
    # Freeze GP, Unfreeze VAE
    mock_trainer.orchestrator.train_vae_and_dirichlet()
    
    # Run a step
    metrics = mock_trainer.step_vae(x_batch)
    
    assert "loss_recon" in metrics
    assert "loss_total" in metrics
    assert isinstance(metrics["loss_total"], float)

def test_gp_step_forward_backward(mock_trainer, tiny_dataset):
    """Tests if Stage 2 (ExactGP Full-batch) executes the KeOps math & LMC Hack."""
    _, full_x, full_y, _ = tiny_dataset
    
    # --- Move everything inside the block ---
    with gpytorch.settings.cholesky_jitter(1e-3), \
         gpytorch.settings.max_preconditioner_size(0), \
         gpytorch.settings.fast_computations(covar_root_decomposition=False):
                
        mock_trainer.orchestrator.train_gp_only()
        
        # This is where the actual matrix inversion happens! 
        # It MUST be inside the jitter context.
        metrics = mock_trainer.step_gp(full_x, full_y)
    
    assert "loss_gp" in metrics


@patch('deepkernels.train.langevin_trainer.mlflow')
def test_full_pipeline_fast_run(mock_mlflow, mock_trainer, tiny_dataset):
    train_loader, full_x, full_y, _ = tiny_dataset
    
    # ✅ Wrap the whole fit process in stability settings
    with gpytorch.settings.cholesky_jitter(1e-3), \
         gpytorch.settings.max_preconditioner_size(0), \
         gpytorch.settings.fast_computations(covar_root_decomposition=False):
        try:
            mock_trainer.fit(
                train_loader=train_loader,
                full_x=full_x,
                full_y=full_y,
                warmup_vae_epochs=1,
                vae_epochs=1,
                warmup_gp_epochs=1,
                gp_epochs=1,
                em_macro_cycles=1,
                e_epochs_per_cycle=1,
                m_epochs_per_cycle=1,
                joint_epochs=1 
            )
        except Exception as e:
            pytest.fail(f"Pipeline crashed during execution: {str(e)}")