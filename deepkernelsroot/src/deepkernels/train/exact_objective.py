import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F
from typing import NamedTuple
import logging
from typing import Union, Optional
from tqdm import tqdm

import linear_operator
from linear_operator.operators import MatmulLinearOperator, RootLinearOperator

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExactObjective(nn.Module):
    def __init__(self, 
                 model,
                 kl_weights=None):
        """
        Args:
            gp_model: Your AcceleratedKernelGP instance.
            likelihood: The GPyTorch likelihood attached to your GP (e.g., GaussianLikelihood).
            num_data: Total number of samples in your entire training dataset (crucial for GP KL scaling).
        """
        super().__init__()
        
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=model.gp.likelihood, 
            model=model.gp
        )
        self.kl_weights = kl_weights or {}
        
    
    def forward(self, model, x_target, ss_history, gp_output, gp_target):
        """
        Args:
            model: overarching SpectralVAE model (to pull added loss terms).
            x_target: ground-truth data sequence [Batch, SeqLen, Features].
            ss_history: VAE state space history namedtuple
            gp_output: The MultivariateNormal returned by your GP.
            gp_target: The sequence the GP is supposed to be predicting.
        """
        device = self.get_device()
        kl_metrics = {}
        
        recon_loss = torch.tensor(0.0, device=device)
        kl_loss = 0.0
        total_vae_loss = 0.0
        
        for name, added_loss_term in model.named_added_loss_terms():
            if 'gp' in name:
                continue
            
            raw_loss = added_loss_term.loss()

            if 'recon' in name:
                recon_loss = raw_loss
                total_vae_loss += recon_loss
                kl_metrics[f'loss_{name}'] = recon_loss.item()
                continue
            
            weight = 1.0
            for key, annealed_val in self.kl_weights.items():
                if key in name:
                    weight = annealed_val
                    break
            
            scaled_loss = weight * raw_loss
            kl_loss += scaled_loss
            total_vae_loss += scaled_loss
            kl_metrics[f'loss_{name}'] = scaled_loss.item()
        
        # --- GP Exact Marginal Log-Likelihood ---
        exact_mll = torch.tensor(0.0, device=device)
        gp_loss = torch.tensor(0.0, device=device)
        
        if gp_output is not None and gp_target is not None:
            lmc = ss_history.lmc_matrices
            batch_lmcs = lmc.mean(dim=1)
            B_mat = batch_lmcs.mean(dim=0)
            lmc_pseudo_inv = torch.linalg.pinv(B_mat) # [8, 30]
            latent_target = torch.matmul(lmc_pseudo_inv, gp_target)
            exact_mll = self.mll(gp_output, latent_target)
            gp_loss = -exact_mll

        # --- Loss function ---
        total_loss = total_vae_loss + gp_loss

        def to_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        
        metrics = {
            'loss_total': to_item(total_loss),
            'loss_vae': to_item(total_vae_loss), 
            'loss_kls': to_item(kl_loss),
            'loss_gp': to_item(exact_mll),
            'loss_recon': to_item(recon_loss),
            **kl_metrics
        }
        
        return total_loss, metrics
    
    def get_device(self, device_request: Union[str, torch.device, None] = None) -> torch.device:

            """
            Resolves the optimal available device for PyTorch operations.
            
            Priority:
            1. explicit device_request (if provided and valid)
            2. cuda:0 (NVIDIA GPU)
            3. mps (Apple Silicon Metal Performance Shaders)
            4. cpu
            
            Args:
                device_request: Optional string ('cuda', 'mps', 'cpu') or torch.device 
                                to force a specific device.
            
            Returns:
                torch.device: The resolved device.
            """
            if device_request is not None:
                device = torch.device(device_request)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    logging.warning(f"CUDA requested but unavailable. Falling back to CPU.")
                    return torch.device('cpu')
                if device.type == 'mps' and not torch.backends.mps.is_available():
                    logging.warning(f"MPS (Apple Silicon) requested but unavailable. Falling back to CPU.")
                    return torch.device('cpu')
                return device
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device('mps')
            return torch.device('cpu')