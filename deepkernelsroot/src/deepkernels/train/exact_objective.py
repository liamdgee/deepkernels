import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F
from typing import NamedTuple
import logging
from typing import Union, Optional
from tqdm import tqdm

import linear_operator
from linear_operator import DenseLinearOperator, MatmulLinearOperator, 

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
        self.kl_weights = kl_weights or {
            "lengthscale_kl": 1.0,
            "alpha_kl": 1.0,
            "global_divergence": 0.1,
            "local_divergence": 0.1
        }
    
    def forward(self, model, x_target, ss_history, gp_output, gp_target):
        device = self.get_device()
        kl_metrics = {}
        
        if gp_output is not None and gp_target is not None:
            if hasattr(ss_history, 'lmc_matrices'):
                # --- Inside ExactObjective.forward() ---
        
        if gp_output is not None and gp_target is not None:
            if hasattr(ss_history, 'W_current_state'): # Ensure you export W!
                W_mat = ss_history.lmc_matrices
                gp_target_batched = gp_target.t().unsqueeze(-1)
                W_pinv = torch.linalg.pinv(W_mat)
                latent_target = torch.bmm(W_pinv, gp_target_batched).squeeze(-1) # -> [N, 8]
                latent_target = latent_target.t().contiguous()
                if torch.isnan(latent_target).any():
                    logger.warning("NaNs detected in batched pseudo-inverse. Falling back to zeros.")
                    latent_target = torch.nan_to_num(latent_target)
                exact_mll = self.mll(gp_output, latent_target)
                gp_loss = -exact_mll
            else:
                gp_loss = -self.mll(gp_output, gp_target)
        else:
            gp_loss = torch.tensor(0.0, device=device)

        kl_metrics["loss_gp"] = gp_loss.item()
        
        total_vae_loss = torch.tensor(0.0, device=device)
        kl_sum = torch.tensor(0.0, device=device)
        recon_loss_val = torch.tensor(0.0, device=device)

        for name, added_loss_term in model.named_added_loss_terms():
            if 'gp' in name: continue
            
            raw_loss = added_loss_term.loss().sum()

            if 'recon' in name:
                recon_loss_val = raw_loss
                total_vae_loss = total_vae_loss + recon_loss_val
            else:
                weight = self.kl_weights.get(name, 1.0)
                for key, val in self.kl_weights.items():
                    if key in name:
                        weight = val
                        break
                
                scaled_kl = weight * raw_loss
                kl_sum = kl_sum + scaled_kl
                total_vae_loss = total_vae_loss + scaled_kl
            
            kl_metrics[f'loss_{name}'] = raw_loss.item()
        
        
        total_loss = total_vae_loss + gp_loss

        
        def to_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        
        metrics = {
            'loss_total': to_item(total_loss),
            'loss_vae': to_item(total_vae_loss), 
            'loss_kls': to_item(kl_sum),
            'loss_gp': to_item(gp_loss),
            'loss_recon': to_item(recon_loss_val),
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