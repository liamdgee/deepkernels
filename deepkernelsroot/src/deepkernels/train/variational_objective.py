import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F
from typing import NamedTuple, Union
import logging
#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvidenceLowerBound(nn.Module):
    def __init__(self, 
                 model,
                 device='cuda', 
                 **kwargs):
        """
        Args:
            gp_model: Your AcceleratedKernelGP instance.
            likelihood: The GPyTorch likelihood attached to your GP (e.g., GaussianLikelihood).
            num_data: Total number of samples in your entire training dataset (crucial for GP KL scaling).
        """
        super().__init__()
        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=model.gp.likelihood,
            model=model.gp,
            num_data=kwargs.get('n_data', 76674)
        )
        self.device = self.get_device(device)
        
    def _resolve_annealer_key(self, name):
        """Maps model loss term names to annealer dictionary keys."""
        if 'global' in name: return 'global_divergence'
        if 'local' in name: return 'local_divergence'
        if 'recon_kl' in name: return 'recon_kl'
        if 'alpha' in name: return 'alpha_kl'
        if 'lengthscale' in name: return 'lengthscale_kl'
        if 'inverse_wishart' in name: return 'inverse_wishart'
        return name
    
    def _get_cyclic_beta(self, global_step, total_steps, n_cycles=4, ratio=0.4):
        if global_step is None or total_steps is None:
            return 1.0
        cycle_length = total_steps // n_cycles
        step_in_cycle = global_step % cycle_length
        ramp_length = int(cycle_length * ratio)
        return min(1.0, step_in_cycle / ramp_length) if ramp_length > 0 else 1.0
    
    def forward(self, model, gp_output, gp_target, state_out, global_step=None, total_steps=None, annealers=None):
        device = gp_target.device if gp_target is not None else next(model.parameters()).device
        kl_metrics = {}
        kl_sum = 0.0
        total_recon = 0.0
        if gp_output is not None and gp_target is not None:
            target = gp_target.view(-1).contiguous()
            gp_loss = -self.mll(gp_output, target)
        else:
            gp_loss = torch.tensor(0.0, device=device)
        for name, added_loss_term in model.named_added_loss_terms():
            if 'gp' in name: continue
            
            raw_loss = added_loss_term.loss().sum()
            
            if 'recon_term' in name:
                weight = 1.0
                total_recon += weight * raw_loss
                kl_metrics[f'loss_{name}'] = raw_loss.item()
                kl_metrics[f'weight_{name}'] = weight
                continue

            # 3. Resolve Weight from Annealers
            annealer_key = self._resolve_annealer_key(name)
            
            if annealers is not None and global_step is not None:
                if annealer_key in annealers:
                    # Pull weight directly from your StochasticAnnealer
                    current_weight = annealers[annealer_key].get_beta(global_step)
                else:
                    # Fail-safe: if training and key is missing, we don't apply the penalty
                    logger.warning(f"No annealer found for {annealer_key} (from {name}). Defaulting to 0.0")
                    current_weight = 0.0
            else:
                current_weight = 1.0

            scaled_kl = current_weight * raw_loss
            kl_sum += scaled_kl
        
            kl_metrics[f'loss_{name}'] = raw_loss.item()
            kl_metrics[f'weight_{name}'] = current_weight
            
        # --- Total Objective ---
        total_loss = total_recon + kl_sum + gp_loss
        
        metrics = {
            'loss_total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'loss_kls': kl_sum.item() if isinstance(kl_sum, torch.Tensor) else kl_sum,
            'loss_gp': gp_loss.item() if isinstance(gp_loss, torch.Tensor) else gp_loss,
            'loss_recon': total_recon.item() if isinstance(total_recon, torch.Tensor) else total_recon,
            **kl_metrics
        }
    
        return total_loss, gp_loss, metrics
    
    
    
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