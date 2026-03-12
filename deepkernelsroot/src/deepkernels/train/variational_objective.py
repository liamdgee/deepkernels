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
                 **kwargs):
        """
        Args:
            gp_model: Your AcceleratedKernelGP instance.
            likelihood: The GPyTorch likelihood attached to your GP (e.g., GaussianLikelihood).
            num_data: Total number of samples in your entire training dataset (crucial for GP KL scaling).
        """
        super().__init__()
        n_data = kwargs.get('n_data', 76674)
        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=model.gp.likelihood, 
            model=model.gp, 
            num_data=n_data
        )
        
        base_kl_weights = {
            "lengthscale_kl": 1.0,
            "alpha_kl": 1.0,
            "global_divergence": 0.1,
            "local_divergence": 0.1,
            "inverse_wishart": 1.0,
            "recon_kl": 0.005
        }

        self.kl_weights = kwargs.get('kl_weights', None)

        if self.kl_weights is None:
            self.kl_weights = base_kl_weights
    
    def _get_cyclic_beta(self, global_step, total_steps, n_cycles=4, ratio=0.4):
        if global_step is None or total_steps is None:
            return 1.0
        cycle_length = total_steps // n_cycles
        step_in_cycle = global_step % cycle_length
        ramp_length = int(cycle_length * ratio)
        return min(1.0, step_in_cycle / ramp_length) if ramp_length > 0 else 1.0
    
    def forward(self, model, gp_output, gp_target, state_out, global_step=None, total_steps=None, annealers=None, **kwargs):
        """
        Args:
            model: overarching SpectralVAE model (to pull added loss terms).
            x_target: ground-truth data sequence [Batch, SeqLen, Features].
            ss_history: VAE state space history namedtuple
            gp_output: The MultivariateNormal returned by your GP.
            gp_target: The sequence the GP is supposed to be predicting. (ground truth)
        """
        device = gp_target.device if gp_target is not None else next(model.parameters()).device
        kl_metrics = {}
        
        cyclic_beta = self._get_cyclic_beta(global_step, total_steps)
        kl_metrics['master_cyclic_beta'] = cyclic_beta

        if gp_output is not None and gp_target is not None:
            gp_loss = -self.mll(gp_output, gp_target)
        else:
            gp_loss = torch.tensor(0.0, device=device)
        
        kl_sum = 0.0

        total_recon = 0.0

        for name, added_loss_term in model.named_added_loss_terms():
            if 'gp' in name: continue
            raw_loss = added_loss_term.loss().sum()
            if 'recon_term' in name:
                total_recon = total_recon + raw_loss
                kl_metrics[f'loss_{name}'] = raw_loss.item()
                continue
            base_weight = self.kl_weights.get(name, 1.0)
            current_beta = cyclic_beta
            if annealers is not None and global_step is not None:
                if name in annealers:
                    current_beta = annealers[name].get_beta(global_step)
                elif 'divergence' in name:
                    target = 'global_divergence' if 'global' in name else 'local_divergence'
                    annealer = annealers.get(target)
                    if annealer:
                        current_beta = annealer.get_beta(global_step)
                elif 'recon' in name:
                    target = 'recon_kl'
                    annealer = annealers.get(target)
                    if annealer:
                        current_beta = annealer.get_beta(global_step)
                elif 'kl' in name:
                    target = 'alpha_kl' if 'alpha' in name else 'lengthscale_kl'
                    annealer = annealers.get(target)
                    if annealer:
                        current_beta = annealer.get_beta(global_step)
                        
                elif 'inverse_wishart' in name:
                    annealer = annealers.get('inverse_wishart')
                    if annealer:
                        current_beta = annealer.get_beta(global_step)
            
            scaled_kl = base_weight * current_beta * raw_loss
            kl_sum = kl_sum + scaled_kl
        
            kl_metrics[f'loss_{name}'] = raw_loss.item()
            kl_metrics[f'beta_{name}'] = current_beta
            
        
        # --- GP Marginal Log Likelihood (Variational ELBO)-- negative for gradient descent --- #
        total_loss = total_recon + kl_sum + gp_loss
        
        def to_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        metrics = {
            'loss_total': to_item(total_loss),
            'loss_kls': to_item(kl_sum),
            'loss_gp': to_item(gp_loss),
            'loss_recon': to_item(total_recon),
            'master_beta': cyclic_beta,
            **kl_metrics
        }
    
        return total_loss, metrics