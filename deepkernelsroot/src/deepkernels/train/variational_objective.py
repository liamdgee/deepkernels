import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F
from typing import NamedTuple

class EvidenceLowerBound(nn.Module):
    def __init__(self, 
                 model,
                 num_data=38003, 
                 kl_weights=None):
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
            num_data=num_data
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
        
        # --- GP Marginal Log Likelihood (Variational ELBO)-- negative for gradient descent --- #
        elbo = self.mll(gp_output, gp_target) #-gp loss-#

        # --- Loss function ---
        total_loss = kl_loss -elbo
        
        def to_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        
        metrics = {
            'loss_total': to_item(total_loss),
            'loss_vae': to_item(total_vae_loss), 
            'loss_kls': to_item(kl_loss),
            'loss_gp': to_item(elbo),
            'loss_recon': to_item(recon_loss),
            **kl_metrics
        }
        
        return total_loss, metrics