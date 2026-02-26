import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F

class EvidenceLowerBound(nn.Module):
    def __init__(self, 
                 gp_model, 
                 likelihood = None, 
                 num_data=38003, 
                 kl_weights=None):
        """
        Args:
            gp_model: Your AcceleratedKernelGP instance.
            likelihood: The GPyTorch likelihood attached to your GP (e.g., GaussianLikelihood).
            num_data: Total number of samples in your entire training dataset (crucial for GP KL scaling).
        """
        super().__init__()
        self.likelihood = likelihood if likelihood is not None else gpytorch.likelihoods.GaussianLikelihood()
        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood, 
            model=gp_model, 
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
        kl_metrics = {}
        
        kl_loss = 0.0

        vae_alpha_loss = 0.0
        vae_ls_loss = 0.0

        dir_global_loss = 0.0
        dir_local_loss = 0.0
        
        recon_loss = F.l1_loss(ss_history.recons, x_target, reduction='sum')
        
        for name, added_loss_term in model.named_added_loss_terms():
            raw_loss = added_loss_term.loss()
            weight = self.kl_weights.get(name, 1.0)
            scaled_loss = weight * raw_loss
            kl_loss += scaled_loss
            
            kl_metrics[f'loss_{name}'] = scaled_loss.item()
            
            if "global" in name:
                dir_global_loss += scaled_loss
            elif "local" in name:
                dir_local_loss += scaled_loss
            elif "lengthscale" in name:
                vae_ls_loss += scaled_loss
            elif "alpha" in name:
                vae_alpha_loss += scaled_loss
        
        # --- GP Marginal Log Likelihood (Variational ELBO)-- negative for gradient descent --- #
        gp_loss = -self.mll(gp_output, gp_target).sum()

        # --- Loss function ---
        total_loss = recon_loss + kl_loss + gp_loss
        
        def to_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        
        metrics = {
            'loss_total': to_item(total_loss),
            'loss_recon': to_item(recon_loss),
            'loss_kls': to_item(kl_loss),
            'loss_gp': to_item(gp_loss),
            'dir_global_kl': to_item(dir_global_loss),
            'dir_local_kl': to_item(dir_local_loss),
            'vae_alpha_kl': to_item(vae_alpha_loss),
            'vae_ls_kl': to_item(vae_ls_loss),
            **kl_metrics
        }
        
        return total_loss, metrics