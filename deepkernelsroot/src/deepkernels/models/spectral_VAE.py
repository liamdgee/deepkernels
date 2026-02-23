import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import torch.distributions

from deepkernels.models.encoder import RecurrentEncoder
from deepkernels.models.decoder import SpectralDecoder
from deepkernels.models.dirichlet import AmortisedDirichlet
from deepkernels.kernels.deepkernel import DeepKernel
from typing import Tuple, Optional, TypeAlias, Tuple, Union
import numpy as np
import logging
from gpytorch.mlls import AddedLossTerm
from torch.distributions import kl_divergence
import gpytorch

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SpectralVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dirichlet = AmortisedDirichlet()
        self.decoder = SpectralDecoder()
        self.encoder = RecurrentEncoder()
        self.eps = 1e-4
    
    def dirichlet_sample(self, alpha):
        alpha = F.softplus(alpha)
        alpha = torch.clamp(alpha, min=4e-2)
        q_alpha= torch.distributions.Dirichlet(alpha)
        pi_sample = q_alpha.rsample()
        return pi_sample
    

    def refinement_loop(self, x, vae_out=None, steps=3, batch_shape=torch.Size([]), **params):
        pi_current = params.get("pi_current", None)
        features_current = params.get("features_currnt", None)
        alpha_current = params.get("alpha_current", None)
        for _ in range(steps):
            encoder_out = self.encoder(x, pi=pi_current, spectral_features=features_current)
            alpha, mu_alpha, factor_alpha, diag_alpha, pi, z, mu_z, logvar_z = 
            if alpha.dim() > 2:
                alpha = alpha.squeeze(-1)
            
            pi_refined = self.dirichlet_sample(alpha)

            
            dirichlet_out = self.dirichlet(z, alpha=alpha, ls=ls, features_only=True)

            post_z_std = torch.exp(0.5 * logvar_z)
            post_z_eps = torch.randn_like(post_z_std)
            post_z = mu_z + post_z_eps * post_z_std #-reparameterisation trick-#
            
        recon_step = self.decoder(features_current)

    def forward(self, x, steps=3, pi=None, spectral_features=None):
        """
        Args:
            x: Input data [Batch, Input_Dim]
            n_steps: Number of refinement iterations (Default: 3)
            pi: Optional initial guess for mixture weights
            spectral_features: Optional initial spectral features
        """
        pi_current = pi
        features_current = spectral_features
        all_recons = []
        all_kls = []
        all_dirichlet = []
        evolving_features = []

        #-refinement loop-#
        for step in range(steps):
            alpha_params, latent_params = self.encoder(x, pi=pi_current, spectral_features=features_current)
            if alpha.dim() > 2:
                alpha = alpha.squeeze(-1)
            alpha = torch.clamp(alpha, min=1e-3, max=100.0)
            q_alpha= torch.distributions.Dirichlet(alpha)
            pi_current = q_alpha.rsample()

            features_current, dirichlet_kls = self.dirichlet(z, alpha=alpha, ls=ls, features_only=True)
            post_z_std = torch.exp(0.5 * logvar)
            post_z_eps = torch.randn_like(post_z_std)
            post_z = mu + post_z_eps * post_z_std #-reparameterisation trick-#
            
            recon_step = self.decoder(features_current)
            
        all_recons.append(recon_step)
        all_kls.append(post_z)
        all_dirichlet.append(dirichlet_kls)
        return {
                'spectral_features': features_current, 
                'recon_loss': all_recons, 
                'kls_latent': all_kls,
                'dirichlet_losses': all_dirichlet, 
                'z': z, 
                'alpha': alpha, 
                'ls':   ls, 
                'pi':pi_current
        }
    
    def compute_loss(self, kls, dir_kl, recons, y_target, input_dim=None, dirichlet_local_beta=1.0, dirichlet_global_beta=1.0, latent_kl_beta=0.5):
        """
        Args:
            vae_out: Dict containing lists 'recons', 'kls' (from VAE forward)
            target_data: The real input x [Batch, D]
        """
        # --- 1. Setup Constants ---
        B = float(y_target.size(0))
        N = 38003.0 
        scalar = 1.0
        input_dim = 30

        # --- 2. Iterative Loss (Reconstruction + Latent KL) ---
        # We sum the loss over all refinement steps
        n_steps = len(recons)
        total_recon_loss = 0.0
        total_latent_kl = 0.0

        device = y_target.device
        latent_dim = 16

        p_z_prior = torch.distributions.Normal(
            torch.zeros(1, device=device), 
            torch.ones(1, device=device)
        )

        for t in range(n_steps):
            recon_t = recons[t]
            step_recon_sum = F.l1_loss(recon_t, y_target, reduction='sum') 
            total_recon_loss += step_recon_sum
            q_z_t = kls[t]
            step_kl = torch.distributions.kl_divergence(q_z_t, p_z_prior).sum()
            total_latent_kl += step_kl
            total_dir_kl = dir_kl[t]
            total_dir_kl += (total_dir_kl / B)
    

        # --- 4. Total Loss ---
        avg_recon = (total_recon_loss / n_steps) * scalar
        avg_kl_z = (total_latent_kl / n_steps) * scalar * latent_kl_beta
        avg_kl_alpha = (total_dir_kl / n_steps) * scalar * dirichlet_global_beta
        total_loss = avg_recon + avg_kl_z + avg_kl_alpha
    
        return {
            "loss": total_loss,
            "recon_loss": avg_recon,
            "kl_z_loss": avg_kl_z,
            "kl_alpha_loss": avg_kl_alpha
        }
    
    def get_diagnostics(self, vae_out):
        """
        Gathers health stats for WandB. Call this during training.
        """
        spectral_features = vae_out['spectral_features']
        z = vae_out['encoder_z']
        ls = vae_out['lengthscale']
        alpha = vae_out['dirichlet_alpha']

        return {
            "manifold/feature_norm": spectral_features.norm(dim=-1).mean(),
            "manifold/z_norm": z.norm(dim=-1).mean(),
            "manifold/ls_mean": ls.mean(),
            "manifold/ls_std": ls.std(),
            "dirichlet/alpha_mean": alpha.mean(),
            "dirichlet/max_alpha": alpha.max(),
            "spectral/freq_magnitude": (1.0 / (ls + 1e-6)).mean() 
        }
    

class CyclicalAnnealer:
    def __init__(self, total_steps, n_cycles=4, ratio=0.5, start_beta=0.0, stop_beta=1.0):
        """
        Args:
            total_steps: Total training steps (Epochs * Batches_Per_Epoch)
            n_cycles: How many times to restart the annealing (e.g., 4)
            ratio: Fraction of the cycle spent annealing (vs. holding at 1.0)
            start_beta: Usually 0.0 or 1e-4
            stop_beta: Maximum weight (usually 1.0)
        """
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.start_beta = start_beta
        self.stop_beta = stop_beta

    def __call__(self, step):
        period = self.total_steps / self.n_cycles
        step_in_cycle = step % period
        cycle_progress = step_in_cycle / period
        
        if cycle_progress < self.ratio:
            rel_progress = cycle_progress / self.ratio
            beta = self.start_beta + (self.stop_beta - self.start_beta) * rel_progress
        else:
            beta = self.stop_beta
            
        return beta