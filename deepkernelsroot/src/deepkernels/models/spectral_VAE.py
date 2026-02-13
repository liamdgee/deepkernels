import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import torch.distributions

from src.deepkernels.models.encoder import RecurrentEncoder
from src.deepkernels.models.decoder import SpectralDecoder
from src.deepkernels.models.dirichlet import AmortisedDirichlet
from src.deepkernels.kernels.deepkernel import DeepKernel

from typing import Tuple, Optional, TypeAlias, Tuple, Union
import wandb
import numpy as np
import logging


#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpectralVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dirichlet = AmortisedDirichlet()
        self.decoder = SpectralDecoder()
        self.encoder = RecurrentEncoder(input_dim=44, hidden_dims=[128, 64, 32], latent_dim=16, dropout=0.1, k_atoms=30, M=128)
        self.eps = 1e-4

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
        all_latents = []
        evolving_features = []

        #-refinement loop-#
        for _ in range(steps):
            z, alpha, ls, mvn, qz_dist = self.encoder(x, pi=pi_current, spectral_features=features_current)
            pi_simplex = torch.distributions.Dirichlet(torch.clamp(alpha, min=1e-3, max=100.0))
            pi_current = pi_simplex.rsample()
            features_current = self.dirichlet(z, alpha=alpha, ls=ls, features_only=True)
            recon_step = self.decoder(features_current)
            all_recons.append(recon_step)
            all_kls.append(mvn)
            all_latents.append(qz_dist)
            evolving_features.append(features_current)
        
        return {
            'reconstruction_loss_per_step': all_recons,
            'empirical_kl_div_per_step': all_kls,
            'spectral_features_per_step': evolving_features,
            'features_out': features_current,
            'z_out': z,
            'alpha_concentration_out': alpha,
            'lengthscales_out': ls,
            'empirical_latent_kl_per_step': all_latents,
            'simplex_sample_out': pi_current
        }
    
    def compute_loss(self, vae_out, y_target, input_dim=None, dirichlet_local_beta=1.0, dirichlet_global_beta=1.0, latent_kl_beta=0.5):
        """
        Args:
            vae_out: Dict containing lists 'recons', 'kls' (from VAE forward)
            target_data: The real input x [Batch, D]
        """
        # --- 1. Setup Constants ---
        B = float(y_target.size(0))
        N = 38003.0 
        scalar = N / B # Scaling for batch
        input_dim = input_dim or 44

        # --- 2. Iterative Loss (Reconstruction + Latent KL) ---
        # We sum the loss over all refinement steps
        
        recons = vae_out['reconstruction_loss_per_step']   # List of Tensors
        kls = vae_out['empirical_kl_div_per_step']         # List of Distributions (mvn)
        latent_kls = vae_out['empirical_latent_kl_per_step']
        
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
            p_z = kls[t]
            q_z_t = latent_kls[t]
            step_kl = torch.distributions.kl_divergence(q_z_t, p_z_prior).sum()
            total_latent_kl += step_kl
        
        total_recon_loss = total_recon_loss / (B * input_dim)
        avg_recon_loss = (total_recon_loss / n_steps) * scalar
        total_latent_kl = total_latent_kl / (B * latent_dim)
        avg_latent_kl = (total_latent_kl / n_steps) * scalar

        # --- 3. Dirichlet Losses (Global/Local) ---
        local_dirichlet_loss = 0.0
        global_dirichlet_loss = 0.0

        if hasattr(self.dirichlet, 'named_added_loss_terms'):
            for name, added_loss in self.dirichlet.named_added_loss_terms():
                if "stick_breaking" in name:
                    global_dirichlet_loss += added_loss.loss()
                else:
                    local_dirichlet_loss += added_loss.loss()

        total_local_dirichlet = (local_dirichlet_loss * scalar) * dirichlet_local_beta
        total_global_dirichlet = (global_dirichlet_loss * 1.0) * dirichlet_global_beta

        # --- 4. Total Loss ---
        total_loss = (
            avg_recon_loss + 
            (avg_latent_kl * latent_kl_beta) + 
            total_local_dirichlet + 
            total_global_dirichlet
        )
        
        return {
            "loss": total_loss,
            "recon_loss": avg_recon_loss.item(),
            "kl_loss": avg_latent_kl.item(),
            "dirichlet_local_loss": total_local_dirichlet.item() if isinstance(total_local_dirichlet, torch.Tensor) else total_local_dirichlet,
            "dirichlet_global_loss": total_global_dirichlet.item() if isinstance(total_global_dirichlet, torch.Tensor) else total_global_dirichlet
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