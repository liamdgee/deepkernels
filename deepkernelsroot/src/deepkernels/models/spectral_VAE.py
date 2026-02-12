import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import torch.distributions.Dirichlet as Dir
from src.deepkernels.models.encoder import RecurrentEncoder
from src.deepkernels.models.harmonic_decoder import SpectralDecoder
from src.deepkernels.models.dirichlet import AmortisedDirichlet
from src.deepkernels.models.NKN import NeuralKernelNetwork
import wandb
import numpy as np
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpectralVAE(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.dirichlet = AmortisedDirichlet()
        self.decoder = SpectralDecoder()
        self.kerneldecoder = NeuralKernelNetwork()
        self.encoder = RecurrentEncoder()
        self.eps = 1e-4
        self.n_steps = n_steps

    def forward(self, x):
        pi_feedback = None 
        z, mu, logvar, alpha, ls = None, None, None, None, None

        # --- Refinement Loop --- #
        for step in range(self.n_steps + 1):
            z, mu, logvar, alpha, ls = self.encoder(x, pi=pi_feedback)

            #-derive pi by sampling alpha conc from dirichlet distribution-#
            if step < self.n_steps:
                conc = torch.clamp(alpha, min=self.eps, max=100.0)
                pi_dist = Dir(conc)
                pi_feedback = pi_dist.rsample().detach()
        # Now we have the refined z, ls, and alpha through approximate methods, proceed with projection and decoding spectral features-#

        #-sample dirichlet module -- logs kl loss internally-#
        spectral_features, _ = self.dirichlet(z, rff_kernel=True, ls_override=ls, alpha_override=alpha)
        recon_x = self.decoder(spectral_features) #- and this-#
        kernel_decomposition = self.kerneldecoder(x1=spectral_features, x2=spectral_features.shape[-1]) #-[input_dim, z] #--****** edit this-#
        l2_norm = torch.norm(spectral_features, dim=-1).mean()
        vae_out = {
            'recon': recon_x,
            'mu': mu,
            'logvar': logvar,
            'dirichlet_alpha': alpha,
            'harmonic_features': spectral_features,
            'kernel_mixture': kernel_decomposition,
            'spectral_l2_norm': l2_norm,
            'encoder_z': z,
            'lengthscale': ls
        }

        return vae_out
    
    def loss(self, vae_out: dict, target_rff: torch.Tensor, latent_kl_beta: float = 0.7, dirichlet_global_beta: float = 0.9, dirichlet_local_beta: float = 1.3):
        """
        Calculates the VAE Construction Loss for the Encoder.
        
        Args:
            vae_out: Dictionary/Tuple containing:
                     - 'recon': The reconstructed spectral features [Batch, Feature_Dim]
                     - 'mu': Latent mean [Batch, Latent_Dim]
                     - 'logvar': Latent log variance [Batch, Latent_Dim]
            target_rff: The "Ground Truth" spectral features derived from the Dirichlet module.
                        Shape: [Batch, Feature_Dim] (Flattened K*M*2)
            kl_beta: Scaling factor for the KL divergence
        """
        #-l1 recon loss (sum for VAE) - Since RF features are bounded between [-1, 1]-#
        B = float(target_rff.size(0))
        N = 38003.0
        scalar = N / B #-equals N/B-#

        recon = vae_out['recon']
        feature_dim = float(recon.shape[1])
        recon_sum = F.l1_loss(recon, target_rff, reduction='sum')
        recon_sum_per_feat = recon_sum / feature_dim
        recon_loss = recon_sum_per_feat * scalar

        #-analytic gaussian KL Divergence-#
        mu, logvar = vae_out['mu'], vae_out['logvar']
        
        latent_dim = float(mu.shape[1])

        kl_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_z = (kl_batch * scalar) / latent_dim
        
       # --- HANDLING DIRICHLET INTERNAL LOSSES ---
        local_dirichlet_loss = 0.0
        global_dirichlet_loss = 0.0

        if hasattr(self.dirichlet, 'added_loss_terms'):
            for name, added_loss in self.dirichlet.named_added_loss_terms():
                
                if "global" in name or "stick_breaking" in name:
                    global_dirichlet_loss += added_loss.loss()
                else:
                    local_dirichlet_loss += added_loss.loss()

        total_local_dirichlet = (local_dirichlet_loss * scalar) * dirichlet_local_beta

        total_global_dirichlet = (global_dirichlet_loss * 1.0) * dirichlet_global_beta

        total_loss = recon_loss + (kl_z * latent_kl_beta) + total_local_dirichlet + total_global_dirichlet
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_z.item(),
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