from src.deepkernels.models.dirichlet import AmortisedDirichlet, HDPConfig
from src.deepkernels.models.spectral_VAE import SpectralVAE
from src.deepkernels.models.encoder import RecurrentEncoder
from src.deepkernels.models.linear_decoder import BayesDecoder
from src.deepkernels.kernels.deepkernel import DeepKernel
from src.deepkernels.models.deepkernels import DeepKernelProcess
from typing import Tuple, Optional, TypeAlias, Tuple, Union
import torch
import gpytorch
import torch.nn as nn
import torch.distributions.Dirichlet as Dir


class GenerativeKernelProcess(nn.Module):
    """
    Data Flow:
    1. real_x -> *skip* -> gp output layer (all means are interprettable as mean module never enters latent space.)
    2. Input (real_x) -> 'RecurrentEncoder' -> Latent (z)
    3. Latent (z) -> 'AmortisedDirichlet' Module for nonparametric clustering -> gradients flow back to 'Recurrent Encoder' via amortised inference
    3. Latent (z) -> VAE Decoder -> Reconstruction (x_hat) [Regularization 2]
    4. Latent (z) -> VAE_decoder -> Primitive Kernel Deconstruction inside gp kernel
    5. latent(z) -> linear model of coregionalisation gaussian process in function space  -> Prediction (y_hat)

    yhat_k_{k=1, 2, ... , 30} ~ sum_{{weight{q_k} * GP(mu_real_{k}, sigma_latent_{q_k})}} for q in rank
    
    where mu_real_{i} is deterministic per cluster
    """
    def __init__(self, n_steps=2, k_atoms=30):
        super().__init__()

        self.vae = SpectralVAE()
    
    def forward(self, x, run_loop=False, **params):
        if run_loop:
            pi = self.vae.pi_loop()
        vae_out = self.vae(x, pi)
        feats, z, omega = vae_out['spectral_features'], vae_out['z'], vae_out['omega']
        phi = self.get_batched_gp_features(z, omega)
        return feats, phi
    
    def get_batched_gp_features(self, z=None, omega=None):
        """
        z: [Batch, D]
        omega: [K, M, D]
        """
        K = omega.size(0)
        z_k = z.unsqueeze(0).expand(K, -1, -1)
        omega_t = omega.transpose(1, 2)
        projection = torch.bmm(z_k, omega_t)
        projection = projection.permute(1, 0, 2)
        cos_features = torch.cos(projection)
        sin_features = torch.sin(projection)
        phi = torch.cat([cos_features, sin_features], dim=-1)
        phi_for_batch_gp = phi.permute(1, 0, 2)
        return torch.bmm(phi_for_batch_gp, phi_for_batch_gp.transpose(1, 2))
