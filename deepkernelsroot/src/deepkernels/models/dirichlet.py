import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
import math
import logging
from typing import Optional

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.losses.simple import SimpleLoss
from deepkernels.models.NKN import KernelNetwork
from pydantic import BaseModel

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HDPConfig(BaseModel):
    K: int = 30
    M: int = 128
    D: int = 16
    eps: float = 1e-3
    gamma_init: float = 1.75
    num_inducing: int = 1024
    atom_factor: float = 0.0075

class AmortisedDirichlet(BaseGenerativeModel):
    def __init__(self, 
                 config=None, 
                 k_atoms=30, 
                 fourier_dim=128, 
                 latent_dim=16, 
                 spectral_emb_dim=2048, 
                 num_latents=8, 
                 bottleneck_dim=64, 
                 gamma_concentration_init=2.5):
        
        super().__init__()
        self.config = config or HDPConfig()
        self.K = k_atoms
        self.M = fourier_dim
        self.D = latent_dim
        self.eps = 1e-3

        #-hypernetworks
        self.compress_spectral_features_head = torch.nn.utils.spectral_norm(nn.Linear(k_atoms * fourier_dim * 2, spectral_emb_dim))
        
        self.bottleneck_mixer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, bottleneck_dim)), 
            nn.LayerNorm(bottleneck_dim), 
            nn.Tanh()
        )
        
        self.kernel_network = KernelNetwork()
        
        #-params-#
        self.q_mu_global = nn.Parameter(torch.zeros(k_atoms - 1))
        self.q_log_sigma_global = nn.Parameter(torch.ones(k_atoms - 1) * -4.0)
        self.h_mu = nn.Parameter(torch.zeros(1, 1, latent_dim)) 
        self.h_log_sigma = nn.Parameter(torch.tensor(3.0))
        self.atom_log_sigma = nn.Parameter(torch.randn(k_atoms, 1, latent_dim) * 0.01)
        self.atom_mu = nn.Parameter(torch.randn(k_atoms, 1, latent_dim) * 2 * math.sqrt(0.01))

        #-buffers-#
        self.register_buffer("noise_weights", torch.randn(k_atoms, fourier_dim, latent_dim))
        self.register_buffer("noise_bias", torch.rand(k_atoms, fourier_dim))

        safe = self.numerically_stable_gamma(gamma_concentration_init)
        
        self.raw_gamma = nn.Parameter(torch.tensor(safe))

    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        """
        performs nonparametric clustering according to a hierarchical dirichlet process using learned lengthscale
        and concentration param refinement via pi (logit mixture weights)
        Args:
            latent z (param: x) -- dim 16
        """

        _, mualpha, cholalpha, diagalpha, _ = vae_out['alpha_params']
        z, _, _ = vae_out['latent_params']
        ls = params.get('ls', None)

        if z is not None:
            x = z

        beta, log_pv, log_qv, gamma_conc = self.global_stick_breaking()

        self.log_global_kl(log_pv, log_qv)
    
        bottleneck, gate = self.run_neural_nets_dirichlet(x)
        
        local_conc = self.get_local_evidence(mualpha, cholalpha, diagalpha)

        pi = self.dirichlet_posterior_inference_and_log_local_loss(x, gamma_conc, beta, local_conc)
        
        ls_pred, bw_learned = self.predict_kernel_lengthscale_and_log_mse_loss(ls)
        
        omega = self.get_omega(bw_learned)

        raw_features = self.random_fourier_features(x, omega, pi)

        gated_features = self.compress_and_gate(raw_features, gate)
        
        spectral_features = {
            "features": gated_features,
            "frequencies": omega,
            "gated_weights": gate
        }

        lengthscale_features = {
            "predicted_lengthscale": ls_pred,
            "learned_bandwidth": bw_learned,
        }
        latent_features = {
            "z": x,
            "bottleneck": bottleneck,
        }

        probabilistic_features = {
            'beta': beta,
            'pi': pi,
            'concentration_prior': gamma_conc,
            'concentration_posterior': local_conc
        }
        if features_only:
            return bottleneck, pi, omega, gated_features
        
        return spectral_features, lengthscale_features, latent_features, probabilistic_features
    
    
   

