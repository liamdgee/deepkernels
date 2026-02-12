import torch
import gpytorch
from torch.distributions import Normal, TransformedDistribution, kl_divergence
import torch.distributions.Dirichlet as Dir
from torch.distributions.transforms import StickBreakingTransform
import math

import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional, TypeAlias, Tuple, Union
from gpytorch.priors import NormalPrior, GammaPrior

from src.deepkernels.losses.kl_divergence import KLDivergence
from src.deepkernels.models.spectral_VAE import SpectralVAE
from src.deepkernels.models.encoder import RecurrentEncoder
from src.deepkernels.models.NKN import NeuralKernelNetwork
from gpytorch.mlls import AddedLossTerm
import logging

DualDirichletOutput: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalKLLoss(AddedLossTerm):
    def __init__(self, kl_val):
        self.kl_val = kl_val
    
    def loss(self):
        return self.kl_val

class HDPConfig(BaseModel):
    n_data: 38003
    K: int = 30
    M: int = 128
    D: int = 16
    eps: float = 1e-3
    gamma_init: float = 1.75
    mu_init: float = 0.0
    sigma_init: float = 1.0
    num_inducing: int = 1024
    sigma_q_init: float = 1.5
    sigma_h_init: float = 1.0
    atom_factor: float = 0.0075


class AmortisedDirichlet(gpytorch.Module):
    def __init__(self, config=None, n_data=38003, k_atoms=30, fourier_dim=128, latent_dim=16):
        super().__init__()
        self.config = config or HDPConfig()
        self.n_data = n_data or config.n_data
        self.K = k_atoms or self.config.K
        self.M = fourier_dim or self.config.M #-rff samples per mixture-#
        self.D = latent_dim or self.config.D #-input dim-#
        self.eps = 1e-3

        #--init constraints--#
        self.mu_init = self.config.mu_init or 0.0
        self.sigma_init = self.config.sigma_init or 1.0
        self.log_sigma_init = math.log(self.sigma_init)
        self.sigma_q_init = self.config.sigma_q_init or 1.5
        self.qsig = (math.log(self.sigma_q_init)) * -1
        self.sigma_h_init = self.config.sigma_h_init or 1.0
        self.hsig = (math.log(self.sigma_h_init)) * -1
        self.atom_factor = self.config.atom_factor or 0.0075

        # ---------------------------------------------------------
        # Structure: GEM Global Mixing Weights (Beta)
        # ---------------------------------------------------------
        #-- variational params - q(v) ~ N(q_mu, q_log_sigma)
        self.q_mu = nn.Parameter(torch.randn(self.K - 1)) #--break symmetry-#
        self.q_log_sigma = nn.Parameter(torch.full((self.K - 1,), self.qsig)) #-cluster stability-#

        #-priors-#
        self.register_prior("global_lengthscale_prior", GammaPrior(concentration=2.5, rate=3.0), lambda m: m.h_log_sigma.exp(), lambda m, v: None)

        self.register_prior("local_lengthscale_prior", GammaPrior(concentration=3.0, rate=5.0), lambda m: m.atom_log_sigma.exp(), lambda m,v: None)
        
        self.register_prior("gamma_prior", GammaPrior(2.5, 1.0), lambda m: F.softplus(m.gamma), lambda m, v: None)

        # ---------------------------------------------------------
        # Content: Spectral Frequencies (Omega)
        # ---------------------------------------------------------
        #--spectral mixture kernel: defined by weight (pi), center (atom_mu), width (atom_log_scale)

        #--Global Base Distribution (H) defines kernel smoothness & freq
        self.h_mu = nn.Parameter(torch.zeros(1, 1, self.D)) #-center of kernels -- mu prior on gram matrix-#
        self.h_log_sigma = nn.Parameter(torch.tensor(self.hsig)) #-global bandwidth-#

        #-local atom deviation (additive)-#
        self.atom_log_sigma = nn.Parameter(torch.randn(self.K, 1, self.D) * math.sqrt(self.atom_factor))
        self.atom_mu = nn.Parameter(torch.randn(self.K, 1, self.D) * self.atom_factor)

        #--RFF Constants - fixed noise params--#
        #-draw standard normal once and freeze (fundamental random fourier projection assumption)--#
        self.register_buffer("noise_weights", torch.randn(self.K, self.M, self.D)) #-Shape: [K, M, D]-#
        self.register_buffer("noise_bias", torch.rand(self.K, self.M) * 2 * math.pi)

        # ---------------------------------------------------------
        # C) Concentration (Gamma)
        # ---------------------------------------------------------
        #-- Learnable scalar for dirichlet process concentration (gamma)-#
        self.gamma_init = 2.0
        self.gamma = nn.Parameter(torch.tensor(float(self.gamma_init)))

        #--Define torch stickbreak module-#
        self.stick_break_transform = StickBreakingTransform()
    
    def dynamic_random_fourier_features(self, z: torch.Tensor, omega: torch.Tensor, pi: Optional[torch.Tensor]=None, raw_transforms: bool=False):
        """
        Args:
            z: [Batch, D]
            omega: [Batch, K, M, D] (dynamic) OR [K, M, D] (static)
            pi: [B, K] 
        """
        B, D = z.shape

        #-determine if omega is dynamic-#
        if omega.dim() == 4:
            #-Dynamic: [B, K, M, D] * [B, 1, 1, D] -> Sum over D -#
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) #-dynamic-#
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W) #-static-#
            proj = proj.view(B, self.K, self.M)
        
        #-phase shift-#
        proj = proj + self.noise_bias.unsqueeze(0) #- [K, M] -> [1, K, M]
        
        #-scale for vae-#
        scale = 1.0 / math.sqrt(self.M)
        
        #-harmonics-#
        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale

        if raw_transforms or pi is None:
            logger.info("returned individual harmonic transforms (no concat) --  these are not scaled by pi Shape [B, K, M]")
            return cos_proj, sin_proj

        #-Mixing weights (amortised)-#
        #-unsqueeze [B, K] -> [B, K, 1]
        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl
        feats = torch.stack([cos_proj, sin_proj], dim=-1) #- [B, K, M] -> [B, K, M, 2]
        return feats.flatten(1) #-[B, (K * M * 2)]
    
    def get_omega(self, bw):
        """
        Stateless helper: inputs -> outputs. Safe for any batch.
        # bw shape: [K, M, D] or [Batch, K, M, D]
        # h_mu: [1, 1, D]
        # atom_mu: [K, 1, D]
        """
        omega = self.h_mu + self.atom_mu + (self.noise_weights * bw)
        return torch.clamp(omega, -100.0, 100.0)

    def forward(self, 
            z: Optional[torch.Tensor]=None, 
            batch_shape=torch.Size([]), 
            rff_kernel=True, 
            ls: Optional[torch.Tensor]=None, 
            alpha: Optional[torch.Tensor]=None) -> DualDirichletOutput:
        """
        Args:
            z: [Batch, D] - Latent (Required for features)
            alpha_amortised: [Batch, K] - Dirichlet Concentration (from Encoder)
            ls_amortised: [Batch, K, D] - Lengthscales (from Encoder or Override)
        """
        
        # ---------------------------------------------------------
        # 1. Global Variational Inference (Stick Breaking)
        # ---------------------------------------------------------
        #--A) define variational posterior - q(v)-#
        q_sig = F.softplus(self.q_log_sigma)
        q_dist = Normal(self.q_mu, q_sig)
        
        #-loss for q-#
        #-sample from logit space-#
        qz_global = q_dist.rsample()
        
        # -- B) Calculate Analytic KL Terms via Softplus --
        #-- Log Posterior q(v) --#
        # log q(v) = log q(z) - log |det J|
        # log |det J| = log(v(1-v)) = -softplus(-z) - softplus(z)
        log_detj = -F.softplus(-qz_global) - F.softplus(qz_global)
        log_qv = q_dist.log_prob(qz_global).sum(-1) - log_detj.sum(-1)

        # 2. Log Prior p(v) ~ Beta(1, gamma)
        # log p(v) = log(gamma) + (gamma - 1) * log(1 - v)
        # Using identity: log(1 - sigmoid(z)) = -softplus(z)
        gamma = F.softplus(self.gamma)
        log_pv = (torch.log(gamma + self.eps) + (gamma - 1) * (-F.softplus(qz_global))).sum(-1)

        # 3. KL Divergence (Monte Carlo Estimate)
        # KL = E[log q(v) - log p(v)]
        kl_div = (log_qv - log_pv).sum()
        self.update_added_loss_term("global_stick_breaking_kl", GlobalKLLoss(kl_div))

        # -- Compute Mixture Weights (beta) for downstream use --
        # Transform z -> v -> beta (stick breaking)
        qv_global = torch.sigmoid(qz_global)
        one_minus_v = 1 - qv_global
        cumprod_one_minus_v = torch.cumprod(one_minus_v, dim=-1)
        previous_remaining = torch.roll(cumprod_one_minus_v, 1, dims=-1)
        previous_remaining[..., 0] = 1.0
        beta_k = qv_global * previous_remaining
        beta_last = cumprod_one_minus_v[..., -1:]
        beta = torch.cat([beta_k, beta_last], dim=-1)

        # ---------------------------------------------------------
        # 2. Local Concentration (Prior vs Posterior)
        # ---------------------------------------------------------
        prior_conc = (gamma * beta) + self.eps
        prior_conc = torch.clamp(prior_conc, min=1e-2, max=100.0)
        ls_pred = None
        #--E) amortised inference--#
        if z is not None:
            batch_size = z.size(0)
            prior_conc_expanded = prior_conc.unsqueeze(0).expand(batch_size, -1) #-[K] -> [1, K] -> [B, K]-#
        else:
            if len(batch_shape) > 0:
                view_shape = [1] * len(batch_shape) + [-1] 
                prior_reshaped = prior_conc.view(*view_shape)
                prior_conc_expanded = prior_reshaped.expand(*batch_shape, -1)
            else:
                prior_conc_expanded = prior_conc.unsqueeze(0)
        
        if alpha is not None:
            #--amortised inference in training-#
            #-Data evidence from neural net (pi_encoder)-#
            #- z: [B, D] -> data_conc: [B, K]-#
            
            #-posterior = prior + local evidence-#
            local_conc = torch.clamp(alpha, min=self.eps, max=100.0)
            post_conc = prior_conc_expanded + local_conc
            post_conc = torch.clamp(post_conc, min=self.eps, max=100.0)
            
            #-KL Posterior-#
            #-Create Distribution objects for kl loss updates-#
            pr_dist_pi = Dir(prior_conc_expanded) #-[B, K]
            post_dist_pi = Dir(post_conc) #-- [B, K]
            self.update_added_loss_term("pi_kl", KLDivergence(post_dist_pi, pr_dist_pi))
            if ls is not None:
                ls_pred = torch.clamp(ls, min=self.eps, max=5.0)
        else:
            # --- GENERATIVE MODE ---
            # No evidence provided. We sample from the Prior.
            post_conc = prior_conc_expanded
            self.update_added_loss_term("gen_0_div", KLDivergence(post_conc, post_conc))

        #--Sample from pi (dynamic)-#
        pi = Dir(post_conc).rsample() #--[B, K]-#

        #--F) Construct Spectal frequencies with H regularisation-#
        #--centres: global center + local frequency/position-#
        spectral_means = self.h_mu + self.atom_mu
        log_scale = self.h_log_sigma + self.atom_log_sigma
        log_scale = torch.clamp(log_scale, max=10.0)
        bw_base = log_scale.exp()

        if ls_pred is not None:
            jitter = 1e-6
            bw_dyn = 1.0 / (ls_pred + jitter) #-inv ls-#
            bw = bw_base * bw_dyn.unsqueeze(2)
        else:
            bw = bw_base 
       
        omega = self.get_omega(bw) #- omega shape: - [K, M, D]-#
        
        if z is not None:
            cos_transform, sin_transform = self.dynamic_random_fourier_features(z, omega, raw_transforms=True)

            if rff_kernel and alpha is not None:
                pi_scl = torch.sqrt(pi).unsqueeze(-1)
                features = torch.stack([cos_transform * pi_scl, sin_transform * pi_scl], dim=-1).flatten(1)
                return features, omega
        
        if bw.dim() == 4 and bw.shape[2] == 1:
            bw = bw.squeeze(2)
        
        return pi, beta, spectral_means, bw, features, omega