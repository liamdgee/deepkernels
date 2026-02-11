import torch
import gpytorch
from torch.distributions import Normal, TransformedDistribution, kl_divergence, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
import math

import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional, TypeAlias, Tuple, Union
from gpytorch.priors import NormalPrior, GammaPrior

from src.deepkernels.losses.kl_divergence import KLDivergence
from src.deepkernels.models.beta_vae import SpectralVAE
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
    K: int = 20
    M: int = 256
    D: int = 128
    eps: float = 1e-3
    gamma_init: float = 1.75
    mu_init: float = 0.0
    sigma_init: float = 1.0
    n_tasks: int = 128
    num_inducing: int = 1024
    sigma_q_init: float = 1.5
    sigma_h_init: float = 1.0
    atom_factor: float = 0.0075


class AmortisedDirichlet(gpytorch.Module):
    def __init__(self, config:HDPConfig, vae=SpectralVAE):
        super().__init__()
        self.config = config or HDPConfig()
        self.n_data = config.n_data
        self.K = self.config.K
        self.M = self.config.M #-rff samples per mixture-#
        self.D = self.config.D #-input dim-#
        self.eps = self.config.eps
    

        #--init constraints--#
        self.mu_init = self.config.mu_init
        self.sigma_init = self.config.sigma_init
        self.log_sigma_init = math.log(self.sigma_init)
        self.sigma_q_init = self.config.sigma_q_init
        self.qsig = (math.log(self.sigma_q_init)) * -1
        self.sigma_h_init = self.config.sigma_h_init
        self.hsig = (math.log(self.sigma_h_init)) * -1
        self.atom_factor = self.config.atom_factor

        # ---------------------------------------------------------
        # Structure: GEM Global Mixing Weights (Beta)
        # ---------------------------------------------------------
        #-- variational params - q(v) ~ N(q_mu, q_log_sigma)
        self.q_mu = nn.Parameter(torch.randn(self.K - 1)) #--break symmetry-#
        self.q_log_sigma = nn.Parameter(torch.full((self.K - 1,), self.qsig)) #-cluster stability-#

        #-priors-#
        self.register_prior("global_lengthscale_prior", GammaPrior(concentration=2.5, rate=3.5), lambda m: m.h_log_sigma.exp(), lambda m, v: None)

        self.register_prior("local_lengthscale_prior", GammaPrior(concentration=3.0, rate=5.0), lambda m: m.atom_log_sigma.exp(), lambda m,v: None)
        
        self.register_prior("gamma_prior", GammaPrior(2.25, 1.25), lambda m: F.softplus(m.gamma), lambda m, v: None)

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
        # C) Amortized Inference (pi_encoder)
        # ---------------------------------------------------------
        #--Data Encoder for local mixing weights-#
        self.pi_encoder = vae(input_dim=self.D, k_atoms=self.K, M=self.M, latent_dim=16, hidden_dim=128)

        # ---------------------------------------------------------
        # D) Concentration (Gamma)
        # ---------------------------------------------------------
        #-- Learnable scalar for dirichlet process concentration (gamma)-#
        self.gamma_init = getattr(self.config, 'gamma_init', 2.0)
        self.gamma = nn.Parameter(torch.tensor(float(self.gamma_init)))

        #--Define torch stickbreak module-#
        self.stick_break_transform = StickBreakingTransform()
    
    def dynamic_random_fourier_features(self, z, omega, pi):
        """
        Args:
            z: [Batch, D]
            omega: [Batch, K, M, D] (dynamic) OR [K, M, D] (static)
            pi: [B, K] 
        """
        B, D = z.shape

        #-determine if omega is dynamic-#
        if omega.dim() == 4:
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) #-dynamic-#
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W) #-static-#
            proj = proj.view(B, self.K, self.M)
        
        #-phase shift-#
        proj = proj + self.noise_bias.unsqueeze(0)

        #-harmonics-#
        cos_proj = torch.cos(proj)
        sin_proj = torch.sin(proj)

        #-Mixing weights (amortised)-#
        #-unsqueeze [B, K] -> [B, K, 1]
        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl

        #-normalisation-#
        scale = 1.0 / math.sqrt(self.M)
        cos_proj = cos_proj.flatten(1) * scale
        sin_proj = sin_proj.flatten(1) * scale

        #-concat/flatten-#
        #-[B, K, M, 2] -> [B, K*M*2]
        feats = torch.cat([cos_proj, sin_proj], dim=-1)

        return feats #-[B, (K * M * 2)]

    def forward(self, z: Optional[torch.Tensor]=None, batch_shape=torch.Size([]), rff_kernel=True) -> DualDirichletOutput:
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

        # -- C) Compute Mixture Weights (beta) for downstream use --
        # Transform z -> v -> beta (stick breaking)
        qv_global = torch.sigmoid(qz_global)
        one_minus_v = 1 - qv_global
        cumprod_one_minus_v = torch.cumprod(one_minus_v, dim=-1)
        
        # 2. Shift cumprod to get the "previous" remaining stick
        previous_remaining = torch.roll(cumprod_one_minus_v, 1, dims=-1)
        previous_remaining[..., 0] = 1.0 

        #-Stick-Breaking Construction-#
        # beta_k = v_k * prod_{j<k} (1 - v_j)
        # 3. Calculate first K-1 weights
        beta_k = qv_global * previous_remaining
        
        # 4. Calculate the final Kth weight (the leftover)
        # The last element is whatever is left after the (K-1)th cut
        beta_last = cumprod_one_minus_v[..., -1:]
        
        # 5. Concatenate to get full [K] simplex
        beta = torch.cat([beta_k, beta_last], dim=-1)

        #--D) Dynamic local weights (pi)--#
        prior_conc = (gamma * beta) + self.eps
        prior_conc = torch.clamp(prior_conc, min=1e-2, max=100.0)

        #--E) amortised inference--#
        ls_pred = None
        
        if z is not None:
            batch_size = z.size(0)
            prior_conc_expanded = prior_conc.unsqueeze(0).expand(batch_size, -1)
        else:
            if len(batch_shape) > 0:
                batch_size = batch_shape[0]
                prior_conc_expanded = prior_conc.expand(*batch_shape, -1)
            else:
                batch_size = 1
                prior_conc_expanded = prior_conc.unsqueeze(0)
        
        if z is None:
            #-generative fallback-#
            post_conc = prior_conc_expanded
            #-0 analytical kl-#
            self.update_added_loss_term("gen_0_div", KLDivergence(post_conc, post_conc))
        else:
            #--amortised inference in training-#
            #-Data evidence from neural net (pi_encoder)-#
            #- z: [B, D] -> data_conc: [B, K]-#
            vae_out = self.pi_encoder(z)
            local_conc = vae_out['alpha'] #-[B, K]-#
            amortised_ls = vae_out['ls']
            ls_pred = torch.clamp(amortised_ls, min=self.eps, max=5.0)
            
            #-Posterior Concentration (global belief + local evidence)-
            local_conc = torch.clamp(local_conc, min=self.eps, max=100.0)

            post_conc = prior_conc.unsqueeze(0) + local_conc
            post_conc = torch.clamp(post_conc, min=self.eps, max=100.0)
            #--kl loss for only when data is present-#
            #-Create Distribution objects for kl loss updates-#
            pr_dist_pi = Dirichlet(prior_conc) #-[B, K]
            post_dist_pi = Dirichlet(post_conc) #-- [B, K]
            
            #-analytical kl for training & inference-#
            self.update_added_loss_term("pi_kl", KLDivergence(post_dist_pi, pr_dist_pi))
        
        #--Sample from pi (dynamic)-#
        pi = Dirichlet(post_conc).rsample() #--[B, K]-#

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
       

        #-omega ~ N(spectral_means, bandwidth)
        #- omega shape: - [K, M, D]-#
        omega = spectral_means + (self.noise_weights * bw)
        omega = torch.clamp(omega, -100.0, 100.0)

        if z is not None and vae_out is not None:
            target_rff = self.dynamic_random_fourier_features(z, omega, pi=torch.ones_like(pi)) #-dummy for kernel learning-#
            vae_loss = self.pi_encoder.loss(vae_out, target_rff.detach())
            self.update_added_loss_term("vae_loss", vae_loss)
        #--Returns:
        #pi: [B, K] -> local mixing weights
        #beta: [K] -> global prevalence
        #omega: [K, M, D] -> spectral frequencies
        #bias: [K, M] -> spectral phases (fixed)
        if rff_kernel:
            fourier_features = self.dynamic_random_fourier_features(z, omega, pi)
            logger.info("Outputs are tailored for RFF kernel: Shape: [Batch, K * M * 2]")
            return fourier_features
        
        if bw.dim() == 4 and bw.shape[2] == 1:
            bw = bw.squeeze(2)
        print("Debugging / Exact Mode: Outputs are tailored for an exact kernel")
        return pi, beta, spectral_means, bw