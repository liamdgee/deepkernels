import torch
import gpytorch
from torch.distributions import Normal, TransformedDistribution, kl_divergence, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
from gpytorch.priors import GammaPrior, NormalPrior
import math

import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional, TypeAlias, Tuple, Union
from gpytorch.mlls import AddedLossTerm
import logging

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.losses.simple import SimpleLoss

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
    mu_init: float = 0.0
    sigma_init: float = 1.0
    num_inducing: int = 1024
    sigma_q_init: float = 1.5
    sigma_h_init: float = 1.0
    atom_factor: float = 0.0075


class AmortisedDirichlet(BaseGenerativeModel):
    def __init__(
            self, 
            config=None, 
            k_atoms=30, 
            fourier_dim=128, 
            latent_dim=16, 
            spectral_emb_dim=2048,
            num_latents=6,
            input_dim=30,
            spectral_clusters=4,
            bottleneck_dim=64
    ):
        self.config = config or HDPConfig()
        self.n_data = 38003
        self.K = k_atoms or self.config.K
        self.M = fourier_dim or self.config.M #-rff samples per mixture-#
        self.D = latent_dim or self.config.D #-input dim-#
        self.eps = 1e-3

        self.compress_spectral_features_head = torch.nn.utils.spectral_norm(nn.Linear(k_atoms * fourier_dim * 2, spectral_emb_dim))
        
        self.bottleneck_mixer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, bottleneck_dim)),
            nn.LayerNorm(bottleneck_dim),
            nn.Tanh()
        )

        self.linear = nn.utils.spectral_norm(nn.Linear(spectral_emb_dim, bottleneck_dim))
        self.linear_scale = nn.Parameter(torch.tensor(0.165))
        
        self.periodic = nn.utils.spectral_norm(nn.Linear(spectral_emb_dim, bottleneck_dim))
        self.rbf = nn.utils.spectral_norm(nn.Linear(spectral_emb_dim, bottleneck_dim))
        self.rational = nn.utils.spectral_norm(nn.Linear(spectral_emb_dim, bottleneck_dim))
        
        self.constant = nn.Parameter(torch.randn(num_latents, bottleneck_dim) * 0.165)
        
        self.primitive_norm = nn.LayerNorm(bottleneck_dim * num_latents)

        self.primitive_kernel_heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(bottleneck_dim, spectral_clusters) #-dim 128 for each latent gp-#
            ) 
            for _ in range(num_latents)
        ])

        # ---------------------------------------------------------
        # Content: Spectral Frequencies (Omega)
        # ---------------------------------------------------------
        #--spectral mixture kernel: defined by weight (pi), center (atom_mu), width (atom_log_scale)

        self.register_parameter(
            name="variational_omega_mu", 
            parameter=torch.nn.Parameter(torch.randn(num_latents, spectral_clusters, input_dim))
        )
        
        # 2. Variational Log Variance (The Frequency Uncertainty)
        # Shape: (6, 4, 16)
        # Init to -4.0 (small variance) to start confident, then expand if needed
        self.register_parameter(
            name="variational_omega_logvar", 
            parameter=torch.nn.Parameter(torch.ones(num_latents, spectral_clusters, input_dim) * -4.0)
        )

        # --- REPARAMETERIZATION NOISE (Standard Normal Buffer) ---
        # Fixed buffer for the "epsilon" in the reparameterization trick
        # We make it large enough for a batch (e.g., M samples)
        # Shape: (1, Latents, Mixtures, Dim) for broadcasting
        self.register_buffer(
            "fixed_epsilon", 
            torch.randn(1, num_latents, spectral_clusters, input_dim)
        )

        #--RFF Constants - fixed noise buffers--#
        #-draw standard normal once and freeze (fundamental random fourier projection assumption)--#
        self.register_buffer("noise_weights", torch.randn(self.K, self.M, input_dim)) #-Shape: [K, M, D]-#
        self.register_buffer("noise_bias", torch.rand(self.K, self.M) * 2 * math.pi)

        # ---------------------------------------------------------
        # C) Concentration (Gamma)
        # ---------------------------------------------------------
        #-- Learnable scalar for dirichlet process concentration (gamma)-#
        self.gamma_init = 2.0
        self.gamma = nn.Parameter(torch.tensor(float(self.gamma_init)))

        self.register_parameter(name="raw_logits", parameter=nn.Parameter(torch.randn(30)))

        self.register_prior(
            "logit_prior",
            NormalPrior(loc=0.0, scale=1.0),
            lambda m: F.softplus(m.raw_logits)
        )

        self.register_prior("gamma_prior", GammaPrior(2.5, 1.0), lambda m: F.softplus(m.gamma), lambda m, v: None)

        #--Define torch stickbreak module-#
        self.stick_break_transform = StickBreakingTransform()

        super().__init__()
    
    def _init_weights(self):
        # 1. Linear Primitive (Identity)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        
        # 2. Periodic Primitive (Cosine) -> gain = sqrt(2)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        
        # 3. RBF Primitive (Gaussian) - centres of rbfs
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        
        # 4. Rational Primitive (Cauchy)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)

        for module in self.primitive_kernel_heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)

        for module in self.bottleneck_mixer.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
        


    def dynamic_random_fourier_features(self, z: torch.Tensor, omega: torch.Tensor, pi: Optional[torch.Tensor]=None, ls: Optional[torch.Tensor]=None):
        """
        Args:
            z: [Batch, D]
            omega: [Batch, K, M, D] (dynamic) - generated by get_omega
            pi: [B, K] 
        """
        B, D = z.shape
        if ls is not None:
            z = z / (ls + 1e-6) #-kernel smoothing-#
        
        if pi is None:
            pi = torch.full((B, self.K), 1.0/self.K, device=z.device)
            pi = pi + (torch.randn_like(pi) * 0.01)
            pi = F.softmax(pi, dim=-1)
        
        #-determine if omega is dynamic-#
        if omega.dim() == 4:
            #-Dynamic: [B, K, M, D] * [B, 1, 1, D] -> Sum over D -#
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) #-dynamic-# z -> [B,1 ,1, D]
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W) #-static-#
            proj = proj.view(B, self.K, self.M)
        
        #-phase shift-#
        weights = self.noise_bias
        proj = proj + weights.view(1, self.K, self.M) #- [K, M] -> [1, K, M]
        
        #-scale for vae-#
        scale = 1.0 / math.sqrt(self.M)
        
        #-harmonics-#
        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale

        #-Mixing weights (amortised)-#
        #-unsqueeze [B, K] -> [B, K, 1]
        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl
        
        feats = torch.stack([cos_proj, sin_proj], dim=-1) #- [B, K, M] -> [B, K, M, 2]

        return feats.flatten(1) #-[B, (K * M * 2)]
    
    def get_omega(self, ls_pred=None):
        """
        Generates Variational Spectral Frequencies (Omega).
        
        Logic: Omega ~ q(omega | ls_pred) = N(mu, sigma_variational * ls_modifier)
        
        Args:
            ls_pred: [Batch, Latents, D] (Optional) - Amortised lengthscales from VAE.
                     If None, uses purely the learned variational variance.
        Returns:
            omega: [Batch, Latents, Mixtures, D]
        """
        
        # 1. Get Variational Parameters
        # mu shape: [Latents, Mixtures, D]
        mu = self.variational_omega_mu
        
        # sigma shape: [Latents, Mixtures, D]
        sigma = F.softplus(self.variational_omega_logvar) + 1e-6

        # 2. Apply Amortized Scaling (Optional)
        # If the VAE says "High Lengthscale", we should SHRINK the frequency variance.
        # Logic: bandwidth ~ 1 / lengthscale
        
        if ls_pred is not None:
            # ls_pred: [Batch, Latents, D]
            # We need to broadcast it to [Batch, Latents, 1, D] to match Mixtures
            ls_inv = 1.0 / (ls_pred.unsqueeze(2) + 1e-6)
            
            # Combine Variational Sigma with Dynamic Lengthscale
            # [1, L, M, D] * [B, L, 1, D] -> [B, L, M, D]
            dynamic_sigma = sigma.unsqueeze(0) * ls_inv
            
            # Expand mu for batch: [1, L, M, D] -> [B, L, M, D]
            # (Or let broadcasting handle it in the next step)
            dynamic_mu = mu.unsqueeze(0)
            
        else:
            # Static Mode (Standard Variational GP)
            dynamic_sigma = sigma.unsqueeze(0) # [1, L, M, D]
            dynamic_mu = mu.unsqueeze(0)       # [1, L, M, D]
        #-Stochastic Sampling for Training
        epsilon = torch.randn_like(dynamic_sigma) 

        omega = dynamic_mu + (dynamic_sigma * epsilon)
        
        return omega # [Batch, Latents, Mixtures, Dim]
    
    def kernel_network(self, z):

    def forward(self, 
            x,
            q_dist, #-low rank mvn-#
            batch_shape=torch.Size([]), 
            ls: Optional[torch.Tensor]=None, 
            alpha: Optional[torch.Tensor]=None,):
        """
        Args:
            z (referenced as x for gpytorch): [Batch, D] - Latent (Required for features)
            alpha_amortised: [Batch, K] - Dirichlet Concentration (from Encoder)
            ls_amortised: [Batch, K, D] - Lengthscales (from Encoder or Override)
        """
        
        # ---------------------------------------------------------
        # 1. Global Variational Inference (Stick Breaking)
        # ---------------------------------------------------------
        #--A) define variational posterior - q(v)-#
        
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
        global_divergence = (log_qv - log_pv).sum()

        self.update_added_loss_term("global_divergence", SimpleLoss(global_divergence))

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
        if x is not None:
            batch_size = x.size(0)
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
            pr_dist_pi = Dirichlet(prior_conc_expanded) #-[B, K]
            post_dist_pi = Dirichlet(post_conc) #-- [B, K]
            local_divergence = torch.distributions.kl_divergence(post_dist_pi, pr_dist_pi)
            self.update_added_loss_term("local_divergence", SimpleLoss(local_divergence.sum()))
            
            if ls is not None:
                ls_pred = torch.clamp(ls, min=self.eps, max=50.0) #- [B, K, D]
                target_ls = torch.tensor(1.0, device=ls_pred.device)
                ls_log_mse = F.mse_loss(torch.log(ls_pred), torch.log(target_ls.expand_as(ls_pred)))
                self.update_added_loss_term("spectral_mse_loss", SimpleLoss(ls_log_mse))
        else:
            # --- GENERATIVE MODE ---
            # No evidence provided. We sample from the Prior.
            post_conc = prior_conc_expanded #-kl will be 0-#

        #--Sample from pi (dynamic)-#
        pi = Dirichlet(post_conc).rsample() #--[B, K]-#

        omega_frequencies = self.get_omega(ls_pred) #-[B, K, M, D]-#

        features = self.dynamic_random_fourier_features(x, omega_frequencies)

        kernel_features_projection = self.compress_spectral_features_head(features)
        
        return kernel_features_projection, omega_frequencies, pi, ls_pred