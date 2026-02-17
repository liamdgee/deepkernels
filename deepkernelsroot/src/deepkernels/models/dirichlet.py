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
    def __init__(self, config=None, k_atoms=30, fourier_dim=128, latent_dim=16, spectral_emb_dim=2048, num_latents=8, input_dim=30, bottleneck_dim=64, gamma_concentration_init=2.5):
        super().__init__() # Call super first
        self.config = config or HDPConfig()
        self.K = k_atoms or self.config.K
        self.M = fourier_dim or self.config.M 
        self.D = latent_dim or self.config.D 
        self.eps = 1e-3

        # 1. Feature Compressors
        self.compress_spectral_features_head = torch.nn.utils.spectral_norm(nn.Linear(k_atoms * fourier_dim * 2, spectral_emb_dim))
        
        self.bottleneck_mixer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, bottleneck_dim)),
            nn.LayerNorm(bottleneck_dim),
            nn.Tanh()
        )

        # 2. Neural Kernel Network & Bridge
        self.kernel_network = KernelNetwork()
        # FIX: Bridge NKN output (256) to Spectral Dim (2048)
        self.nkn_bridge = nn.Sequential(
            nn.Linear(256, spectral_emb_dim), # Assuming NKN outputs 256 (8*32)
            nn.Sigmoid() # Gate 0-1
        )

        # 3. Global HDP Parameters
        self.register_parameter(
            "q_mu_global", 
            nn.Parameter(torch.zeros(self.K - 1)) 
        )
        self.register_parameter(
            "q_log_sigma_global", 
            nn.Parameter(torch.ones(self.K - 1) * -4.0) # Efficient Vector init
        )

        # 4. Spectral Parameters
        self.h_mu = nn.Parameter(torch.zeros(1, 1, latent_dim)) 
        self.h_log_sigma = nn.Parameter(torch.tensor(3.0)) 

        self.atom_log_sigma = nn.Parameter(torch.randn(self.K, 1, latent_dim) * 0.01)
        self.atom_mu = nn.Parameter(torch.randn(self.K, 1, latent_dim) * 2 * math.sqrt(0.01))

        self.register_buffer("noise_weights", torch.randn(self.K, fourier_dim, latent_dim))
        self.register_buffer("noise_bias", torch.rand(self.K, fourier_dim))

        # 5. Gamma Parameters
        target_value = float(gamma_concentration_init)
        inv_softplus_gamma = math.log(math.exp(target_value) - 1)
        self.register_parameter(
            name = "raw_gamma",
            parameter = nn.Parameter(torch.tensor(inv_softplus_gamma))
        )

    def forward(self, x, encoder_out, ls, batch_shape=torch.Size([])):     
        # ---------------------------------------------------------
        # 1. Global Variational Inference (Stick Breaking)
        # ---------------------------------------------------------
        q_sig_global = F.softplus(self.q_log_sigma_global) # Vector op is fine
        q_dist_global = Normal(self.q_mu_global, q_sig_global)
        
        qz_global = q_dist_global.rsample()
        
        # GLOBAL KL
        log_detj = -F.softplus(-qz_global) - F.softplus(qz_global)
        log_qv = q_dist_global.log_prob(qz_global).sum() - log_detj.sum()
        
        gamma_conc = self.apply_softplus(self.raw_gamma) # FIX: Pass tensor, not string
        log_pv = (torch.log(gamma_conc + self.eps) + (gamma_conc - 1) * (-F.softplus(qz_global))).sum()
        
        self.update_added_loss_term("global_divergence", SimpleLoss(log_qv - log_pv))

        # Compute Global Weights (Beta)
        qv_global = torch.sigmoid(qz_global)
        one_minus_v = 1 - qv_global
        cumprod_one_minus_v = torch.cumprod(one_minus_v, dim=-1)
        previous_remaining = torch.roll(cumprod_one_minus_v, 1, dims=-1)
        previous_remaining[..., 0] = 1.0
        beta_k = qv_global * previous_remaining
        beta_last = cumprod_one_minus_v[..., -1:]
        beta = torch.cat([beta_k, beta_last], dim=-1) 

        # ---------------------------------------------------------
        # 2. Local Amortized Inference
        # ---------------------------------------------------------
        bottleneck = self.bottleneck_mixer(x)

        # NKN Structure Learning
        raw_nkn_out = self.kernel_network(bottleneck) # [B, 256]
        nkn_gate = self.nkn_bridge(raw_nkn_out)       # [B, 2048] FIX: Dimension Match
        
        # Get Local Dirichlet Evidence
        _, mualpha, cholalpha, diagalpha, _ = encoder_out['alpha_params']
        alpha_logits = self.lowrankmultivariatenorm(mualpha, cholalpha, diagalpha)
        local_conc = F.softplus(alpha_logits) + 1e-6

        # ---------------------------------------------------------
        # 3. HDP Posterior
        # ---------------------------------------------------------
        prior_conc = (gamma_conc * beta) + self.eps
        prior_conc = torch.clamp(prior_conc, min=1e-2, max=100.0)
        prior_conc = prior_conc.unsqueeze(0).expand(x.size(0), -1)

        post_conc = prior_conc + local_conc
        post_conc = torch.clamp(post_conc, min=self.eps, max=100.0)
        
        # LOCAL KL (Exact Dirichlet)
        dist_prior = Dirichlet(prior_conc)
        dist_post = Dirichlet(post_conc)
        local_divergence = torch.distributions.kl_divergence(dist_post, dist_prior)
        self.update_added_loss_term("local_divergence", SimpleLoss(local_divergence.sum()))

        pi = dist_post.rsample()

        # ---------------------------------------------------------
        # 4. Generate Spectral Features
        # ---------------------------------------------------------
        # Bandwidths
        log_scale = F.softplus(self.h_log_sigma + self.atom_log_sigma) 
        bw_base = log_scale.exp() 
        
        # Lengthscale Scaling
        if ls is not None:
            ls_pred = torch.clamp(ls, min=self.eps, max=50.0)
            ls_mse = F.mse_loss(ls_pred, torch.ones_like(ls_pred))
            self.update_added_loss_term("lengthscale_prior_reg", SimpleLoss(ls_mse))
            
            bw_learned = bw_base.unsqueeze(0) * (1.0 / (ls_pred.unsqueeze(2) + 1e-6))
        else:
            ls_pred = None
            bw_learned = bw_base.unsqueeze(0)

        # Omega & Features
        omega = self.get_omega(bw_learned)
        raw_features = self.dynamic_random_fourier_features(x, omega, pi)
        projected_features = self.compress_spectral_features_head(raw_features) # [B, 2048]

        # ---------------------------------------------------------
        # 5. FUSE
        # ---------------------------------------------------------
        # Gate the spectral features with the NKN structure
        gated_features = projected_features * nkn_gate
        
        return gated_features, omega, pi, ls_pred

    # --- Helper Methods Indented correctly inside class ---
    
    def dynamic_random_fourier_features(self, z, omega, pi=None):
        B, D = z.shape
        if pi is None:
            pi = torch.full((B, self.K), 1.0/self.K, device=z.device)
            pi = F.softmax(pi, dim=-1)
        
        # Determine if omega is dynamic
        if omega.dim() == 4:
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) 
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W).view(B, self.K, self.M)
        
        proj = proj + self.noise_bias.unsqueeze(0)
        scale = 1.0 / math.sqrt(self.M)
        
        # Harmonics
        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale

        # Mixing
        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl
        
        feats = torch.stack([cos_proj, sin_proj], dim=-1)
        return feats.flatten(1) 
        
    def get_omega(self, bw):
        # bw: [B, K, M, D]
        # Broadcasting: [1, 1, D] + [K, 1, D] + ([K, M, D] * [B, K, 1, D])
        # Note: self.noise_weights is [K, M, D]. 
        # To broadcast with [B, K, 1, D], we need noise to be [1, K, M, D]
        omega = self.h_mu + self.atom_mu + (self.noise_weights.unsqueeze(0) * bw)
        return torch.clamp(omega, -100.0, 100.0)

    def lowrankmultivariatenorm(self, mu, factor, diag):
        mvn = torch.distributions.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        logits = mvn.rsample()
        return logits # Return logits, softplus them later
    
    def apply_softplus(self, x, jitter=1e-6):
        return torch.nn.functional.softplus(x) + jitter