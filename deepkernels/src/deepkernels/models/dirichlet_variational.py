import torch
import gpytorch
from torch.distributions import Normal, TransformedDistribution, kl_divergence, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
import math
from src.models.model_config import RootConfig
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional
from src.losses.kl_divergence import KLDivergence



class HDPConfig(BaseModel):
    K: int = 30
    M: int = 512
    D: int = 128
    eps: float = 1e-3
    gamma_prior: float = 2.0
    mu_init: float = 0.0
    sigma_init: float = 1.0
    n_tasks: int = 128
    num_inducing: int = 1024


class VariationalDirichlet(gpytorch.Module):
    def __init__(self, config:HDPConfig):
        super().__init__()
        self.config = config
        self.K = self.config.K
        self.M = self.config.M #-rff samples per mixture-#
        self.D = self.config.D #-input dim from vit-#
        self.eps = self.config.eps

        #--init constraints--#
        self.mu_init = self.config.mu_init
        self.sigma_init = self.config.sigma_init
        self.log_sigma_init = math.log(self.sigma_init)

        # ---------------------------------------------------------
        # Structure: GEM Global Mixing Weights (Beta)
        # ---------------------------------------------------------
        #-- variational params - q(v) ~ N(q_mu, q_log_sigma)
        self.q_mu = nn.Parameter(torch.randn(self.K - 1)) #--break symmetry-#
        self.q_log_sigma = nn.Parameter(torch.full((self.K - 1,), -2.0)) #-cluster stability-#

        #-- fixed prior params -- regularisation anchors-#
        #---uniform on simplex ~ N(0, 1)--#
        self.register_buffer("prior_mu", torch.full((self.K - 1,), self.mu_init))
        self.register_buffer("prior_log_sigma", torch.full((self.K - 1,), self.log_sigma_init)) #-anchor strength-#

        # ---------------------------------------------------------
        # Content: Spectral Frequencies (Omega)
        # ---------------------------------------------------------
        #--spectral mixture kernel: defined by weight (pi), center (atom_mu), width (atom_log_scale)

        #--Global Base Distribution (H) defines kernel smoothness & freq
        self.h_mu = nn.Parameter(torch.zeros(1, 1, self.D)) #-center of kernels -- mu prior on gram matrix-#
        self.h_log_sigma = nn.Parameter(torch.tensor(-2.0)) #-global bandwidth-#

        #-local atom deviation (additive)-#
        self.atom_log_sigma = nn.Parameter(torch.randn(self.K, 1, self.D) * 0.1)
        self.atom_mu = nn.Parameter(torch.randn(self.K, 1, self.D))

        #--RFF Constants - fixed noise params--#
        #-draw standard normal once and freeze (fundamental random fourier projection assumption)--#
        self.register_buffer("noise_weights", torch.randn(self.K, self.M, self.D)) #-Shape: [K, M, D]-#
        self.register_buffer("noise_bias", torch.rand(self.K, self.M) * 2 * math.pi)


        # ---------------------------------------------------------
        # C) Amortized Inference (pi_encoder)
        # ---------------------------------------------------------
        #--Data Encoder for local mixing weights-#
        self.pi_encoder = nn.Sequential(
            nn.Linear(self.D, 2*self.D),
            nn.LayerNorm(2*self.D),
            nn.Tanh(),
            nn.Linear(2 * self.D, self.K),
            nn.Softplus()
        )

        # ---------------------------------------------------------
        # D) Concentration (Gamma)
        # ---------------------------------------------------------
        #-- Learnable scalar for dirichlet process concentration (gamma)-#
        self.gamma_init = getattr(self.config, 'gamma_prior', 2.0)
        self.gamma = nn.Parameter(torch.tensor(float(self.gamma_init)))

        #--Define torch stickbreak module-#
        self.stick_break_transform = torch.distributions.transforms.StickBreakingTransform()
    
    def _random_fourier_features(self, z, omega, pi):
        B, D = z.shape
        K, M, _ = omega.shape

        W = omega.view(-1, D)

        #-project to frequencies (dot product)-#
        proj = F.linear(z, W)

        #-harmonics + reshape for weights-#
        cos_proj = (torch.cos(proj)).view(B, K, M)
        sin_proj = (torch.sin(proj)).view(B, K, M)

        #-Mixing weights (amortised)-#
        #-unsqueeze [B, K] -> [B, K, 1] to broadcast over inducing points-#
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

        return feats

    def forward(self, z: Optional[torch.Tensor]=None, batch_shape=torch.Size([]), rff_only=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #--A) define variational posterior - q(v)-#
        q_sig = F.softplus(self.q_log_sigma)
        q_dist = Normal(self.q_mu, q_sig)
        #-prior p(v)-#
        p_sig = F.softplus(self.prior_log_sigma)
        p_dist = Normal(self.prior_mu, p_sig)

        #--B) register loss--#
        self.update_added_loss_term("hdp_kl", KLDivergence(q_dist, p_dist))

        #--C) Global weights (beta)-#
        #-q(beta) = GEM(q(v))
        simplex_dist = TransformedDistribution(q_dist, [self.stick_break_transform])
        beta = simplex_dist.rsample() #-shape:[K]

        #--D) Dynamic local weights (pi)--#
        gamma = F.softplus(self.gamma)

        prior_conc = (gamma * beta) + self.eps
        prior_conc = torch.clamp(prior_conc, min=self.eps, max=150.0)

        if z is None:
            if len(batch_shape) > 0:
                post_conc = prior_conc.unsqueeze(0).expand(*batch_shape, -1)
            else:
                post_conc = prior_conc
        else:
            #--amortised inference in training-#
            #-Data evidence from neural net (pi_encoder)-#
            #- z: [B, D] -> data_conc: [B, K]-#
            local_conc = self.pi_encoder(z) + self.eps #-[B, K]-#
            #-Posterior Concentration (global belief + local evidence)-
            post_conc = prior_conc.unsqueeze(0) + local_conc
            post_conc = torch.clamp(post_conc, min=self.eps, max=150.0)
            #--kl loss for only when data is present-#
            #-Create Distribution objects for kl loss updates-#
            pr_dist_pi = Dirichlet(prior_conc) #-KL(Post|Pr) --- Batch: [], Event: [K]--#
            post_dist_pi = Dirichlet(post_conc) #--Batch: [], Event: [K]--#
            #-Register KL loss for weights-#
            self.update_added_loss_term("pi_kl", KLDivergence(post_dist_pi, pr_dist_pi))
        
        #--Sample from pi (dynamic)-#
        pi = Dirichlet(post_conc).rsample() #--[B, K]-#

        #--E) Construct Spectal frequencies with H regularisation-#
        #-shift by h_mu, scale by h_log_sigma
        global_log_scale = self.h_log_sigma #-base global freq-#
        local_log_scale = self.atom_log_sigma #-deviation locally--shape:[K, 1, D] #
        log_scale = global_log_scale + local_log_scale
        log_scale = torch.clamp(log_scale, max=10.0)
        bandwidth = log_scale.exp() #-inv lengthscale-

        #--centres: global center + local frequency/position-#
        spectral_means = self.h_mu + self.atom_mu

        #--reparameterisation trick-#
        #-omega ~ N(spectral_means, bandwidth)
        #- omega shape: - [K, M, D]-#
        omega = spectral_means + (self.noise_weights * bandwidth)
        omega = torch.clamp(omega, -150.0, 150.0)

        if rff_only and z is not None:
            #-returns amortised output in rff-#
            return self._random_fourier_features(z, omega, pi)
        
        else:
            #--Returns:
            #pi: [B, K] -> local mixing weights
            #beta: [K] -> global prevalence
            #omega: [K, M, D] -> spectral frequencies
            #bias: [K, M] -> spectral phases (fixed)
            return pi, beta, omega, self.noise_bias