import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
import math
import logging
import gpytorch
from gpytorch.priors import NormalPrior, GammaPrior
from typing import Optional

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.losses.simple import SimpleLoss
from deepkernels.models.NKN import KernelNetwork, KernelNetworkOutput, GPParams
from pydantic import BaseModel
import torch.distributions as dist

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from typing import NamedTuple

class DirichletOutput(NamedTuple):
    features: torch.Tensor
    frequencies: torch.Tensor
    gated_weights: torch.Tensor
    ls_pred: torch.Tensor
    bw_learned: torch.Tensor
    z_in: torch.Tensor
    bottleneck: torch.Tensor
    beta: torch.Tensor
    pi: torch.Tensor
    conc_prior: torch.Tensor
    conc_post: torch.Tensor
    ls_logvar: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    gp_params: GPParams

class LossTerm(gpytorch.mlls.AddedLossTerm):
    """
    A concrete implementation of an AddedLossTerm that simply 
    returns a pre-calculated scalar tensor.
    """
    def __init__(self, loss_tensor):
        self.loss_tensor = loss_tensor
        
    def loss(self):
        return self.loss_tensor

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
        self.atom_log_sigma = nn.Parameter(torch.randn(k_atoms, 1, latent_dim) * 0.025)
        self.atom_mu = nn.Parameter(torch.randn(k_atoms, 1, latent_dim) * 2 * math.sqrt(0.025))
        self.raw_gamma = nn.Parameter(torch.tensor((self.numerically_stable_gamma(gamma_concentration_init))))
        self.lengthscale_log_uncertainty = nn.Parameter(torch.zeros(1, k_atoms))

        #-priors-#
        self.register_prior(
            "h_scale_prior", 
            NormalPrior(loc=3.0, scale=1.0), 
            lambda m: m.h_log_sigma, 
            lambda m, v: None
        )

        self.register_prior(
            "atom_log_sigma_prior",
            NormalPrior(loc=0.0, scale=0.025),
            lambda m: m.atom_log_sigma,
            lambda m, v: None
        )

        self.register_prior(
            "gamma_prior", 
            GammaPrior(concentration=1.5, rate=0.5), 
            lambda m: F.softplus(m.raw_gamma),
            lambda m, v: None
        )

        #-buffers-#
        self.register_buffer("noise_weights", torch.randn(k_atoms, fourier_dim, latent_dim))
        self.register_buffer("noise_bias", torch.rand(k_atoms, fourier_dim))

        #-loss terms-#
        self.register_added_loss_term("global_divergence")
        self.register_added_loss_term("local_divergence")
        

    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params) -> DirichletOutput:
        """
        performs nonparametric clustering according to a hierarchical dirichlet process using learned lengthscale
        and concentration param refinement via pi (logit mixture weights)
        Args:
            latent z (param: x) -- dim 16
        """
        
        pi = params.get('pi')
        if pi is None and vae_out is not None:
            pi = getattr(vae_out, 'pi', None)
        
        if isinstance(pi, torch.Tensor) and pi.numel() == 0:
            pi = None
        
        mualpha, factoralpha, diagalpha = vae_out.alpha_mu, vae_out.alpha_factor, vae_out.alpha_diag
        mu_z, logvar_z = vae_out.mu_z, vae_out.logvar_z
        
        ls = params.get('ls')
        if ls is None and vae_out is not None:
            ls = getattr(vae_out, 'ls', None)
        
        if isinstance(ls, torch.Tensor) and ls.numel() == 0:
            ls = None
        
        beta, log_pv, log_qv, gamma_conc = self.global_stick_breaking()

        self.log_global_kl(log_pv, log_qv)
    
        bottleneck, gate, gp_params = self.run_neural_nets_dirichlet(x)
        
        local_conc = self.get_local_evidence(mualpha, factoralpha, diagalpha)

        pi = self.dirichlet_posterior_inference_and_log_local_loss(x, gamma_conc, beta, local_conc)
        
        ls_pred, bw_learned, ls_logvar = self.predict_kernel_lengthscales(ls)
        
        omega = self.get_omega(bw_learned)

        raw_features = self.random_fourier_features(x, omega, pi)

        gated_features = self.compress_and_gate(raw_features, gate)
        
        return DirichletOutput(
            features=gated_features,
            frequencies=omega,
            gated_weights=gate,
            ls_pred=ls_pred,
            bw_learned=bw_learned,
            z_in=x,
            bottleneck=bottleneck,
            beta=beta,
            pi=pi,
            conc_prior=gamma_conc,
            conc_post=local_conc,
            ls_logvar=ls_logvar,
            gp_params=gp_params,
            mu_z=mu_z,
            logvar_z=logvar_z
        )
    
    def log_global_kl(self, log_pv, log_qv):
        self.update_added_loss_term("global_divergence", LossTerm(log_qv - log_pv))
    
    def numerically_stable_gamma(self, gamma_concentration_init):
        raw = float(gamma_concentration_init)
        safe = math.log(math.exp(raw) - 1)
        return safe
    
    def get_omega(self, bw, k_atoms=30, fourier_dim=128, latent_dim=16, **params):
        # Broadcasting: [1, 1, D] + [K, 1, D] + ([K, M, D] * [B, K, 1, D])
        noise_weights = self.noise_weights
        omega = self.h_mu + self.atom_mu + noise_weights.unsqueeze(0) * bw
        return torch.clamp(omega, -100.0, 100.0)
    
    def random_fourier_features(self, z, omega, pi, k_atoms=30, M=128, latent_dim=16, **params):
        """inputs latent dim z"""
        noise_bias = self.noise_bias
        B, D = z.shape
        if pi is None:
            pi = torch.full((B, k_atoms), 1.0/k_atoms, device=z.device)
            if self.training:
                pi = pi + (torch.randn_like(pi) * 0.01)
            pi = F.softmax(pi, dim=-1)
        if omega.dim() == 4:
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) 
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W).view(B, k_atoms, M)
        
        proj = proj + noise_bias.unsqueeze(0)
        scale = 1.0 / math.sqrt(M)

        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale

        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl

        feats = torch.stack([cos_proj, sin_proj], dim=-1)
        return feats.flatten(1)
    
    def global_stick_breaking(self, k_atoms=30, **params):
        #-variational inference-#
        q_sig_global = self.apply_softplus(self.q_log_sigma_global)
        q_dist_global = torch.distributions.Normal(self.q_mu_global, q_sig_global)
        qz_global = q_dist_global.rsample()

        #-jacobian // entropy-#
        log_detj = -F.softplus(-qz_global) - F.softplus(qz_global)
        log_qv = q_dist_global.log_prob(qz_global).sum() - log_detj.sum()
        
        #-prior log likelihood-#
        gamma_conc = self.apply_softplus(self.raw_gamma)
        log_pv = (torch.log(gamma_conc + 1e-3) + (gamma_conc - 1) * (-F.softplus(qz_global))).sum()
        
        #-stick breaking logic-#
        qv_global = torch.sigmoid(qz_global)
        one_minus_v = 1 - qv_global
        cumprod_one_minus_v = torch.cumprod(one_minus_v, dim=-1)

        pad = torch.ones_like(cumprod_one_minus_v[..., :1])
        previous_remaining = torch.cat([pad, cumprod_one_minus_v[..., :-1]], dim=-1)

        beta_k = qv_global * previous_remaining
        beta_last = cumprod_one_minus_v[..., -1:]
        beta = torch.cat([beta_k, beta_last], dim=-1)
        
        return beta, log_pv, log_qv, gamma_conc
    
    def predict_kernel_lengthscales(self, ls, vae_out: Optional[dict]=None, eps=1e-3, max_ls=100.0, k_atoms=30, latent_dim=16, **params):
        
        sigmas = self.h_log_sigma + self.atom_log_sigma 
        log_scale = self.apply_softplus(sigmas) if hasattr(self, 'apply_softplus') else F.softplus(sigmas)
        bw_base = log_scale.exp()
        
        if ls is None:
            ls_pred = bw_base.squeeze(1).mean(dim=-1).unsqueeze(0) 
            ls_pred = torch.clamp(ls_pred, min=eps, max=max_ls)
            ls_logvar = torch.zeros(1, k_atoms, device=ls_pred.device)
            bw_learned = bw_base.unsqueeze(0) 
        else:
            ls_pred = torch.clamp(ls, min=eps, max=max_ls)
            batch_size = ls_pred.size(0)
            precision = 1.0 / (ls_pred.view(batch_size, k_atoms, 1, 1) + eps)
            bw_learned = bw_base.unsqueeze(0) * precision
            ls_logvar = self.lengthscale_log_uncertainty.expand(batch_size, -1)

        return ls_pred, bw_learned, ls_logvar
    
    def run_neural_nets_dirichlet(self, x):
        bottleneck = self.bottleneck_mixer(x) #-takes latent z[B,16] -> [B,64]
        nkn_out = self.kernel_network(bottleneck)
        return bottleneck, nkn_out.dirichlet_features, nkn_out.gp_params
    
    def compress_and_gate(self, features, gate):
        embedded_features = self.compress_spectral_features_head(features)
        return gate * embedded_features
    
    def dirichlet_posterior_inference_and_log_local_loss(self, x, gamma_conc, beta, local_conc, eps=4e-2):
        prior_conc = (gamma_conc * beta) + eps
        prior_conc = torch.clamp(prior_conc, min=eps)
        prior_conc = prior_conc.unsqueeze(0).expand(x.size(0), -1)

        post_conc = prior_conc + local_conc
        post_conc = torch.clamp(post_conc, min=eps)

        dist_prior = dist.Dirichlet(prior_conc)
        dist_post = dist.Dirichlet(post_conc)
        pi_posterior = dist_post.rsample()
        local_divergence = torch.distributions.kl_divergence(dist_post, dist_prior)
        self.update_added_loss_term("local_divergence", LossTerm(local_divergence.sum()))
        
        return pi_posterior
    
    def get_local_evidence(self, mualpha, factoralpha, diagalpha):
        alpha_logits = self.lowrankmultivariatenorm(mualpha, factoralpha, diagalpha)
        local_conc = self.apply_softplus(alpha_logits)
        return local_conc
    
    
   

