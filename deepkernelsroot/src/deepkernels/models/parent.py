import torch
import gpytorch
from gpytorch.mlls import AddedLossTerm
from gpytorch.priors import NormalPrior, GammaPrior, HorseshoePrior
import torch.distributions as dist
import torch.nn.functional as F
from deepkernels.losses.simple import SimpleLoss
import torch.distributions as dist
import math
import logging
from typing import Union, Optional, Dict, Tuple, TypeAlias, Union

import torch
from torch.distributions import LowRankMultivariateNormal, Independent, Normal, kl_divergence

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseGenerativeModel(gpytorch.Module):
    def __init__(self):
        super().__init__()
    
    def register_constrained_parameter(self, name, parameter, constraint):
        self.register_parameter(name, parameter)
        self.register_constraint(name, constraint)
        return self
    
    def register_priors_for_dirichlet(self):
        if hasattr(self, "gamma"):
            self.register_prior("gamma_prior", GammaPrior(2.5, 1.0), lambda m: F.softplus(m.gamma), lambda m, v: None)
        
        if hasattr(self, "raw_logits"):
            self.register_prior("logit_prior", NormalPrior(loc=0.0, scale=1.0),lambda m: F.softplus(m.raw_logits))
    
    
    def register_kernel_priors(self):
        if hasattr(self, "covar_module"):
            self.register_prior("sparsity_prior", HorseshoePrior(scale=0.1), lambda m: m.raw_inv_bandwidth)

    def log_loss(self, name, value):
        """
        Wraps the raw tensor in an AddedLossTerm and updates it.
        usage: self.log_loss("reconstruction_loss", recon_tensor)
        """
        if not hasattr(self, "_added_loss_terms") or name not in self._added_loss_terms:
             raise RuntimeError(f"Loss term '{name}' not registered in Base __init__")
        scalar_loss = value.sum() if value.dim() > 0 else value
        self.update_added_loss_term(name, SimpleLoss(scalar_loss))

    
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        raise NotImplementedError("Subclass must implement forward()")
        
    def get_variational_strategy(self):
        raise NotImplementedError("Get strategy from subclass: model or orchestrate")
    
    def multivariate_projection(self, mu, factor, diag, jitter=1e-6):
        """for alpha params: input projections from three alpha heads"""
        mvn = dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=factor,
            cov_diag=diag
        )

        logits = mvn.rsample()
        alpha = torch.nn.functional.softplus(logits) + jitter
        
        return alpha

    def get_device(self, device_request: Union[str, torch.device, None] = None) -> torch.device:

        """
        Resolves the optimal available device for PyTorch operations.
        
        Priority:
        1. explicit device_request (if provided and valid)
        2. cuda:0 (NVIDIA GPU)
        3. mps (Apple Silicon Metal Performance Shaders)
        4. cpu
        
        Args:
            device_request: Optional string ('cuda', 'mps', 'cpu') or torch.device 
                            to force a specific device.
        
        Returns:
            torch.device: The resolved device.
        """
        if device_request is not None:
            device = torch.device(device_request)
            if device.type == 'cuda' and not torch.cuda.is_available():
                logging.warning(f"CUDA requested but unavailable. Falling back to CPU.")
                return torch.device('cpu')
            if device.type == 'mps' and not torch.backends.mps.is_available():
                logging.warning(f"MPS (Apple Silicon) requested but unavailable. Falling back to CPU.")
                return torch.device('cpu')
            return device
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        return torch.device('cpu')
    
    def lowrankmultivariatenorm(self, mu, factor, diag):
        mvn = torch.distributions.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        logits = mvn.rsample()
        return logits # Return logits, softplus them later
    
    def apply_softplus(self, x, jitter=1e-6):
        return torch.nn.functional.softplus(x) + jitter
    
    
    def stack_features(self, latent_kernels):
        return torch.stack(latent_kernels)
    
    def get_resource(self, name_string, **params):
        return getattr(self, name_string, None)
    
    def numerically_stable_gamma(self, gamma_concentration_init):
        raw = float(gamma_concentration_init)
        safe = math.log(math.exp(raw) - 1)
        return safe
        
    def log_global_kl(self, log_pv, log_qv):
        self.update_added_loss_term("global_divergence", SimpleLoss(log_qv - log_pv))
    
    def dirichlet_posterior_inference_and_log_local_loss(self, x, gamma_conc, beta, local_conc, eps=1e-3):
        prior_conc = (gamma_conc * beta) + eps
        prior_conc = torch.clamp(prior_conc, min=eps)
        prior_conc = prior_conc.unsqueeze(0).expand(x.size(0), -1)

        post_conc = prior_conc + local_conc
        post_conc = torch.clamp(post_conc, min=eps)

        dist_prior = dist.Dirichlet(prior_conc)
        dist_post = dist.Dirichlet(post_conc)

        pi_posterior = dist_post.rsample()
        local_divergence = torch.distributions.kl_divergence(dist_post, dist_prior)
        self.update_added_loss_term("local_divergence", SimpleLoss(local_divergence.sum()))
        
        return pi_posterior
    
    def get_local_evidence(self, mualpha, factoralpha, diagalpha):
        alpha_logits = self.lowrankmultivariatenorm(mualpha, factoralpha, diagalpha)
        local_conc = self.apply_softplus(alpha_logits)
        return local_conc
    
    
    @staticmethod
    def init_inducing_with_omega(
        omega: torch.Tensor, 
        n_inducing: int, 
        latent_dim: int = 16,
        pi: Optional[torch.Tensor] = None
    ):
        """
        Initializes inducing points directly in the spectral feature space 
        by projecting base latent points through the given frequencies.

        Mirrors logic in the original projection
        
        Args:
            omega: [K, M, D] or [1, K, M, D] tensor of raw frequencies.
            n_inducing: Number of inducing points (e.g., 128).
            latent_dim: The dimension of the latent space z.
            pi: Optional gating probabilities [n_inducing, K_atoms].
        Returns:
            inducing: [n_inducing, K * M * 2] tensor of inducing points.
        """
        device = omega.device
        
        z_init = torch.randn(n_inducing, latent_dim, device=device) * 1.5
        # omega shape expected to be: [K, M, D] or [B, K, M, D]
        if omega.dim() == 3: 
            proj = (z_init.view(n_inducing, 1, 1, latent_dim) * omega.unsqueeze(0)).sum(dim=-1)
        elif omega.dim() == 4:
            proj = (z_init.view(n_inducing, 1, 1, latent_dim) * omega).sum(dim=-1)
        else:
            raise ValueError(f"Expected omega to be 3D or 4D, got {omega.dim()}D")
            
        #-fourier mapping-#
        M = omega.size(-2)
        scale = 1.0 / math.sqrt(M)
        
        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale
        
        if pi is not None:
             pi_scl = torch.sqrt(pi).unsqueeze(-1) # [n_inducing, K, 1]
             cos_proj = cos_proj * pi_scl
             sin_proj = sin_proj * pi_scl
        else:
             # Uniform fallback if pi is omitted
             k_atoms = omega.size(-3)
             pi_scl = math.sqrt(1.0 / k_atoms)
             cos_proj = cos_proj * pi_scl
             sin_proj = sin_proj * pi_scl
             
        # 5. Stack and flatten to match feature_dim
        feats = torch.stack([cos_proj, sin_proj], dim=-1) # [n_inducing, K, M, 2]
        
        inducing = feats.flatten(1) # [n_inducing, K * M * 2]
        
        # Add a tiny bit of jitter to prevent singular matrices at initialization
        inducing_jitter = torch.randn_like(inducing) * 1e-4
        
        return inducing + inducing_jitter
    
    @staticmethod
    def init_inducing_with_fft(y_target, n_inducing, feature_dim):
        """
        Initializes inducing point values based on the FFT of the target signal.
        Args:
        y_target: [N_data] tensor of training targets used for initialization (Assuming somewhat evenly spaced or interpolated)
        n_inducing: Number of inducing points (must match model)
        M: Number of RFF components (M from dirichlet or encoder -- these will match)
        feature_dim: K clusters (30) * M fourier features (128) * 2 = 7680
        """
        #-fast fourier transform of target variable y-#
        if y_target is None:
            return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        
        yflat = y_target.flatten().cpu()
        n_points = len(yflat)
        
        if n_points == 0 or yflat.abs().sum() < 1e-6:
            return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        
        fourier_vals = torch.fft.rfft(yflat)
        freqs = torch.fft.rfftfreq(n_points)
        jitter = 1e-6
        eps = 1e-9
        #-Construct a Probability Distribution from FFT magnitudes to sample weights from-#
        density = torch.abs(fourier_vals)
        if density.shape[0] > 0:
            density[0] = 0 #-remove DC component-#
            cdf = density.sum()
        if cdf < eps:
            return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        p = density / cdf
        indices = torch.multinomial(p, feature_dim, replacement=True)
        samples = freqs[indices]
        binary_mask = torch.bernoulli(torch.full((feature_dim,), 0.5))
        plus_or_minus_one = 2 * binary_mask - 1
        sigma_y = y_target.std() + jitter
        sigma_samples = samples.std() + jitter

        sqrt_feature_dim_scale = sigma_y / (sigma_samples * math.sqrt(feature_dim))
        weights_flat = samples * plus_or_minus_one * sqrt_feature_dim_scale

        inducing = weights_flat.unsqueeze(0).repeat(n_inducing, 1)
        inducing_jitter = torch.randn_like(inducing) * (sqrt_feature_dim_scale * sigma_samples * 0.13)
        inducing = inducing + inducing_jitter
        return inducing