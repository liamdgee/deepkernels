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
from typing import Union, Optional, Dict, Tuple, TypeAlias

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
    
    def lowrankmultivariatenorm(self, mu, factor, diag):
        mvn = torch.distributions.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        logits = mvn.rsample()
        return logits # Return logits, softplus them later
    
    def apply_softplus(self, x, jitter=1e-6):
        return torch.nn.functional.softplus(x) + jitter
    
    def random_fourier_features(self, z, omega, pi, k_atoms=30, M=128, latent_dim=16, **params):
        """inputs latent dim z"""
        noise_bias = params.get("noise_bias", getattr(self, "noise_bias", torch.rand(k_atoms, M)))
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
        
    def get_omega(self, bw, k_atoms=30, fourier_dim=128, latent_dim=16, **params):
        # Broadcasting: [1, 1, D] + [K, 1, D] + ([K, M, D] * [B, K, 1, D])
        h_mu = params.get("h_mu", getattr(self, "h_mu", torch.zeros(1, 1, latent_dim)))
        noise_weights = params.get("noise_weights", getattr(self, "noise_weights", torch.randn(k_atoms, fourier_dim, latent_dim)))
        atom_mu = params.get("atom_mu", getattr(self, "atom_mu", torch.randn(k_atoms, 1, latent_dim) * 2 * math.sqrt(0.01)))
        omega = h_mu + atom_mu + noise_weights.unsqueeze(0) * bw
        return torch.clamp(omega, -100.0, 100.0)
    
    def stack_features(self, latent_kernels):
        return torch.stack(latent_kernels)
    
    def get_resource(self, name_string, **params):
        return getattr(self, name_string, None)
    
    def numerically_stable_gamma(self, gamma_concentration_init):
        raw = float(gamma_concentration_init)
        safe = math.log(math.exp(raw) - 1)
        return safe
    
    def global_stick_breaking(self, k_atoms=30, **params):
        q_sig_global = params.get("q_sig_global", getattr(self, "q_sig_global", torch.ones(k_atoms - 1) * -4.0))
        q_mu_global = params.get("q_mu_global", getattr(self, "q_mu_global", torch.zeros(k_atoms - 1)))
        safe = self.numerically_stable_gamma(2.5)
        raw_gamma = params.get("raw_gamma", getattr(self, "raw_gamma", torch.tensor(safe)))
        q_sig_global = self.apply_softplus(q_sig_global)
        q_dist_global = dist.Normal(q_mu_global, q_sig_global)
        qz_global = q_dist_global.rsample()
        log_detj = -F.softplus(-qz_global) - F.softplus(qz_global)
        log_qv = q_dist_global.log_prob(qz_global).sum() - log_detj.sum()
        gamma_conc = self.apply_softplus(raw_gamma)
        log_pv = (torch.log(gamma_conc + 1e-3) + (gamma_conc - 1) * (-F.softplus(qz_global))).sum()
        qv_global = torch.sigmoid(qz_global)
        one_minus_v = 1 - qv_global
        cumprod_one_minus_v = torch.cumprod(one_minus_v, dim=-1)
        previous_remaining = torch.roll(cumprod_one_minus_v, 1, dims=-1)
        previous_remaining[..., 0] = 1.0
        beta_k = qv_global * previous_remaining
        beta_last = cumprod_one_minus_v[..., -1:]
        beta = torch.cat([beta_k, beta_last], dim=-1)
        return beta, log_pv, log_qv, gamma_conc
    
    def log_global_kl(self, log_pv, log_qv):
        self.update_added_loss_term("global_divergence", SimpleLoss(log_qv - log_pv))
    
    def dirichlet_posterior_inference_and_log_local_loss(self, x, gamma_conc, beta, local_conc, eps=1e-3):
        prior_conc = (gamma_conc * beta) + self.eps
        prior_conc = torch.clamp(prior_conc, min=1e-2, max=100.0)
        prior_conc = prior_conc.unsqueeze(0).expand(x.size(0), -1)
        post_conc = prior_conc + local_conc
        post_conc = torch.clamp(post_conc, min=eps, max=100.0)
        dist_prior = dist.Dirichlet(prior_conc)
        dist_post = dist.Dirichlet(post_conc)
        pi_posterior = dist_post.rsample()
        local_divergence = torch.distributions.kl_divergence(dist_post, dist_prior)
        self.update_added_loss_term("local_divergence", SimpleLoss(local_divergence.sum()))
        return pi_posterior
    
    def predict_kernel_lengthscale_and_log_mse_loss(self, ls, vae_out: Optional[dict]=None, eps=1e-3, k_atoms=30, latent_dim=16, **params):
        if ls is None and vae_out:
            ls = vae_out.get('ls_params', {}).get('ls')
        if ls is None:
            logger.warning("both lengthscale and vae_out dict are missing")
            return None, self.apply_softplus(params.get("h_log_sigma", torch.tensor(3.0))).exp(), self.apply_softplus(params.get("atom_log_sigma", getattr(self, "atom_log_sigma", torch.randn(k_atoms, 1, latent_dim) * 0.01))).exp()

        ls_pred = torch.clamp(ls, min=eps, max=50.0)
        log_ls = torch.log(ls_pred) 
        log_target = torch.zeros_like(log_ls)
        ls_mse = F.mse_loss(log_ls, log_target)
        self.update_added_loss_term("lengthscale_prior_reg", SimpleLoss(ls_mse))
        h_log_sigma = params.get("h_log_sigma", getattr(self, "h_log_sigma", torch.tensor(3.0)))
        atom_log_sigma = params.get("atom_log_sigma", getattr(self, "atom_log_sigma", torch.randn(k_atoms, 1, latent_dim) * 0.01))
        sigmas = h_log_sigma + atom_log_sigma
        log_scale = self.apply_softplus(sigmas)
        bw_base = log_scale.exp()
        precision = 1.0 / (ls_pred.unsqueeze(2) + eps)
        bw_learned = bw_base.unsqueeze(0) * precision
        return ls_pred, bw_learned
    
    def get_local_evidence(self, mualpha, cholalpha, diagalpha):
        alpha_logits = self.lowrankmultivariatenorm(mualpha, cholalpha, diagalpha)
        local_conc = self.apply_softplus(alpha_logits)
        return local_conc
    
    def compute_primitives(self, x):
        lin = self.linear(x) * self.linear_scale
        per = torch.cos(self.periodic(x))
        rbf = torch.exp(-torch.pow(self.rbf(x), 2))
        rat = 1.0 / (1.0 + torch.pow(self.rational(x), 2))
        return lin, per, rbf, rat
    
    def compute_kernel_interactions(self, lin, per, rbf, rat):
        "input stack: [B, 4, 32]"
        stack = torch.stack([lin, per, rbf, rat], dim=1)
        mask = torch.sigmoid(self.selection_weights).unsqueeze(-1)
        stack_safe = torch.abs(stack) + 1e-6
        log_stack = torch.log(stack_safe)
        
        #- b p d (batch, primitives, dim), k p   (products, primitives) -> b k d (batch, products, dim)
        log_product = torch.einsum('bpd, kp -> bkd', log_stack, mask.squeeze(-1))

        product_features = torch.exp(log_product) #-[B, 12, 32]
        interactions_matrix = product_features.flatten(start_dim=1) #-[B, 384]
        return self.complex_interactions(interactions_matrix) #-[B, products, 128]
    
    def feed_dirichlet_gate(self, kernel_features):
        return self.spectral_feedback_loop(kernel_features)
    
    def get_cov_matrices(self, features):
        latent_kernels = self.get_latent_kernels(features)
        stacked_features = self.stack_features(latent_kernels)
        L = torch.tril(stacked_features)
        diag_mask = torch.eye(L.size(-1), device=L.device).bool()
        diag_elements = L[..., diag_mask.unsqueeze(0).expand_as(L)]
        L[..., diag_mask.expand_as(L)] = torch.nn.functional.softplus(diag_elements) + 1e-6
        cov_matrices = torch.einsum('lbi, lci -> lbc', L, L)
        return cov_matrices
    
    def get_latent_kernels(self, kernel_features, **params):
        return [latent(kernel_features) for latent in self.latent_kernel_heads]

    def get_alpha_mvn_heads_decoder(self, bottleneck):
        """
        Returns a valid Cholesky factor L such that Sigma = L @ L.T
        Uses a Low-Rank + Diagonal parameterization for stability.
        """
        mu = self.mu_alpha(bottleneck)
        res = self.chol_alpha(bottleneck).view(-1, self.k_atoms, self.rank_r)
        diag = F.softplus(self.diag_alpha(bottleneck)) + 1e-6
        return mu, res, diag
    
    @staticmethod
    def init_inducing_with_fft(
        y_target, 
        n_inducing, 
        feature_dim
    ):
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
        if yflat.abs().sum() < 1e-6:
            return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        
        fourier_vals = torch.fft.rfft(yflat)
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
        
        samples = density[indices]
        
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
    
    def run_neural_nets_dirichlet(self, x):
        bottleneck_dim = self.bottleneck_mixer(x) #-takes latent z[B,16] -> [B,64]
        features = self.kernel_network(bottleneck_dim, features_only=True)
        return bottleneck_dim, features
    
    def compress_and_gate(self, features, gate):
        embedded_features = self.compress_spectral_features_head(features)
        return gate * embedded_features
    
    def log_alpha_kl(self, mu_alpha, chol_alpha):
        """
        Computes KL[ q(logits) || p(logits) ]
        Where q is your learned posterior (from the decoder)
        And p is your chosen MVN prior.
        """
        batch_size, k_atoms = mu_alpha.shape
        

        posterior = dist.MultivariateNormal(
            loc=mu_alpha, 
            scale_tril=chol_alpha 
        )
        
        prior = dist.MultivariateNormal(
            loc=torch.zeros(k_atoms, device=mu_alpha.device),
            covariance_matrix=torch.eye(k_atoms, device=mu_alpha.device)
        )

        kl = dist.kl_divergence(posterior, prior).mean()

        self.update_added_loss_term("alpha_kl", SimpleLoss(kl))
        
        return self
    
    def disentangle(self, bottleneck):
        z_ls, z_pi, z_dt = torch.split(bottleneck, [4, 30, 30], dim=-1)
        
        trend     = self.ls_head_recon(z_ls) # (-1 to 1)
        amplitude = self.pi_head_recon(z_pi) # (0 to 1)
        residual  = self.data_head_recon(z_dt) # Unbounded
        
        # Physics-Informed Composition: Base trend + (Local Amplitude * High Frequency Details)
        recon = trend + (amplitude * residual)
        return recon, trend, amplitude, residual
    

    def log_alpha_kl_low_rank(self, mu, chol, diag, k_atoms=30):
        """
        Args:
            mu: [Batch, k_atoms]
            chol: [Batch, k_atoms * 3] (The low-rank factor)
            diag: [Batch, k_atoms] (The diagonal variance)
        Returns:
            kl_div: [Batch]
        """
        
        batch_size = mu.size(0)
        rank = 3 
        
        
        cov_factor = chol.view(batch_size, k_atoms, rank)
        
        cov_diag = torch.nn.functional.softplus(diag) + 1e-5
        
        q_dist = LowRankMultivariateNormal(
            loc=mu,
            cov_factor=cov_factor,
            cov_diag=cov_diag
        )
        p_dist = Independent(Normal(
            loc=torch.zeros_like(mu), 
            scale=torch.ones_like(mu)
        ), 1)
        
        kl = kl_divergence(q_dist, p_dist)
        self.update_added_loss_term("alpha_kl", SimpleLoss(kl))
        return self
    
    def dirichlet_sample(self, alpha):
        alpha = self.apply_softplus(alpha)
        alpha = torch.clamp(alpha, min=1e-3, max=100.0)
        q_alpha= torch.distributions.Dirichlet(alpha)
        pi_sample = q_alpha.rsample()
        return pi_sample
