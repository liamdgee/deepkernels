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

import linear_operator
from linear_operator.operators import RootLinearOperator, MatmulLinearOperator, DiagLinearOperator, AddedDiagLinearOperator

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
    real_x: torch.Tensor
    lmc_matrices: torch.Tensor

class AmortisedDirichlet(BaseGenerativeModel):
    def __init__(self, 
                 config=None,
                 **kwargs):
        
        super().__init__()
        self.config = config if config else None
        self.kwargs = kwargs
        self.jitter = self.kwargs.get("jitter", 1e-6)
        self.dropout = self.kwargs.get("dropout", 0.05)
        self.latent_dim = self.kwargs.get("latent_dim", 16)
        self.input_dim = self.kwargs.get("input_dim", 30) ## -- or self.kwargs.get("num_features_real_x", 30)
        self.bottleneck_dim = self.kwargs.get("bottleneck_dim", 64)
        self.k_atoms = self.kwargs.get("k_atoms", 30)
        self.rank = self.kwargs.get("alpha_factor_rank", 3)
        self.M = self.kwargs.get("num_fourier_features", 128)
        self.spectral_emb_dim = self.kwargs.get("spectral_emb_dim", 2048)
        self.eps = self.kwargs.get("eps_dirichlet", 1e-3)
        self.gamma_concentration_init = self.kwargs.get("gamma_concentration_init", 2.5)
        self.num_latents = self.kwargs.get("num_latents", 8)
        self.large_eps = self.kwargs.get("large_eps", 4e-2)
        self.posterior_eps = self.kwargs.get("posterior_dirichlet_epsilon", 4e-5) #-cannot be too large or throws out simplex vals-#
        self.upper_clamp = self.kwargs.get("conc_clamp", 150.0)

        #-hypernetworks
        self.compress_spectral_features_head = torch.nn.utils.spectral_norm(nn.Linear(self.k_atoms * self.M * 2, self.spectral_emb_dim))
        
        self.bottleneck_mixer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.latent_dim, self.bottleneck_dim)), 
            nn.LayerNorm(self.bottleneck_dim), 
            nn.Tanh()
        )
        
        self.kernel_network = KernelNetwork()

        #-params-#
        self.q_mu_global = nn.Parameter(torch.zeros(self.k_atoms - 1))
        self.q_log_sigma_global = nn.Parameter(torch.ones(self.k_atoms - 1) * -4.0)
        self.h_mu = nn.Parameter(torch.zeros(1, 1, self.latent_dim)) 
        self.h_log_sigma = nn.Parameter(torch.tensor(3.0))
        self.atom_log_sigma = nn.Parameter(torch.randn(self.k_atoms, 1, self.latent_dim) * 0.025)
        self.atom_mu = nn.Parameter(torch.randn(self.k_atoms, 1, self.latent_dim) * 2 * math.sqrt(0.025))
        self.raw_gamma = nn.Parameter(torch.tensor((self.numerically_stable_gamma(self.gamma_concentration_init))))
        self.lengthscale_log_uncertainty = nn.Parameter(torch.ones(1, self.k_atoms))
        
        self.lmc_matrix = nn.Parameter(torch.randn(self.k_atoms, self.num_latents) * 0.1)
        self.lmc_var = nn.Parameter(torch.randn(self.k_atoms) * 0.1 - 2.25)

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
        self.register_buffer("noise_weights", torch.randn(self.k_atoms, self.M, self.latent_dim))
        self.register_buffer("noise_bias", torch.rand(self.k_atoms, self.M))

         # --- Inverse Wishart Prior Buffers (psi: mean 0 (tasks are indepenjdent, nu: degrees freedom)) ---
        self.psi_scale = self.kwargs.get("psi_scale", 1.0)
        self.register_buffer("pr_psi", torch.eye(self.k_atoms) * self.psi_scale)
        self.register_buffer("pr_nu", torch.tensor(self.k_atoms + 2.0))

        #-loss terms-#
        self.register_added_loss_term("global_divergence")
        self.register_added_loss_term("local_divergence")
        self.register_added_loss_term("inverse_wishart")
        

    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params) -> DirichletOutput:
        """
        performs nonparametric clustering according to a hierarchical dirichlet process using learned lengthscale
        and concentration param refinement
        Args:
            latent z (param: x) -- dim 16
        """

        batch_size = x.size(0)
        device = x.device
        empty_tensor = torch.empty(0, device=device)

        if vae_out:
            mu_z, logvar_z, real_x = vae_out.mu_z, vae_out.logvar_z, vae_out.real_x
        else:
            mu_z, logvar_z, real_x = empty_tensor, empty_tensor, empty_tensor
            logger.warning("could not retreive core variables (mu_z, logvar_z and input_x) from vae_out. likely encoder output not received.")
        
        
        alpha_mu = params.get('alpha_mu', None)
        if alpha_mu is None and vae_out is not None:
            alpha_mu = getattr(vae_out, 'alpha_mu', None)
        
        if alpha_mu is None:
            alpha_mu = torch.zeros(batch_size, self.k_atoms, device=device)
        
        
        alpha_diag = params.get('alpha_diag', None)
        if alpha_diag is None and vae_out is not None:
            alpha_diag = getattr(vae_out, 'diag', None)
        
        if alpha_diag is None:
            alpha_factor = torch.ones(batch_size, self.k_atoms, device=device)
        
        
        alpha_factor = params.get('alpha_factor', None)
        if alpha_factor is None and vae_out is not None:
            alpha_factor = getattr(vae_out, 'alpha_factor', None)
        
        if alpha_factor is None:
            alpha_factor = torch.zeros(batch_size, self.k_atoms, self.rank, device=device)
        
        
        ls = params.get('ls')
        if ls is None and vae_out is not None:
            ls = getattr(vae_out, 'ls', None)
        
        if isinstance(ls, torch.Tensor) and ls.numel() == 0:
            ls = None
        
        
        beta, log_pv, log_qv, gamma_conc = self.global_stick_breaking()

        self.update_added_loss_term("global_divergence", LossTerm(log_qv - log_pv))
    
        bottleneck, gate, gp_params = self.run_neural_nets_dirichlet(x)
        
        local_conc = self.get_local_evidence(alpha_mu, alpha_factor, alpha_diag) #-alpha reparameterisation-#

        pi, local_kl = self.dirichlet_posterior_inference_and_log_local_loss(x, gamma_conc, beta, local_conc) #-< alpha is here as local conc-#

        B_mat_current_state = self.coregionalisation_matrix(pi)

        iw_loss = self.inverse_wishart_penalty(B_mat_current_state)

        self.update_added_loss_term("inverse_wishart", LossTerm(iw_loss))

        self.update_added_loss_term("local_divergence", LossTerm(local_kl.sum()))
        
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
            logvar_z=logvar_z,
            real_x=real_x,
            lmc_matrices=B_mat_current_state
        )
    
    
    def numerically_stable_gamma(self, gamma_concentration_init):
        raw = float(gamma_concentration_init)
        safe = math.log(math.exp(raw) - 1)
        return safe
    
    def get_omega(self, bw, **params):
        # Broadcasting: [1, 1, D] + [K, 1, D] + ([K, M, D] * [B, K, 1, D])
        noise_weights = self.noise_weights
        omega = self.h_mu + self.atom_mu + noise_weights.unsqueeze(0) * bw
        return torch.clamp(omega, -100.0, 100.0)
    
    def coregionalisation_matrix(self, pi):
        batch_size = pi.size(0)
        pi_bc = pi.unsqueeze(-1)
        W_current_state = pi_bc * self.lmc_matrix
        v = F.softplus(self.lmc_var) + self.jitter
        v_bc = v.unsqueeze(0).expand(batch_size, -1)
        lazyroot = RootLinearOperator(W_current_state)
        lazydiag = DiagLinearOperator(v_bc)
        B_lazy = AddedDiagLinearOperator(lazydiag, lazyroot)
        return B_lazy
    
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
    
    def global_stick_breaking(self, **params):
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
    
    def predict_kernel_lengthscales(self, ls, eps=1e-3, max_ls=100.0, **params):
        
        sigmas = self.h_log_sigma + self.atom_log_sigma 
        log_scale = F.softplus(sigmas) + 1e-4
        bw_base = log_scale.exp()
        
        if ls is None:
            ls_pred = bw_base.squeeze(1).mean(dim=-1).unsqueeze(0) 
            ls_pred = torch.clamp(ls_pred, min=eps, max=max_ls)
            ls_logvar = torch.zeros(1, self.k_atoms, device=ls_pred.device)
            bw_learned = bw_base.unsqueeze(0) 
        else:
            ls_pred = torch.clamp(ls, min=eps, max=max_ls)
            batch_size = ls_pred.size(0)
            precision = 1.0 / (ls_pred.view(batch_size, self.k_atoms, 1, 1) + eps)
            bw_learned = bw_base.unsqueeze(0) * precision
            ls_logvar = self.lengthscale_log_uncertainty.expand(batch_size, -1) #--is this actually doing anything??-#

        return ls_pred, bw_learned, ls_logvar
    
    def run_neural_nets_dirichlet(self, x):
        bottleneck = self.bottleneck_mixer(x) #-takes latent z[B,16] -> [B,64]
        nkn_out = self.kernel_network(bottleneck)
        return bottleneck, nkn_out.dirichlet_features, nkn_out.gp_params
    
    def compress_and_gate(self, features, gate):
        embedded_features = self.compress_spectral_features_head(features)
        return gate * embedded_features
    
    def dirichlet_posterior_inference_and_log_local_loss(self, x, gamma_conc, beta, local_conc):
        
        
        prior_conc = (gamma_conc * beta) + self.large_eps
        prior_conc = torch.clamp(prior_conc, min=self.large_eps, max=self.upper_clamp)
        prior_conc = prior_conc.unsqueeze(0).expand(x.size(0), -1)

        post_conc = prior_conc + local_conc
        post_conc = torch.clamp(post_conc, min=self.large_eps, max=self.upper_clamp)

        dist_prior = dist.Dirichlet(prior_conc)
        dist_post = dist.Dirichlet(post_conc)
        pi_posterior = torch.clamp(dist_post.rsample(), min=self.posterior_eps)
        pi_posterior = pi_posterior / pi_posterior.sum(dim=-1, keepdim=True)
        local_divergence = torch.distributions.kl_divergence(dist_post, dist_prior)
        
        return pi_posterior, local_divergence
    
    def get_local_evidence(self, mualpha, factoralpha, diagalpha):
        """NB: this is actually alpha logits -- do not recalc in decoder from same value"""
        alpha_logits = self.lowrankmultivariatenorm(mualpha, factoralpha, diagalpha)
        local_conc = F.softplus(alpha_logits) + self.jitter
        return local_conc

    def inverse_wishart_penalty(self, B_lazy):
        """
        Computes the Inverse-Wishart penalty for a batch of lazy covariance matrices.
        
        Args:
            B_lazy: The AddedDiagLinearOperator from your coregionalisation_matrix
            nu: Degrees of freedom (must be > k_atoms - 1) #-buffer pr_nu-#
            psi_scalar: Scalar for the isotropic scale matrix Psi = psi * I
            k_atoms: The dimension K of the covariance matrix
        """
        batch_size = B_lazy.size(0)
        logdet_B = B_lazy.logdet() 
        identity = torch.eye(self.k_atoms, device=B_lazy.device).expand(batch_size, self.k_atoms, self.k_atoms)
        B_inverse = B_lazy.solve(identity) 
        trace_B_inv = torch.diagonal(B_inverse, dim1=-2, dim2=-1).sum(-1)
        
        #-penalty math-#
        # L_IW = 0.5 * (nu + K + 1) * log|B| + 0.5 * Tr(Psi * B^-1)

        term1 = 0.5 * (self.pr_nu + self.k_atoms + 1) * logdet_B
        term2 = 0.5 * self.psi_scale * trace_B_inv
        iw_penalty = term1 + term2
        return iw_penalty.mean()

class LossTerm(gpytorch.mlls.AddedLossTerm):
    """
    A concrete implementation of an AddedLossTerm that simply 
    returns a pre-calculated scalar tensor.
    """
    def __init__(self, loss_tensor):
        self.loss_tensor = loss_tensor
        
    def loss(self):
        return self.loss_tensor
    
    
   

