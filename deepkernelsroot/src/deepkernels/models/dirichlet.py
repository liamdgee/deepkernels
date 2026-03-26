import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Laplace
from torch.distributions.transforms import StickBreakingTransform
import math
import logging
import gpytorch
from gpytorch.priors import NormalPrior, GammaPrior, LogNormalPrior, Prior
from typing import Optional
from gpytorch.module import Module
from gpytorch.constraints import Positive, GreaterThan, Interval

import linear_operator

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.models.NKN import KernelNetwork, GPParams
from deepkernels.kernels.keops import CustomLaplacePrior
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
    local_conc: torch.Tensor
    ls_logvar: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    real_x: torch.Tensor
    lmc_matrices: torch.Tensor
    gates: torch.Tensor
    linear: torch.Tensor
    periodic: torch.Tensor
    rational: torch.Tensor
    polynomial: torch.Tensor
    matern: torch.Tensor
    lmc_consensus: torch.Tensor


from dataclasses import dataclass

@dataclass
class DirichletConfig:
    # --- Dimensions & Architecture ---
    latent_dim: int = 16
    input_dim: int = 30
    bottleneck_dim: int = 64
    k_atoms: int = 30
    alpha_factor_rank: int = 3
    num_latents: int = 8
    num_experts: int = 8
    num_fourier_features: int = 128
    spectral_emb_dim: int = 2048
    n_data: float = 87636.0 # Kept as float for division stability
    
    # --- Hyperparameters ---
    dropout: float = 0.0
    gamma_concentration_init: float = 1.5417
    noise_scalar: float = 0.316
    psi_scale: float = 1.0
    
    # --- Numerical Bounds & Tolerances ---
    jitter: float = 1e-6
    eps: float = 1e-3
    large_eps: float = 4e-2
    posterior_dirichlet_epsilon: float = 2e-5
    conc_clamp: float = 9.0
    concentration_strength: float = 77.0
    min_ls: float = 0.05
    max_ls: float = 15.0
    sigma_lower_bound: float = 1e-4
    sigma_upper_bound: float = 5.0
    mu_lower_bound: float = -17.0
    mu_upper_bound: float = 17.0
    eps_clip: float = 2.7
    stick_breaking_epsilon: float = 3e-3
    uniform_dist_clamp: float = 5e-5
    tiny_eps: float = 3e-8
    M_series: int = 9
    num_primitives: int = 5
    individual_kernel_dim_out: int = 32
    evidence_dim: int = 58
    training_batch_size: int = 1024


class AmortisedDirichlet(BaseGenerativeModel):
    def __init__(self,
                 config=None,
                 **kwargs):
        
        super().__init__()
        self.config = config if config is not None else DirichletConfig()
        self.stick_breaking_epsilon = self.config.stick_breaking_epsilon
        self.uniform_dist_clamp = self.config.uniform_dist_clamp
        self.tiny_eps = self.config.tiny_eps
        self.M_series = self.config.M_series
        self.jitter = self.config.jitter
        self.dropout = self.config.dropout
        self.latent_dim = self.config.latent_dim
        self.input_dim = self.config.input_dim ## -- or self.kwargs.get("num_features_real_x", 30)
        self.bottleneck_dim = self.config.bottleneck_dim
        self.k_atoms = self.config.k_atoms
        self.rank = self.config.alpha_factor_rank
        self.M = self.config.num_fourier_features
        self.spectral_emb_dim = self.config.spectral_emb_dim
        self.eps = self.config.eps
        self.gamma_concentration_init = self.config.gamma_concentration_init
        self.num_latents = self.config.num_latents
        self.large_eps = self.config.large_eps
        self.posterior_eps = self.config.posterior_dirichlet_epsilon
        self.conc_clamp = self.config.conc_clamp
        self.min_ls = self.config.min_ls
        self.max_ls = self.config.max_ls
        self.noise_scalar = self.config.noise_scalar
        self.num_experts = self.config.num_experts
        self.sigma_lower_bound = self.config.sigma_lower_bound
        self.sigma_upper_bound = self.config.sigma_upper_bound
        self.mu_lower_bound = self.config.mu_lower_bound
        self.mu_upper_bound = self.config.mu_upper_bound
        self.eps_clip = self.config.eps_clip
        self.psi_scale = self.config.psi_scale
        self.concentration_strength = self.config.concentration_strength
        self.evidence_dim = self.config.evidence_dim
        self.batch_size = self.config.training_batch_size
        
        #-global & dynamic:
        self.input_dim = kwargs.get("input_dim", 30)
        self.n_data = kwargs.get('n_data', 76674.0)
        
        #-hypernetworks
        self.compress_spectral_features_head = torch.nn.utils.spectral_norm(nn.Linear(self.k_atoms * self.M * 2, self.spectral_emb_dim))
        
        self.bottleneck_mixer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.latent_dim, self.bottleneck_dim)), 
            nn.LayerNorm(self.bottleneck_dim), 
            nn.ELU(inplace=True)
        )
        
        self.kernel_network = KernelNetwork(config=self.config)

        scale_constraint = gpytorch.constraints.GreaterThan(1e-4)
        positive_constraint = gpytorch.constraints.Positive()

        
        ls_constraint = gpytorch.constraints.Interval(lower_bound=self.min_ls, upper_bound=self.max_ls)
        lmc_constraint = gpytorch.constraints.Interval(lower_bound=-5.0, upper_bound=5.0)
        freq_bounds = gpytorch.constraints.Interval(-30.0, 20.0)
        broad_var = gpytorch.constraints.Interval(0.01, 5.0)
        tight_var = gpytorch.constraints.Interval(0.01, 1.5)
        
        #============
        #--params--#
        #============
        #-variational q dist-#
        #-kumaraswamy distribution for global stick breaking-#
        self.register_parameter(name="raw_q_a_global", parameter=nn.Parameter(torch.zeros(self.k_atoms - 1)))
        self.register_constraint("raw_q_a_global", positive_constraint)
        
        self.register_parameter(name="raw_q_b_global", parameter=nn.Parameter(torch.zeros(self.k_atoms - 1)))
        self.register_constraint("raw_q_b_global", positive_constraint)

        #-learnable gamma conc param-#
        self.register_parameter(name="raw_gamma", parameter=torch.nn.Parameter(torch.zeros(1)))

        #-global hierarchical dist (H)-# 
        self.register_parameter(name="raw_h_mu", parameter=nn.Parameter(torch.zeros(1, 1, self.latent_dim)))
        self.register_constraint("raw_h_mu", freq_bounds)

        self.register_parameter(name="raw_h_sigma", parameter=nn.Parameter(torch.zeros(1, 1, self.latent_dim)))
        self.register_constraint("raw_h_sigma", broad_var)
        
        #-atoms for spectral features (gaussian)-#
        self.register_parameter(name="raw_atom_loc", parameter=nn.Parameter(torch.randn(self.k_atoms, 1, self.latent_dim) * 2 * math.sqrt(0.1)))
        self.register_constraint("raw_atom_loc", freq_bounds)

        self.register_parameter(name="raw_atom_scale", parameter=nn.Parameter(torch.randn(self.k_atoms, 1, self.latent_dim) * 0.1))
        self.register_constraint("raw_atom_scale", tight_var)

        #kernel lengthscales-#
        self.register_parameter(name="raw_lengthscale_uncertainty", parameter=nn.Parameter(torch.zeros(1, self.k_atoms)))
        self.register_constraint("raw_lengthscale_uncertainty", ls_constraint)

        #-lmc params-#
        self.register_parameter(name="raw_lmc_var", parameter=nn.Parameter(torch.zeros(self.k_atoms)))
        self.register_constraint("raw_lmc_var", scale_constraint)

        self.register_parameter(name="raw_lmc_matrix", parameter=nn.Parameter(torch.zeros(self.k_atoms, self.num_latents)))
        self.register_constraint("raw_lmc_matrix", lmc_constraint)
        


        #-priors-$
        self.register_prior(
            "h_scale_prior", 
            NormalPrior(loc=0.0025, scale=1.05), 
            lambda m: m.h_sigma,
            lambda m, v: None
        )

        self.register_prior(
            "atom_scale_prior", 
            gpytorch.priors.LogNormalPrior(loc=0.0, scale=1.0), 
            lambda m: m.atom_scale,
            lambda m, v: None
        )

        self.register_prior(
            "gamma_prior", 
            GammaPrior(concentration=1.0, rate=0.5), 
            lambda m: m.gamma,
            lambda m, v: None
        )


        #-buffers-#
        #-fourier buffers drawn once-#
        self.register_buffer("noise_weights", torch.randn(self.k_atoms, self.M, self.latent_dim))
        self.register_buffer("noise_bias", torch.rand(self.k_atoms, self.M))
        
        # --- Inverse Wishart Prior Buffers (psi: mean 0 (tasks are indepenjdent, nu: degrees freedom)) ---
        self.register_buffer("pr_psi", torch.eye(self.k_atoms) * self.psi_scale)
        self.register_buffer("pr_nu", torch.tensor(self.k_atoms + 2.0))
        

        #-loss terms-#
        self.register_added_loss_term("global_divergence")
        self.register_added_loss_term("local_divergence")
        self.register_added_loss_term("inverse_wishart")

        self._init_weights()

        neutral_stick = torch.full((self.k_atoms - 1,), 0.5413)
        atom_spread = torch.linspace(-5.0, 5.0, self.k_atoms).view(self.k_atoms, 1, 1)
        initial_atom_loc = atom_spread.expand(-1, 1, self.latent_dim).clone()
        initial_atom_loc += torch.randn_like(initial_atom_loc) * 0.1
        initial_atom_scale = torch.full((self.k_atoms, 1, self.latent_dim), -1.5)
        initial_lmc = torch.full((self.k_atoms, self.num_latents), -2.25)
        target_lmc_var = torch.ones(self.k_atoms) + (torch.rand(self.k_atoms) * 0.1)
        
        self.initialize(
            raw_q_a_global=neutral_stick,
            raw_q_b_global=neutral_stick,
            raw_h_mu=torch.zeros(1, 1, self.latent_dim),
            raw_gamma=torch.full((1,), -1.87),
            raw_h_sigma=torch.tensor([0.5413]),
            raw_atom_loc=initial_atom_loc,
            raw_atom_scale=initial_atom_scale,
            raw_lengthscale_uncertainty=torch.zeros(1, self.k_atoms),
            raw_lmc_var=target_lmc_var,
            raw_lmc_matrix=initial_lmc
        )

    @property
    def q_a_global(self): return self.raw_q_a_global_constraint.transform(self.raw_q_a_global)

    @property
    def gamma(self):
        return 0.35 + (6.0 - 0.35) * torch.sigmoid(self.raw_gamma)
        
    @property
    def q_b_global(self): return self.raw_q_b_global_constraint.transform(self.raw_q_b_global)
    
    @property
    def h_mu(self): return self.raw_h_mu_constraint.transform(self.raw_h_mu)
        
    @property
    def h_sigma(self): return self.raw_h_sigma_constraint.transform(self.raw_h_sigma)
    
    @property
    def atom_loc(self): return self.raw_atom_loc_constraint.transform(self.raw_atom_loc)
        
    @property
    def atom_scale(self): return self.raw_atom_scale_constraint.transform(self.raw_atom_scale)
        
    @property
    def lengthscale_uncertainty(self): return self.raw_lengthscale_uncertainty_constraint.transform(self.raw_lengthscale_uncertainty)
        
    @property
    def lmc_var(self): return self.raw_lmc_var_constraint.transform(self.raw_lmc_var)

    @property
    def lmc_matrix(self): return self.raw_lmc_matrix_constraint.transform(self.raw_lmc_matrix)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_orig'):
                    nn.init.orthogonal_(module.weight_orig, gain=1.41)
                else:
                    nn.init.orthogonal_(module.weight, gain=1.41)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    #-helper-#
    def _safe_tensor(self, value):
        """Helper to ensure we always pass a tensor to the inverse transform"""
        return value if torch.is_tensor(value) else torch.tensor(value)
    
    def forward(self, x, vae_out, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, generative_mode:bool=False, **params) -> DirichletOutput:
        """
        performs nonparametric clustering according to a hierarchical dirichlet process using learned lengthscale
        and concentration param refinement
        Args:
            latent z (param: x) -- dim 16
        """
        t = params.get("t", 0)
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
            alpha_mu = torch.zeros(batch_size, self.evidence_dim, device=device)
        
        
        alpha_diag = params.get('alpha_diag', None)
        if alpha_diag is None and vae_out is not None:
            alpha_diag = getattr(vae_out, 'diag', None)
        
        if alpha_diag is None:
            alpha_diag = torch.ones(batch_size, self.evidence_dim, device=device)
        
        
        alpha_factor = params.get('alpha_factor', None)
        if alpha_factor is None and vae_out is not None:
            alpha_factor = getattr(vae_out, 'alpha_factor', None)
        
        if alpha_factor is None:
            alpha_factor = torch.zeros(batch_size, self.evidence_dim, self.rank, device=device)
        
        ls = params.get('ls')
        if ls is None and vae_out is not None:
            ls = getattr(vae_out, 'ls', None)
        
        if isinstance(ls, torch.Tensor) and ls.numel() == 0:
            ls = None
        
        beta, _, global_kl = self.global_stick_breaking_kumaraswamy()

        self.update_added_loss_term("global_divergence", LossTerm(global_kl, t_index=t))
    
        bottleneck, gate, gp_params = self.run_neural_nets_dirichlet(x)
        
        qa, qb, alpha = self.get_local_evidence(alpha_mu, alpha_factor, alpha_diag) #-alpha reparameterisation-#

        pi, local_kl = self.local_stick_breaking(qa, qb, beta)

        self.update_added_loss_term("local_divergence", LossTerm(local_kl, t_index=t))

        pi = pi.clamp(min=4e-3)

        Bmat, Wcon = self.coregionalisation_matrix(pi)

        iw_loss = self.inverse_wishart_penalty(Bmat)

        self.update_added_loss_term("inverse_wishart", LossTerm(iw_loss, t_index=t))

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
            local_conc=alpha,
            ls_logvar=ls_logvar,
            gates=gp_params.gates,
            linear=gp_params.linear,
            periodic = gp_params.periodic,
            rational = gp_params.rational,
            polynomial = gp_params.polynomial,
            matern = gp_params.matern,
            mu_z=mu_z,
            logvar_z=logvar_z,
            real_x=real_x,
            lmc_matrices=Bmat,
            lmc_consensus=Wcon
        )
    
    def get_omega(self, bw, **params):
        base = self.h_mu.unsqueeze(2) + self.atom_loc.transpose(0, 1).unsqueeze(2)
        omega = torch.einsum('bkd, kmd -> bkmd', bw, self.noise_weights)
        
        omega.add_(base)
        
        omega.clamp_(-30.0, 20.0)
        
        return omega
    
    def coregionalisation_matrix(self, pi, tau=0.86, hard=True, jitter=1e-5):
        B_base = torch.mm(self.lmc_matrix, self.lmc_matrix.t())
        B_dense = B_base * pi.unsqueeze(-1) * pi.unsqueeze(-2)
        v = self.lmc_var + jitter
        B_dense.diagonal(dim1=-2, dim2=-1).add_(v)
        raw_logits = pi.unsqueeze(-1) * self.lmc_matrix.unsqueeze(0)
        
        W_gumbel = torch.nn.functional.gumbel_softmax(
            raw_logits, 
            tau=tau, 
            hard=hard, 
            dim=1
        )
        
        W_consensus = W_gumbel.mean(dim=0)
        
        W_consensus = torch.clamp(W_consensus, min=-5.0, max=5.0)
        
        return B_dense.contiguous(), W_consensus
    
    
    def random_fourier_features(self, z, omega, pi, **params):
        """inputs latent dim z"""
        B, D = z.shape
        
        if pi is None:
            pi = torch.full((B, self.k_atoms), 1.0/self.k_atoms, device=z.device)
            if self.training:
                # This is safe because pi isn't part of the autograd graph yet
                pi.add_(torch.randn_like(pi).mul_(0.01))
            pi = F.softmax(pi, dim=-1)
            
        if omega.dim() == 4:
            proj = torch.einsum('bd, bkmd -> bkm', z, omega)
        else:
            proj = F.linear(z, omega.view(-1, D)).view(B, self.k_atoms, self.M)
        
        proj = proj + self.noise_bias.unsqueeze(0) 
        
        scale = 1.0 / math.sqrt(self.M)
        
        pi_scl = torch.sqrt(pi).unsqueeze(-1) * scale
        
        cos_proj = torch.cos(proj) * pi_scl
        sin_proj = torch.sin(proj) * pi_scl
        
        feats = torch.cat([cos_proj, sin_proj], dim=-1)
        return feats.flatten(1)

    def global_stick_breaking_kumaraswamy(self, **kwargs):
        q_a = F.softplus(self.q_a_global) + 1.035
        q_b = F.softplus(self.q_b_global) + 1.035
        u = torch.rand_like(q_a).clamp(self.uniform_dist_clamp, 1.0 - self.uniform_dist_clamp)
        
        #-Inverse CDF transform-: fomula: v = (1 - (1 - u)^(1/b))^(1/a)
        qv_global = (1.0 - (1.0 - u).pow(1.0 / q_b)).pow(1.0 / q_a)
        qv_global = qv_global.clamp(self.uniform_dist_clamp, 1.0 - self.uniform_dist_clamp)
        gamma_conc = self.gamma + self.stick_breaking_epsilon
        
        log_v = torch.log(qv_global)
        log_1_minus_v = torch.log(1.0 - qv_global)
        pad_log_1_minus_v = F.pad(log_1_minus_v, (1, 0), value=0.0)
        log_pi_k = log_v + torch.cumsum(pad_log_1_minus_v, dim=-1)[..., :-1]
        log_pi_last = torch.cumsum(pad_log_1_minus_v, dim=-1)[..., -1:]
        
        log_beta = torch.cat([log_pi_k, log_pi_last], dim=-1)
        beta = torch.exp(log_beta)
        beta = beta / beta.sum(dim=-1, keepdim=True)

        euler_gamma = 0.5772156649
        psi_b = torch.digamma(q_b)
        
        kl = (
            ((q_a - 1.0) / q_a) * (-euler_gamma - psi_b - 1.0 / q_b) 
            + torch.log(q_a * q_b) 
            - torch.log(gamma_conc) 
            - (q_b - 1.0) / q_b
        )
        Ms = self.M_series
        m = torch.arange(1, Ms + 1, device=q_b.device, dtype=q_b.dtype)
        m_view = m.view([Ms] + [1] * q_b.dim())
        taylor_sum = (1.0 / (m_view * (m_view * q_b + q_a + 1e-6))).sum(dim=0)
        kl += (gamma_conc - 1.0) * q_b * taylor_sum
        global_kl = kl.sum(dim=-1).mean() / self.n_data
        return beta, gamma_conc, global_kl
    
    def local_stick_breaking(self, qa, qb, global_beta):
        prior_a = 1.0 
        prior_b = global_beta[..., :-1] * 37.0 + 1.0
        clamped_qa = torch.clamp(prior_a + qa, max=self.conc_clamp, min=self.jitter)
        clamped_qb = torch.clamp(prior_b + qb, max=self.conc_clamp, min=self.jitter)
        u = torch.rand_like(clamped_qa).clamp(self.uniform_dist_clamp, 1.0 - self.uniform_dist_clamp)
        v = (1.0 - (1.0 - u).pow(1.0 / clamped_qb)).pow(1.0 / clamped_qa)
        v = v.clamp(self.uniform_dist_clamp, 1.0 - self.uniform_dist_clamp)
        
        log_v = torch.log(v)
        log_1_minus_v = torch.log(1.0 - v)
        pad_log_1_minus_v = F.pad(log_1_minus_v, (1, 0), value=0.0)
        log_pi_k = log_v + torch.cumsum(pad_log_1_minus_v, dim=-1)[..., :-1]
        log_pi_last = torch.cumsum(pad_log_1_minus_v, dim=-1)[..., -1:]
        
        log_pi = torch.cat([log_pi_k, log_pi_last], dim=-1)
        pi_posterior = torch.exp(log_pi)
        pi_posterior = torch.clamp(pi_posterior, min=self.posterior_eps)
        pi_posterior = pi_posterior / pi_posterior.sum(dim=-1, keepdim=True)
        
        euler_gamma = 0.5772156649
        psi_b = torch.digamma(clamped_qb)
        kl = (
            ((clamped_qa - 1.0) / clamped_qa) * (-euler_gamma - psi_b - 1.0 / clamped_qb) 
            + torch.log(clamped_qa * clamped_qb) 
            - torch.log(prior_b) 
            - (clamped_qb - 1.0) / clamped_qb
        )
        Ms = self.M_series
        m = torch.arange(1, Ms + 1, device=clamped_qb.device, dtype=clamped_qb.dtype)
        m_view = m.view([Ms] + [1] * clamped_qb.dim())
        taylor_sum = (1.0 / (m_view * (m_view * clamped_qb + clamped_qa + 1e-6))).sum(dim=0)
        kl += (prior_b - 1.0) * clamped_qb * taylor_sum
        local_kl = kl.sum(dim=-1).mean()
        kl = local_kl/self.batch_size
        return pi_posterior, kl

    ##def global_stick_breaking(self, **params):
        #-variational inference-#
        #clamped_q_mu_global = torch.clamp(self.q_mu_global, min=-6.0, max=6.0)
        
        # NB: No softplus needed! self.q_sigma_global is strictly > 1e-6 already.
        #clamped_q_sig_global = torch.clamp(self.q_sigma_global, max=8.0)
        
        #q_dist_global = torch.distributions.Normal(clamped_q_mu_global, clamped_q_sig_global)
        #qz_global = q_dist_global.rsample()

        #-jacobian // entropy-#
        #log_detj = -F.softplus(-qz_global) - F.softplus(qz_global)
        #log_qv = q_dist_global.log_prob(qz_global).sum() - log_detj.sum()
        
        #-prior log likelihood-#
        # NB: No softplus needed! self.gamma is strictly > 1e-4 already.
        #gamma_conc = self.gamma
        #log_pv = (torch.log(gamma_conc + self.jitter) + (gamma_conc - 1) * (-F.softplus(qz_global))).sum()
        
        #-numerically stable stick-breaking-#
        #v = torch.clamp(torch.sigmoid(qz_global), min=1e-5, max=1.0 - 1e-5)
        #log_v = torch.log(v)
        #log_1_minus_v = torch.log(1 - v)

        #pad_log_1_minus_v = F.pad(log_1_minus_v, (1, 0), value=0.0)
        #log_pi_k = log_v + torch.cumsum(pad_log_1_minus_v, dim=-1)[..., :-1]
        #log_pi_last = torch.cumsum(pad_log_1_minus_v, dim=-1)[..., -1:]
        
        #log_beta = torch.cat([log_pi_k, log_pi_last], dim=-1)
        #beta = torch.exp(log_beta)
        
        #beta = torch.clamp(beta, min=1e-7)
        #beta = beta / beta.sum(dim=-1, keepdim=True)
        #return beta, log_pv, log_qv, gamma_conc
    def predict_kernel_lengthscales(self, ls, **params):
        
        bw_base = (self.h_sigma * self.atom_scale) + self.jitter
        
        bw_base = bw_base.squeeze(1)
        
        if ls is None:
            ls_pred = bw_base.mean(dim=-1, keepdim=True)
            ls_pred = torch.clamp(ls_pred, min=self.min_ls, max=self.max_ls)
            ls_logvar = torch.zeros(1, self.k_atoms, device=bw_base.device)
            bw_learned = bw_base.unsqueeze(0) # [1, 30, 16]
        else:
            ls_pred = torch.clamp(ls, min=self.min_ls, max=self.max_ls)
            batch_size = ls_pred.size(0)
            
            # ls_pred is [128, 30]. We unsqueeze to [128, 30, 1]
            precision = 1.0 / (ls_pred.unsqueeze(-1) + self.eps)
            
            # bw_base is [30, 16]. We unsqueeze to [1, 30, 16]
            bw_base_expanded = bw_base.unsqueeze(0)
            
            #- BROADCAST -> [1, 30, 16] * [128, 30, 1] -> [128, 30, 16]
            bw_learned = bw_base_expanded * precision
            
            ls_logvar = self.lengthscale_uncertainty.expand(batch_size, -1)

        return ls_pred, bw_learned, ls_logvar
    
    def run_neural_nets_dirichlet(self, x):
        bottleneck = self.bottleneck_mixer(x) #-takes latent z[B,16] -> [B,64]
        gp_params, features = self.kernel_network.forward(bottleneck)
        return bottleneck, features, gp_params
    
    def compress_and_gate(self, features, gate):
        embedded_features = self.compress_spectral_features_head(features)
        return gate * embedded_features
    
    
    def get_local_evidence(self, mu, factor, diag):
        """NB: this is actually alpha logits -- do not recalc in decoder from same value"""
        if diag is None:
            diag = torch.full_like(mu, self.eps)
        
        safe_diag = F.softplus(diag) + self.jitter

        if factor.dim() == 2:
            factor = factor.view(mu.shape[0], mu.shape[1], -1)
        
        dist = torch.distributions.LowRankMultivariateNormal(mu, factor, safe_diag)
        ab_logits = dist.rsample()
        a_logits, b_logits = ab_logits.chunk(2, dim=-1)
        
        local_evidence_a = torch.clamp(a_logits, min=-7.0, max=4.0)
        local_a = F.softplus(local_evidence_a) + self.eps
        local_evidence_b = torch.clamp(b_logits, min=-7.0, max=4.0)
        local_b = F.softplus(local_evidence_b) + self.eps
        
        return local_a, local_b, ab_logits #-for loss term-#
    def inverse_wishart_penalty(self, B_dense, k_atoms=30):
        eye = torch.eye(k_atoms, dtype=B_dense.dtype, device=B_dense.device)
        B_psd = B_dense + (eye * 1e-5)
        
        L, info = torch.linalg.cholesky_ex(B_psd)
        
        if info.any():
            B_psd = B_psd + (eye * 1e-3)
            L = torch.linalg.cholesky(B_psd) 

        # 3. LOG DETERMINANT (In-place view)
        logdet_B = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        
        # 4. THE INVERSE TRACE FIX
        # torch.cholesky_inverse is a heavily optimized CUDA kernel.
        # It takes L and directly returns B^-1. 
        # We just grab the diagonal and sum it. No identity matrices required!
        B_inv = torch.cholesky_inverse(L)
        trace_B_inv = B_inv.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        
        # 5. SCALAR MATH (Removed the slow as_tensor calls)
        term1 = 0.5 * (self.pr_nu + k_atoms + 1) * logdet_B
        term2 = 0.5 * self.psi_scale * trace_B_inv
        
        iw_penalty = term1 + term2
        iw = iw_penalty.mean() / self.batch_size
        return iw

class LossTerm(gpytorch.mlls.AddedLossTerm):
    """
    A concrete implementation of an AddedLossTerm that simply 
    returns a pre-calculated scalar tensor.
    """
    def __init__(self, loss_tensor, t_index=None):
        self.loss_tensor = loss_tensor
        self.t_index = t_index
        
    def loss(self):
        return self.loss_tensor