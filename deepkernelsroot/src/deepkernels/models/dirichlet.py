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
from linear_operator.operators import RootLinearOperator, MatmulLinearOperator, DiagLinearOperator, AddedDiagLinearOperator

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.models.NKN import KernelNetwork, GPParams
from deepkernels.kernels.keops import CustomLaplacePrior
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
    n_data: float = 76674.0  # Kept as float for division stability
    
    # --- Hyperparameters ---
    dropout: float = 0.05
    gamma_concentration_init: float = 1.5417
    noise_scalar: float = 0.316
    psi_scale: float = 1.0
    
    # --- Numerical Bounds & Tolerances ---
    jitter: float = 1e-6
    eps: float = 1e-3
    large_eps: float = 4e-2
    posterior_dirichlet_epsilon: float = 4e-5
    conc_clamp: float = 30.0
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
    M_series: int = 8
    num_primitives: int = 5
    individual_kernel_dim_out: int = 32


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

        #-global & dynamic:
        self.input_dim = kwargs.get("input_dim", 30)
        self.n_data = kwargs.get('n_data', 76674.0)
        #-hypernetworks
        self.compress_spectral_features_head = torch.nn.utils.spectral_norm(nn.Linear(self.k_atoms * self.M * 2, self.spectral_emb_dim))
        
        self.bottleneck_mixer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.latent_dim, self.bottleneck_dim)), 
            nn.LayerNorm(self.bottleneck_dim), 
            nn.SiLU()
        )
        
        self.kernel_network = KernelNetwork(config=self.config)


        scale_constraint = gpytorch.constraints.GreaterThan(1e-4)
        jitter_constraint = gpytorch.constraints.GreaterThan(1e-6)
        positive_constraint = gpytorch.constraints.Positive()

        
        mu_constraint = gpytorch.constraints.Interval(lower_bound=-30.0, upper_bound=20.0)
        sigma_constraint = gpytorch.constraints.Interval(lower_bound=0.01, upper_bound=5.0)
        ls_constraint = gpytorch.constraints.Interval(lower_bound=self.min_ls, upper_bound=self.max_ls)
        atom_mu_constraint = gpytorch.constraints.Interval(lower_bound=-30.0, upper_bound=20.0)
        atom_sigma_constraint = gpytorch.constraints.Interval(lower_bound=0.01, upper_bound=5.0)
        lmc_constraint = gpytorch.constraints.Interval(lower_bound=-5.0, upper_bound=5.0)
        
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
        self.register_parameter(name="raw_gamma", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_gamma", scale_constraint)

        #-global hierarchical dist (H)-# 
        self.register_parameter(name="raw_h_mu", parameter=nn.Parameter(torch.zeros(1, 1, self.latent_dim)))
        self.register_constraint("raw_h_mu", mu_constraint)

        self.register_parameter(name="raw_h_sigma", parameter=nn.Parameter(torch.zeros(1, 1, self.latent_dim)))
        self.register_constraint("raw_h_sigma", sigma_constraint)
        
        #-atoms for spectral features (gaussian)-#
        self.register_parameter(name="raw_atom_loc", parameter=nn.Parameter(torch.randn(self.k_atoms, 1, self.latent_dim) * 2 * math.sqrt(0.1)))
        self.register_constraint("raw_atom_loc", atom_mu_constraint)

        self.register_parameter(name="raw_atom_scale", parameter=nn.Parameter(torch.randn(self.k_atoms, 1, self.latent_dim) * 0.1))
        self.register_constraint("raw_atom_scale", atom_sigma_constraint)

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
            CustomLaplacePrior(loc=0.0025, scale=0.37), 
            lambda m: m.atom_scale,
            lambda m, v: None
        )

        self.register_prior(
            "gamma_prior", 
            GammaPrior(concentration=1.5, rate=0.5), 
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

        #-initialisations-#
        random_log_noise_mu = torch.abs(torch.randn(self.k_atoms, 1, self.latent_dim)) * self.noise_scalar
        target_atom_mu = 2 * torch.sqrt(random_log_noise_mu)
        target_atom_sigma = torch.exp(torch.randn(self.k_atoms, 1, self.latent_dim) * self.noise_scalar)
        
        target_lmc = torch.randn(self.k_atoms, self.num_latents) * 0.1
        target_lmc_var = torch.ones(self.k_atoms) + (torch.rand(self.k_atoms) * 0.1)

        self.initialize(
            raw_q_a_global=torch.tensor(0.0),
            raw_q_b_global=torch.tensor(0.0),
            raw_gamma=torch.tensor(0.0),
            raw_h_mu=torch.tensor(0.0),
            raw_h_sigma=torch.tensor(0.0),
            raw_lengthscale_uncertainty=torch.tensor(0.0),
            raw_atom_loc=torch.randn_like(target_atom_mu) * 0.05,
            raw_atom_scale=torch.randn_like(target_atom_sigma) * 0.05, 
            raw_lmc_var=torch.randn_like(target_lmc_var) * 0.05,
            raw_lmc_matrix=torch.randn_like(target_lmc) * 0.05
        )
    
    #-properties for params -#

    @property
    def q_a_global(self): return self.raw_q_a_global_constraint.transform(self.raw_q_a_global)
        
    @property
    def q_b_global(self): return self.raw_q_b_global_constraint.transform(self.raw_q_b_global)

    @property
    def gamma(self): return self.raw_gamma_constraint.transform(self.raw_gamma)
    
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

    #-helper-#
    def _safe_tensor(self, value):
        """Helper to ensure we always pass a tensor to the inverse transform"""
        return value if torch.is_tensor(value) else torch.tensor(value)
    
    def forward(self, x, vae_out, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params) -> DirichletOutput:
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
            alpha_mu = torch.zeros(batch_size, self.k_atoms, device=device)
        
        
        alpha_diag = params.get('alpha_diag', None)
        if alpha_diag is None and vae_out is not None:
            alpha_diag = getattr(vae_out, 'diag', None)
        
        if alpha_diag is None:
            alpha_diag = torch.ones(batch_size, self.k_atoms, device=device)
        
        
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
        
        
        beta, gamma_conc, global_kl = self.global_stick_breaking_kumaraswamy()

        self.update_added_loss_term("global_divergence", LossTerm(global_kl, t_index=t))
    
        bottleneck, gate, gp_params = self.run_neural_nets_dirichlet(x)
        
        local_conc = self.get_local_evidence(alpha_mu, alpha_factor, alpha_diag) #-alpha reparameterisation-#

        pi, local_kl = self.dirichlet_posterior_inference_and_log_local_loss(x, gamma_conc, beta, local_conc) #-< alpha is here as local conc-#

        B_mat_current_state, W_consensus = self.coregionalisation_matrix(pi)

        iw_loss = self.inverse_wishart_penalty(B_mat_current_state)

        self.update_added_loss_term("inverse_wishart", LossTerm(iw_loss, t_index=t))

        self.update_added_loss_term("local_divergence", LossTerm(local_kl, t_index=t))
        
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
            local_conc=local_conc,
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
            lmc_matrices=W_consensus
        )
    
    
    def numerically_stable_gamma(self, gamma_concentration_init):
        raw = float(gamma_concentration_init)
        if raw <= 0:
            raise ValueError("Gamma concentration must be strictly positive.")
        if raw > 20.0:
            return raw
        safe = math.log(math.expm1(raw))
        return safe
    def get_omega(self, bw, **params):
        # 1. h_mu is [1, 1, 16] -> Already perfect
        h = self.h_mu
        
        # 2. atom_loc is [30, 1, 16] -> Swap to [1, 30, 16]
        a = self.atom_loc.transpose(0, 1)
        
        # 3. bw is [128, 30, 16] -> Already perfect
        # 4. noise_weights is [30, 128, 16] -> Swap to [128, 30, 16]
        w = self.noise_weights.transpose(0, 1)
        
        # 5. Math is now perfectly aligned:
        # [128, 30, 16] * [128, 30, 16] = [128, 30, 16]
        dynamic_part = bw * w
        
        # [1, 1, 16] + [1, 30, 16] + [128, 30, 16] = [128, 30, 16]
        omega = h + a + dynamic_part
        
        return torch.clamp(omega, -100.0, 100.0)
    
    def coregionalisation_matrix(self, pi):
        batch_size = pi.size(0)
        pi_bc = pi.unsqueeze(-1)
        W_current_state = pi_bc * self.lmc_matrix
        W_consensus = W_current_state.mean(dim=0)
        cholesky_jitter = 1e-4 
        v = self.lmc_var + cholesky_jitter
        v_bc = v.unsqueeze(0).expand(batch_size, -1)
        lazyroot = RootLinearOperator(W_current_state)
        lazydiag = DiagLinearOperator(v_bc)
        B_lazy = AddedDiagLinearOperator(lazydiag, lazyroot)
        return B_lazy.to_dense(), W_consensus
    
    def random_fourier_features(self, z, omega, pi, **params):
        """inputs latent dim z"""
        noise_bias = self.noise_bias
        B, D = z.shape
        if pi is None:
            pi = torch.full((B, self.k_atoms), 1.0/self.k_atoms, device=z.device)
            if self.training:
                pi = pi + (torch.randn_like(pi) * 0.01)
            pi = F.softmax(pi, dim=-1)
        if omega.dim() == 4:
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) 
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W).view(B, self.k_atoms, self.M)
        
        proj = proj + noise_bias.unsqueeze(0)
        scale = 1.0 / math.sqrt(self.M)

        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale

        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl

        feats = torch.stack([cos_proj, sin_proj], dim=-1)
        return feats.flatten(1)
    

    def global_stick_breaking_kumaraswamy(self, **kwargs):
        q_a = self.q_a_global + self.stick_breaking_epsilon
        q_b = self.q_b_global + self.stick_breaking_epsilon
        u = torch.rand_like(q_a).clamp(self.uniform_dist_clamp, 1.0 - self.uniform_dist_clamp)
        
        #-Inverse CDF transform-: fomula: v = (1 - (1 - u)^(1/b))^(1/a)
        qv_global = (1.0 - (1.0 - u).pow(1.0 / q_b)).pow(1.0 / q_a)
        qv_global = qv_global.clamp(self.uniform_dist_clamp, 1.0 - self.uniform_dist_clamp)
        
        #log_qv = ( --for monte carlo kl-#
        #    torch.log(q_a) + torch.log(q_b) 
         #   + (q_a - 1.0) * torch.log(qv_global) 
          #  + (q_b - 1.0) * torch.log((1.0 - qv_global.pow(q_a)).clamp(min=8e-8))
        #).sum()
        
        gamma_conc = self.gamma + self.stick_breaking_epsilon
        
        #log_pv = (
        #    torch.log(gamma_conc) 
        #    + (gamma_conc - 1.0) * torch.log(1.0 - qv_global)
        #).sum()

        
        log_v = torch.log(qv_global)
        log_1_minus_v = torch.log(1.0 - qv_global)
        pad_log_1_minus_v = F.pad(log_1_minus_v, (1, 0), value=0.0)
        log_pi_k = log_v + torch.cumsum(pad_log_1_minus_v, dim=-1)[..., :-1]
        log_pi_last = torch.cumsum(pad_log_1_minus_v, dim=-1)[..., -1:]
        
        log_beta = torch.cat([log_pi_k, log_pi_last], dim=-1)
        beta = torch.exp(log_beta)
        
        beta = torch.clamp(beta, min=1e-7)
        beta = beta / beta.sum(dim=-1, keepdim=True)

        euler_gamma = 0.5772156649
        psi_b = torch.digamma(q_b)
        
        kl = (
            ((q_a - 1.0) / q_a) * (-euler_gamma - psi_b - 1.0 / q_b) 
            + torch.log(q_a * q_b) 
            + torch.log(gamma_conc) 
            - (q_b - 1.0) / q_b
        )
        
        taylor_sum = 0
        M_series = self.M_series
        for m in range(1, M_series + 1):
            taylor_sum += 1.0 / (m * (m * q_b + q_a))
            
        kl += (gamma_conc - 1.0) * q_b * taylor_sum
        global_kl = kl.sum(dim=-1) / self.n_data
        
        return beta, gamma_conc, global_kl

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
            
            # Perfect Broadcast: [1, 30, 16] * [128, 30, 1] -> [128, 30, 16]
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
    
    def dirichlet_posterior_inference_and_log_local_loss(self, x, gamma_conc, beta, local_conc):
        
        prior_conc = (gamma_conc * beta) + self.large_eps
        prior_conc = torch.clamp(prior_conc, min=self.large_eps, max=self.conc_clamp)
        prior_conc = prior_conc.unsqueeze(0).expand(x.size(0), -1)

        post_conc = prior_conc + local_conc
        post_conc = torch.clamp(post_conc, min=self.large_eps, max=self.conc_clamp)

        dist_prior = dist.Dirichlet(prior_conc)
        dist_post = dist.Dirichlet(post_conc)
        pi_posterior = torch.clamp(dist_post.rsample(), min=self.posterior_eps)
        pi_posterior = pi_posterior / pi_posterior.sum(dim=-1, keepdim=True)
        local_divergence = torch.distributions.kl_divergence(dist_post, dist_prior)
        kl = local_divergence.mean()
        return pi_posterior, kl
    
    def get_local_evidence(self, mu, factor, diag):
        """NB: this is actually alpha logits -- do not recalc in decoder from same value"""
        if diag is None:
            diag = torch.full_like(mu, self.eps)
        
        safe_diag = F.softplus(diag) + self.jitter

        if factor.dim() == 2:
            factor = factor.view(mu.shape[0], mu.shape[1], -1)
        
        dist = torch.distributions.LowRankMultivariateNormal(mu, factor, safe_diag)
        alpha_logits = dist.rsample()
        local_conc = F.softplus(alpha_logits) + self.jitter
        
        return local_conc
    def inverse_wishart_penalty(self, B_dense):
        batch_size = B_dense.size(0)
        
        # 1. Dense LogDet (Extremely fast and stable for small KxK)
        # slogdet returns (sign, logabsdet). Since B is positive definite, sign is 1.
        _, logdet_B = torch.linalg.slogdet(B_dense) 
        
        # 2. Dense Solve (No GPyTorch CG approximations)
        identity = torch.eye(self.k_atoms, dtype=B_dense.dtype, device=B_dense.device)
        identity = identity.expand(batch_size, self.k_atoms, self.k_atoms)
        
        B_inverse = torch.linalg.solve(B_dense, identity)
        
        trace_B_inv = torch.diagonal(B_inverse, dim1=-2, dim2=-1).sum(-1)
        
        term1 = 0.5 * (self.pr_nu + self.k_atoms + 1) * logdet_B
        term2 = 0.5 * self.psi_scale * trace_B_inv
        iw_penalty = term1 + term2
        
        return iw_penalty.mean()

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