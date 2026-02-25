import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from deepkernels.models.parent import BaseGenerativeModel

import torch.nn.functional as F
import torch.distributions as dist

import gpytorch
import torch

from torch.distributions import LowRankMultivariateNormal, Independent, Normal, kl_divergence
from deepkernels.models.NKN import GPParams

from typing import NamedTuple

class LossTerm(gpytorch.mlls.AddedLossTerm):
    """
    A concrete implementation of an AddedLossTerm that simply 
    returns a pre-calculated scalar tensor.
    """
    def __init__(self, loss_tensor):
        self.loss_tensor = loss_tensor
        
    def loss(self):
        return self.loss_tensor

class DecoderOutput(NamedTuple):
    """Structured output for the SpectralDecoder."""
    bottleneck: torch.Tensor
    alpha: torch.Tensor
    alpha_mu: torch.Tensor
    alpha_factor: torch.Tensor
    alpha_diag: torch.Tensor
    gp_features: torch.Tensor
    gp_params: GPParams
    parameters_per_expert: torch.Tensor
    recon: torch.Tensor
    bandwidth_mod: torch.Tensor
    pi: torch.Tensor
    amp: torch.Tensor
    trend: torch.Tensor
    res: torch.Tensor
    ls: torch.Tensor

class SpectralDecoder(BaseGenerativeModel):
    def __init__(self, 
                 input_dim=30,       # Output shape (reconstruction)
                 spectral_dim=256,   # Features per cluster (M*2)
                 num_clusters=30,
                 spectral_emb_dim=2048,
                 input_dim_data=30,
                 bottleneck_dim=64,
                 num_experts=8,
                 k_atoms=30,
                 latent_dim=16,
                 hidden_dims=None,
                 dropout=0.1):
        super().__init__()
        
        spectral_compressions = [1024, 512, 128, 64]
        self.num_experts = num_experts
        self.k_atoms = k_atoms
        current_dim = spectral_emb_dim
        self.latent_dim = latent_dim
        layers = []
        for compression in spectral_compressions:
            layers.append(torch.nn.utils.spectral_norm(nn.Linear(current_dim, compression)))
            layers.append(nn.LayerNorm(compression))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            current_dim = compression
        

        #-current_dim is now 64
        self.compression_network = nn.Sequential(*layers)

        self.ls_head_recon = nn.Sequential(
            nn.Linear(4, input_dim_data),
            nn.Tanh()            
        )
        self.pi_head_recon = nn.Sequential(
            nn.Linear(30, input_dim_data),
            nn.LayerNorm(input_dim_data),
            nn.Sigmoid()
        )
        self.data_head_recon = nn.Sequential(
            nn.LayerNorm(30),
            nn.utils.spectral_norm(nn.Linear(30, input_dim_data))
        )

        self.expert_variational_heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(bottleneck_dim, k_atoms)
            ) 
            for _ in range(num_experts)
        ])
        
        #-roughly equates to prior assertion that 50% of explanatory power comes from mixture weights
        self.expert_logit_heads = nn.ModuleList([
            nn.Linear(k_atoms, latent_dim)
            for _ in range(num_experts)
        ])

        self.rank=3
        self.mu_alpha = nn.Linear(bottleneck_dim, k_atoms)
        self.factor_alpha = nn.Linear(bottleneck_dim, k_atoms * self.rank)
        self.diag_alpha = nn.Linear(bottleneck_dim, k_atoms)
       

        self.lengthscale_mu = nn.Linear(bottleneck_dim, k_atoms)
        self.lengthscale_logvar = nn.Linear(bottleneck_dim, k_atoms)
        self.scale_head_per_expert = torch.nn.utils.spectral_norm(nn.Linear(k_atoms, num_experts))

        self.register_added_loss_term("lengthscale_kl")
        self.register_added_loss_term("alpha_kl")

        self._init_weights()
    
    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)


    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        """
        Args:
            spectral_features: [Batch, K*M*2] OR [Batch, K, M*2] as 'x' (embedded dim = 2048)
            vae_out: dirichlet_out
        """

        ls_pred_prior = getattr(vae_out, 'ls_pred', None)

        ls_logvar_prior = getattr(vae_out, 'ls_logvar', None)
        
        spectral_bottleneck = self.compression_network(x)
        
        bottleneck = torch.tanh(spectral_bottleneck)

        recon, trend, amp, residuals = self.disentangle(bottleneck)

        latent_expert_functions = [variational(bottleneck) for variational in self.expert_variational_heads]

        variational_parameters = self.stack_features(latent_expert_functions)

        mu, factor, diag = self.get_alpha_mvn_heads_decoder(bottleneck)

        alpha_logits = self.lowrankmultivariatenorm(mu, factor, diag)

        self.log_alpha_kl_low_rank(mu, factor, diag)

        pi_sample = self.dirichlet_sample(alpha_logits)
        
        ls_sample = self.predict_lengthscale_and_log_kl(bottleneck, ls_pred_prior, ls_logvar_prior)

        bandwidth_mod = torch.sigmoid(self.scale_head_per_expert(ls_sample)) * 2.0

        latent_features_per_expert = [head(var_state) for head, var_state in zip(self.expert_logit_heads, latent_expert_functions)]

        gp_features = torch.stack(latent_features_per_expert, dim=1)

        vae_out = DecoderOutput(
            bottleneck=bottleneck,
            alpha=alpha_logits,
            alpha_mu=mu,
            alpha_factor=factor,
            alpha_diag=diag,
            gp_features=gp_features,
            gp_params=vae_out.gp_params,
            parameters_per_expert=variational_parameters,
            recon=recon,
            bandwidth_mod=bandwidth_mod,
            pi=pi_sample,
            amp=amp,
            trend=trend,
            res=residuals,
            ls=ls_sample
        )
        
        return vae_out
    
    def get_alpha_mvn_heads_decoder(self, bottleneck):
        """
        Returns parameters for a Low-Rank Multivariate Normal.
        Covariance is parameterized as: Sigma = V @ V.T + diag
        
        Returns:
            mu: Mean vector [Batch, k_atoms]
            factor (V): Dense low-rank factor matrix [Batch, k_atoms, rank_r]
            diag (D): Strictly positive diagonal variance [Batch, k_atoms]
        """

        mu = self.mu_alpha(bottleneck)
        factor = self.factor_alpha(bottleneck).view(-1, self.k_atoms, self.rank)
        diag = F.softplus(self.diag_alpha(bottleneck)) + 1e-6
        return mu, factor, diag
    
    
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
        
        p_dist = LowRankMultivariateNormal(
            loc=torch.zeros_like(mu), 
            cov_factor=torch.zeros_like(cov_factor),
            cov_diag=torch.ones_like(cov_diag)
        )
        
        kl = kl_divergence(q_dist, p_dist)
        self.update_added_loss_term("alpha_kl", LossTerm(kl))

        return self

    def predict_lengthscale_and_log_kl(self, bottleneck, ls_pred_prior=None, ls_logvar_prior=None, eps=1e-4):
        """
        Computes the posterior over lengthscales, applies the reparameterisation trick, 
        and logs the KL divergence against a Log-Normal prior.
        """
        mu = self.lengthscale_mu(bottleneck)
        sigma = torch.exp(0.5 * self.lengthscale_logvar(bottleneck)) + eps
        
        #- Variational Posterior q(log_l | x) -- normal in log space (lognormal)
        q_log_ls = dist.Normal(mu, sigma)
        log_ls_sample = q_log_ls.rsample()
        
        if ls_pred_prior is not None and ls_logvar_prior is not None:
            prior_loc = torch.log(ls_pred_prior + eps)
            prior_scale = torch.exp(0.5 * ls_logvar_prior) + eps
        else:
            prior_loc = torch.zeros_like(mu)
            prior_scale = torch.ones_like(sigma) * 2.5
        #- Prior p(log_l) -- expect lengthscales are centered around exp(0) = 1.0 -#
        
        p_log_ls = dist.Normal(prior_loc, prior_scale)
        
        kl_div = dist.kl_divergence(q_log_ls, p_log_ls)
        
        self.update_added_loss_term("lengthscale_kl", LossTerm(kl_div.sum()))
        
        #-transform to lengthscale space-#
        ls_sample = torch.exp(log_ls_sample)
        
        ls_sample = torch.clamp(ls_sample, min=eps, max=100.0)
        
        return ls_sample