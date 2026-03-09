import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from deepkernels.models.parent import BaseGenerativeModel
import math
import torch.nn.functional as F
import torch.distributions as dist

import gpytorch
import torch

from torch.distributions import LowRankMultivariateNormal, Independent, Normal, kl_divergence
from deepkernels.models.NKN import GPParams

from typing import NamedTuple, List, Union, Optional

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
    gp_params: GPParams
    bottleneck: torch.Tensor
    alpha: torch.Tensor
    alpha_mu: torch.Tensor
    alpha_factor: torch.Tensor
    alpha_diag: torch.Tensor
    gp_features: torch.Tensor
    parameters_per_expert: torch.Tensor
    recon: torch.Tensor
    bandwidth_mod: torch.Tensor
    pi: torch.Tensor
    amp: torch.Tensor
    trend: torch.Tensor
    res: torch.Tensor
    ls: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    lmc_matrices: torch.Tensor


class SpectralDecoder(BaseGenerativeModel):
    def __init__(self,
                 config=None,
                 **kwargs
):
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
        self.num_latents = self.kwargs.get("num_latents", 8)
        self.spectral_compressions = self.kwargs.get("spectral_compressions", [1024, 512, 128, 64])
        self.eps = self.kwargs.get("eps", 1e-3)
        
        self.min_ls = self.kwargs.get("min_ls", 0.05)
        self.max_ls = self.kwargs.get("max_ls", 15.0)
        self.beta = self.kwargs.get("beta", 1.0)
        self.conc_clamp = self.kwargs.get("conc_clamp", 30.0)
        self.large_eps = self.kwargs.get("large_eps", 4e-2)

        split_override = self.kwargs.get("disentangle_split_override", None)
        self.disentangle_split = split_override if split_override is not None else [4, 30, 30]
        
        # ---Safety Check ---
        if sum(self.disentangle_split) != self.bottleneck_dim:
            raise ValueError(
                f"Architecture mismatch! bottleneck_dim is {self.bottleneck_dim}, "
                f"but disentangle_split sums to {sum(self.disentangle_split)} {self.disentangle_split}."
            )
        
        #--# safeguard
        self.num_experts = self.kwargs.get("num_experts", 8)

        layers = []
        current_dim = self.spectral_emb_dim

        for compression in self.spectral_compressions:
            layers.append(torch.nn.utils.spectral_norm(nn.Linear(current_dim, compression)))
            layers.append(nn.LayerNorm(compression))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(self.dropout))
            current_dim = compression
        

        #-current_dim is now 64
        self.compression_network = nn.Sequential(*layers)

        self.ls_head_recon = nn.Sequential(
            nn.Linear(4, self.input_dim), #-where input dim is input to the encoder i.e. num_features in real_x
            nn.Tanh()            
        )
        self.pi_head_recon = nn.Sequential(
            nn.Linear(30, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.Sigmoid()
        )
        self.data_head_recon = nn.Sequential(
            nn.LayerNorm(30),
            nn.utils.spectral_norm(nn.Linear(30, self.input_dim))
        )

        self.expert_variational_heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(self.bottleneck_dim, self.k_atoms)
            ) 
            for _ in range(self.num_experts)
        ])
        
        #-roughly equates to prior assertion that 50% of explanatory power comes from mixture weights
        self.expert_logit_heads = nn.ModuleList([
            nn.Linear(self.k_atoms, self.latent_dim)
            for _ in range(self.num_experts)
        ])

        self.rank=3

        self.mu_alpha = nn.Linear(self.bottleneck_dim, self.k_atoms)
        self.factor_alpha = nn.Linear(self.bottleneck_dim, self.k_atoms * self.rank)
        self.diag_alpha = nn.Linear(self.bottleneck_dim, self.k_atoms)
        
        self.lengthscale_mu = nn.Linear(self.bottleneck_dim, self.k_atoms)
        self.lengthscale_logvar = nn.Linear(self.bottleneck_dim, self.k_atoms)
        self.scale_head_per_expert = torch.nn.utils.spectral_norm(nn.Linear(self.k_atoms, self.num_latents))

        self.register_added_loss_term("lengthscale_kl")
        self.register_added_loss_term("alpha_kl")
        self.register_added_loss_term("recon_kl")

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
        mu_z = getattr(vae_out, 'mu_z', None)
        logvar_z = getattr(vae_out, 'logvar_z', None)
        real_x = getattr(vae_out, 'real_x', None)
        

        spectral_bottleneck = self.compression_network(x)
        
        bottleneck = torch.tanh(spectral_bottleneck)

        recon, trend, amp, residuals = self.disentangle(bottleneck)

        lossterm = self.log_recon_kl(real_x, recon, mu_z, logvar_z, beta=self.beta)

        self.update_added_loss_term("recon_kl", LossTerm(lossterm))

        latent_expert_functions = [variational(bottleneck) for variational in self.expert_variational_heads]

        variational_parameters = torch.stack(latent_expert_functions)

        mu, factor, diag = self.get_alpha_mvn_heads_decoder(bottleneck)

        kl = self.log_alpha_kl_low_rank(mu, factor, diag)

        self.update_added_loss_term("alpha_kl", LossTerm(kl))
        
        ls_sample, kl_div = self.predict_lengthscale_and_log_kl(bottleneck, ls_pred_prior, ls_logvar_prior)

        self.update_added_loss_term("lengthscale_kl", LossTerm(kl_div.sum()))

        bandwidth_mod = torch.sigmoid(self.scale_head_per_expert(ls_sample)) * 2.0

        latent_features_per_expert = [head(var_state) for head, var_state in zip(self.expert_logit_heads, latent_expert_functions)]

        gp_features = torch.stack(latent_features_per_expert, dim=1)

        vae_out = DecoderOutput(
            gp_params=vae_out.gp_params,
            bottleneck=bottleneck,
            alpha=vae_out.local_conc,
            alpha_mu=mu,
            alpha_factor=factor,
            alpha_diag=diag,
            gp_features=gp_features,
            parameters_per_expert=variational_parameters,
            recon=recon,
            bandwidth_mod=bandwidth_mod,
            pi=vae_out.pi,
            amp=amp,
            trend=trend,
            res=residuals,
            ls=ls_sample,
            mu_z=mu_z,
            logvar_z=logvar_z,
            lmc_matrices=vae_out.lmc_matrices
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

    def log_recon_kl(self, x, recon, mu_z, logvar_z, beta=1.0):
        loss = F.mse_loss(recon, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        lossterm = loss + (beta * kl)
        return lossterm
    
    
    def disentangle(self, bottleneck):
        disentangle_split = self.disentangle_split or [4, 30, 30]
        z_ls, z_pi, z_dt = torch.split(bottleneck, disentangle_split, dim=-1)
        
        trend     = self.ls_head_recon(z_ls) # (-1 to 1)
        amplitude = self.pi_head_recon(z_pi) # (0 to 1)
        residual  = self.data_head_recon(z_dt) # Unbounded
        
        # Physics-Informed Composition: Base trend + (Local Amplitude * High Frequency Details)
        recon = trend + (amplitude * residual)
        return recon, trend, amplitude, residual
    

    def log_alpha_kl_low_rank(self, mu, chol, diag):
        batch_size = mu.size(0)
        r = self.kwargs.get("alpha_factor_rank", 3)
        k = self.k_atoms
        
        factor = chol.view(batch_size, k, r)
        
        noise_factor = torch.zeros(batch_size, k, r, device=mu.device)
        for j in range(r):
            start, end = j * (k // r), (j + 1) * (k // r)
            noise_factor[:, start:end, j] = 1.0
        
        cov_factor = factor + noise_factor 

        cov_diag = torch.clamp(diag + self.jitter, min=0.025)
        
        q_dist = torch.distributions.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=cov_factor,
            cov_diag=cov_diag
        )
        
        prior_logit = self.inverse_softplus(1.0).to(mu.device)
        
        p_dist = torch.distributions.LowRankMultivariateNormal(
            loc=torch.full_like(mu, prior_logit), 
            cov_factor=torch.zeros_like(cov_factor),
            cov_diag=torch.ones_like(cov_diag)
        )
        
        return kl_divergence(q_dist, p_dist)
    
    def dirichlet_sample(self, alpha):
        alpha = F.softplus(alpha)
        alpha = torch.clamp(alpha, min=self.large_eps)
        q_alpha= torch.distributions.Dirichlet(alpha)
        pi_sample = q_alpha.rsample()
        return pi_sample

    def predict_lengthscale_and_log_kl(self, bottleneck, ls_pred_prior=None, ls_logvar_prior=None):
        """
        Computes the posterior over lengthscales, applies the reparameterisation trick, 
        and logs the KL divergence against a Log-Normal prior.
        """
        mu = self.lengthscale_mu(bottleneck)
        sigma = torch.exp(0.5 * self.lengthscale_logvar(bottleneck)) + self.eps
        
        #- Variational Posterior q(log_l | x) -- normal in log space (lognormal)
        q_log_ls = dist.Normal(mu, sigma)
        log_ls_sample = q_log_ls.rsample()
        
        safe_log_ls = torch.clamp(
            log_ls_sample, 
            min=math.log(self.min_ls), 
            max=math.log(self.max_ls)
        )
        
        ls_sample = torch.exp(safe_log_ls)
        if ls_pred_prior is not None and ls_logvar_prior is not None:
            prior_loc = torch.log(ls_pred_prior)
            prior_scale = torch.exp(0.5 * ls_logvar_prior) + self.eps
        else:
            prior_loc = torch.zeros_like(mu)
            prior_scale = torch.ones_like(sigma) * 2.5
        
        #- Prior p(log_l) -- expect lengthscales are centered around exp(0) = 1.0 -#
        
        p_log_ls = dist.Normal(prior_loc, prior_scale)
        
        kl_div = dist.kl_divergence(q_log_ls, p_log_ls)
        
        return ls_sample, kl_div.sum(dim=-1)