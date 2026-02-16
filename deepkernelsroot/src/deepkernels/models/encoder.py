import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from pydantic import Field, BaseModel, PositiveInt, PositiveFloat, validator
import math
from typing import Optional, Annotated
import torch.nn.utils.spectral_norm as sn
from typing import Tuple, Optional, TypeAlias, Tuple, Union
import torch.nn.functional as F
import torch.distributions as dist

class VAEConfig(BaseModel):
    """
    Configuration for the Dirichlet VAE Base Model.
    """
    # --- Input Dimensions ---
    input_dim: PositiveInt = Field(
        default=30, 
        description="Dimension of input features (D)"
    )
    
    # --- VAE Architecture ---
    latent_dim: PositiveInt = Field(
        default=16,
        le=64,
        ge=8, 
        description="Size of the VAE bottleneck (z)"
    )
    hidden_dim: Union[int, list[int]] = [128, 64, 32]
    
    # --- Mixture & Spectral Params ---
    k_atoms: PositiveInt = Field(
        default=30, 
        description="Number of Dirichlet components / Experts (K)"
    )
    M: PositiveInt = Field(
        default=128,
        ge=32,
        le=512,
        description="Number of RFF spectral samples (Frequency modes)"
    )
    
    target_rff: PositiveInt = Field(
        default=256,
        le=1024,
        ge=64,
        description="Number of RFF targets (2M)"
    )

    # --- Regularization & Stability ---
    kl_beta: PositiveFloat = Field(
        default=0.5,
        le=1.5,
        ge=0.005, 
        description="Beta-VAE disentanglement factor"
    )
    jitter: float = Field(
        default=1e-6, 
        description="Numerical stability term for matrix operations"
    )

    class Config:
        extra = "forbid" 
        json_schema_extra = {
            "example": {
                "input_dim": 128,
                "k_atoms": 50,
                "M": 512,
                "latent_dim": 32,
                "beta": 4.0
            }
        }

class RecurrentEncoder(nn.Module):
    def __init__(self,
                 config: Optional[VAEConfig]=None,
                 input_dim=30
        ):
        super().__init__()
        self.config = config or VAEConfig()
        
        self.input_dim = input_dim
        
        hidden_dims = [128, 64, 32]
        self.hidden_dims = hidden_dims
    
        self.latent_dim = 16
        self.k_atoms = 30
        self.M = 128
        self.jitter = 1e-6
        self.rank = 3
        self.dropout = 0.1
        self.total_features = 900
        self.spectral_input_dim = 7680 #~7680
        self.spectral_emb_dim = 128
        self.spectral_compressor = torch.nn.utils.spectral_norm(nn.Linear(self.spectral_input_dim, self.spectral_emb_dim))

        # --- Deep Encoder Network (x -> h) ---
        layers = []
        #-- current_dim = Data(D) + LogPi(K) + SpectralEmbedding(D) -#
        current_dim = 188 #--188 = 37 + 30 + 121 (spectral emb_dim)
        
        for hdim in hidden_dims:
            layers.append(torch.nn.utils.spectral_norm(nn.Linear(current_dim, hdim)))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(self.dropout))
            current_dim = hdim
        
        self.encoder_network = nn.Sequential(*layers)

        # --- Latent Projections (h -> mu, logvar) # 32 -> 16---
        self.mu_latent_z = nn.Linear(current_dim, self.latent_dim)
        self.var_latent_z = nn.Linear(current_dim, self.latent_dim) #-> to latent

        #-kernel reconstruction in real time-#
        #- we want chol.chol.t() + diag for positive definite kernel outputs
        self.mu_alpha = nn.Linear(self.latent_dim, self.k_atoms)
        self.chol_alpha = nn.Linear(self.latent_dim, 90)
        self.diag_alpha = nn.Linear(self.latent_dim, self.k_atoms) #-diag_alpha is logvar diagonal-#
        
        #-kernel lengthscale-#
        self.ls_head = nn.Linear(self.latent_dim, 30*30)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        return mu
    
    def multivariate_projection(self, mu, factor, diag):
        """for alpha params: input projections from three alpha heads"""
        mvn = dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=factor,
            cov_diag=diag
        )
        logits = mvn.rsample()
        alpha = F.softplus(logits) + self.jitter
        
        return alpha, mvn

    def forward(self, real_x, step=3, pi: Optional[torch.Tensor] = None, spectral_features: Optional[torch.Tensor]=None):
        """
        Args:
            real_x: Input features [Batch, Input_Dim]
            pi: Dirichlet mixture weights [Batch, K_Atoms]. 
            spectral_features: [Batch, K * M * 2] --> unflatten with ' .view(B, K, -1). '
        """
        #-for first iter: Handle missing args and scale by 2.1x-#

        if pi is None:
            pi = torch.full((real_x.size(0), self.k_atoms), 1.0/self.k_atoms, device=real_x.device)
            pi = pi + (torch.randn_like(pi) * 0.01)
            pi = F.softmax(pi, dim=-1)
        
        if spectral_features is None:
            spectral_features = torch.zeros(
                real_x.size(0), 
                self.spectral_input_dim, 
                device=real_x.device, 
                dtype=real_x.dtype
            )
            
        #-processing steps:-#
        log_pi = torch.log(pi + self.jitter)

        #-compression of spectral features--tanh activation -> [256, 7680] -> [256, 128]
        spectral_emb = self.spectral_compressor(spectral_features)
        spectral_emb = torch.tanh(spectral_emb)

        #-concat [Data, logpi, spectral_emb] [30, 30, 128] i think?
        weights = torch.cat([real_x, log_pi, spectral_emb], dim=-1)

        #-network forward passes-#
        h = self.encoder_network(weights)
        mu = self.mu_latent_z(h)
        logvar = self.var_latent_z(h)
        z = self.reparameterize(mu, logvar) #-latent reparameterisation-> [Batch, 16]
        
        mu_alpha_out = self.mu_alpha(z)
        chol = self.chol_alpha(z).view(-1, 30, 3)
        diag = F.softplus(self.diag_alpha(z)) + self.jitter

        #-sample from mvn-#
        alpha, mvn = self.multivariate_projection(mu_alpha_out, chol, diag)
        
        #-Kernel parameters- [256, 30 * 30] -> [256, 30, 30]-#
        ls = self.ls_head(z)
        ls = F.softplus(ls) + self.jitter
        ls = ls.view(-1, 30, 30) #- [B, K, D]

        return z, alpha, ls, mu, logvar