import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from pydantic import Field, BaseModel, PositiveInt, PositiveFloat, validator
import math
from typing import Optional, Annotated
import torch.nn.utils.spectral_norm as sn

class VAEConfig(BaseModel):
    """
    Configuration for the Dirichlet VAE Base Model.
    """
    # --- Input Dimensions ---
    input_dim: PositiveInt = Field(
        default=128, 
        description="Dimension of input features (D)"
    )
    
    # --- VAE Architecture ---
    latent_dim: PositiveInt = Field(
        default=64, 
        description="Size of the VAE bottleneck (z)"
    )
    hidden_dim: PositiveInt = Field(
        default=128, 
        description="Width of hidden layers in encoder/decoder"
    )
    depth: PositiveInt = Field(
        default=4, 
        description="Number of layers in the deep spectral networks"
    )
    
    # --- Mixture & Spectral Params ---
    k_atoms: PositiveInt = Field(
        default=30, 
        description="Number of Dirichlet components / Experts (K)"
    )
    M: PositiveInt = Field(
        default=256, 
        description="Number of RFF spectral samples (Frequency modes)"
    )
    
    target_rff: PositiveInt = Field(
        default=512, 
        description="Number of RFF targets (2M)"
    )

    # --- Regularization & Stability ---
    beta: PositiveFloat = Field(
        default=3.0, 
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

class Encoder(nn.Module):
    def __init__(self,
                 config: Optional[VAEConfig]=None, 
                 input_dim: Optional[int] = 44,
                 hidden_dims: Optional[list[int]] = None, #-hidden depth dims-#
                 latent_dim: Optional[int] = 16,
                 dropout: Optional[float] = 0.07,
                 k_atoms: Optional[int]=20
        ):
        super().__init__()
        self.config = config or VAEConfig()
        self.input_dim = input_dim or 44
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64, 32]
        self.latent_dim = latent_dim or self.config.latent_dim
        self.k_atoms = self.config.k_atoms
        self.jitter = self.config.jitter or 1e-6

        # --- Deep Encoder Network (x -> h) ---
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
            layers.append(sn(nn.Linear(prev_dim, hdim)))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        
        self.encoder_net = nn.Sequential(*layers)

        # --- Latent Projections (h -> mu, logvar) ---
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_var = nn.Linear(prev_dim, self.latent_dim)

        #-DKL Heads (GP Parameters)-
        #-predict these from Z (the bottleneck) to encourage latent manifold learning-#
        self.alpha_head = sn(nn.Linear(self.latent_dim, self.k_atoms))
        
        #-Lengthscales (K atoms * D input dims)
        self.ls_head = sn(nn.Linear(self.latent_dim, self.k_atoms * self.input_dim))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self.reparameterize(mu, logvar)

        #-GP Parameters
        alpha = F.softplus(self.alpha_head(z)) + self.jitter
        
        #-reshape lengthscale head out [Batch, K * D] -> [Batch, K, D]
        ls_flat = F.softplus(self.ls_head(z)) + self.jitter
        ls = ls_flat.view(-1, self.k_atoms, self.input_dim)

        return z, mu, logvar, alpha, ls