import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from pydantic import Field, BaseModel, PositiveInt, PositiveFloat, validator
import math
from typing import Optional, Annotated

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
        default=16, 
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

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(), 
            P.spectral_norm(nn.Linear(dim, dim)),
            nn.Dropout(dropout),
            
            nn.LayerNorm(dim),
            nn.SiLU(),
            P.spectral_norm(nn.Linear(dim, dim)),
            nn.Dropout(dropout)
        )
        
        # Learnable skip connection scaling (helps initial convergence)
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.skip_scale * self.net(x)

class Encoder(nn.Module):
    def __init__(self,
                 config: Optional[VAEConfig]=None, 
                 input_dim: Optional[int]=None,
                 hidden_dim: Optional[int]=None,
                 depth: Optional[int]=None,
                 beta: Optional[Annotated[float, Field(ge=0.5, le=8.0)]] = 3.0,
                 latent_dim: Optional[int] = None
        ):
        super().__init__()
        self.input_dim = input_dim if input_dim else config.input_dim
        self.hidden_dim = hidden_dim if hidden_dim else config.hidden_dim
        self.latent_dim = latent_dim if latent_dim else config.latent_dim
        self.depth = depth if depth else config.depth

        # --- Deep Encoder Network (x -> h) ---
        enc_layers = []
        dims = [self.input_dim] + [self.hidden_dim] * (self.depth - 1)
        
        for i in range(len(dims) - 1):
            lipschitzlinear = P.spectral_norm(nn.Linear(dims[i], dims[i+1]))
            enc_layers.append(lipschitzlinear)
            enc_layers.append(nn.SiLU())
            enc_layers.append(nn.LayerNorm(dims[i+1]))
        
        self.encoder_net = nn.Sequential(*enc_layers)

        # --- Latent Projections (h -> mu, logvar) ---
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar