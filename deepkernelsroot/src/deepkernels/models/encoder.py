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

class RecurrentEncoder(nn.Module):
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
        current_dim = self.input_dim + self.k_atoms
        
        for hdim in hidden_dims:
            layers.append(sn(nn.Linear(current_dim, hdim)))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hdim
        
        self.encoder_network = nn.Sequential(*layers)

        # --- Latent Projections (h -> mu, logvar) ---
        self.fc_mu = nn.Linear(current_dim, self.latent_dim)
        self.fc_var = nn.Linear(current_dim, self.latent_dim)

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

    def forward(self, real_x: torch.Tensor, pi: Optional[torch.Tensor] = None, pi_init_scale_factor: float = 2.1):
        """
        Args:
            real_x: Input features [Batch, Input_Dim]
            pi: Dirichlet mixture weights [Batch, K_Atoms]. 
                If None (1st iteration), it is initialized to zeros.
        """
        #-for first iter-#
        if pi is None:
            p_uniform = 1.0 / self.k_atoms
            pi_init = p_uniform * pi_init_scale_factor #-flexible scale for first cluster assignment-#
            pi = torch.full((real_x.size(0), self.k_atoms), pi_init, device=real_x.device)
        
        #-log transform-#
        log_pi = torch.log(pi + self.jitter)
        
        #-concat data input and probabilistic weights in log space-#
        x = torch.cat([real_x, log_pi], dim=-1)

        h = self.encoder_network(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self.reparameterize(mu, logvar) #-latent reparameterisation-#

        #-Dirichlet Parameters-#
        alpha = F.softplus(self.alpha_head(z)) + self.jitter
        
        #-Kernel parameters- [Batch, K * D] -> [Batch, K, D]-#
        ls_flat = F.softplus(self.ls_head(z)) + self.jitter
        ls = ls_flat.view(-1, self.k_atoms, self.input_dim)

        return z, mu, logvar, alpha, ls