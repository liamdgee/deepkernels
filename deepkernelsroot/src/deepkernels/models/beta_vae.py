import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from pydantic import Field, BaseModel, PositiveInt, PositiveFloat, validator
import math

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

    @validator("M")
    def check_M_is_even(cls, v):
        """Optional: Enforce M is even if using Sin/Cos RFF pairs."""
        if v % 2 != 0:
            raise ValueError(f"M (spectral samples) must be even, got {v}")
        return v

class SpectralVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        """
        Args:
            input_dim: Dimension of input features
            n_mixtures: Number of Dirichlet components (K)
            latent_dim: Size of the VAE bottleneck (z)
            M: Number of RFF spectral samples
            beta: Disentanglement factor
            hidden_dim: Width of hidden layers.
        """
        super().__init__()
        self.config = config
        self.input_dim = self.config.input_dim
        self.latent_dim = self.config.latent_dim
        self.K = self.config.k_atoms
        self.beta = self.config.beta
        self.H = self.config.hidden_dim
        self.jitter = self.config.jitter
        self.D = self.config.depth
        self.M = self.config.M
        self.target_rff = 2 * self.M

        #-output dim-#
        self.rff_out = self.K * self.M * 2

        # --- 1. The Encoder (x -> z) ---
        enc_layers = []
        dims = [self.input_dim] + [self.H] * (self.D - 1)
        
        for i in range(len(dims) - 1):
            linear = P.spectral_norm(nn.Linear(dims[i], dims[i+1]))
            enc_layers.append(linear)
            enc_layers.append(nn.SiLU())
            enc_layers.append(nn.LayerNorm(dims[i+1]))
        
        self.encoder = nn.Sequential(*enc_layers)

        #-latent projections-#
        self.fc_mu = nn.Linear(self.H, self.latent_dim)
        self.fc_var = nn.Linear(self.H, self.latent_dim)
        
        #-dkl heads-#
        self.alpha_head = P.spectral_norm(nn.Linear(self.latent_dim, self.K))

        #-Automatic Relevance Determination (ARD) for learnable lengthscales-#
        self.ls_head = P.spectral_norm(nn.Linear(self.latent_dim, self.K * self.input_dim))

        #-Decoder in Reproducing Kernel Hilbert Space (z -> x_hat)-#
        dec_layers = []
        ddims = [self.latent_dim] + [self.H] * (self.D - 1)
        
        for i in range(len(ddims) - 1):
            linear = nn.Linear(ddims[i], ddims[i+1])
            linear = P.spectral_norm(linear)
            dec_layers.append(linear)
            dec_layers.append(nn.SiLU())
            dec_layers.append(nn.LayerNorm(ddims[i+1]))
        
        self.decoder = nn.Sequential(*dec_layers)

        #-RKHS Head-#
        self.rkhs_head = P.spectral_norm(nn.Linear(self.H, self.rff_out))

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        z = self._reparameterize(mu, logvar)

        #-predict dirichlet & kernel params-#
        alpha = F.softplus(self.alpha_head(z)) + self.jitter
        ls = F.softplus(self.ls_head(z)) + self.jitter
        ls = ls.view(-1, self.K, self.input_dim)

        #-recon in rkhs space-#
        decoder_hidden = self.decoder(z)
        recon_rff = self.rkhs_head(decoder_hidden)

        self.z = z

        vae_outputs = {
            "alpha": alpha,          #-Dirichlet-#
            "ls": ls,                #-Kernel-#
            "recon_rff": recon_rff,    #-VAE Loss-#
            "mu": mu,                #-VAE KL-#
            "logvar": logvar,        #-VAE KL-#
            "input": x               #-Testing-#
        }

        return vae_outputs

    def loss(self, vae_outputs, target_rff, beta_divergence_factor_override=None):
        """
        Calculates the spectral reconstruction loss
        Args:
            vae_outputs: dict from forward pass
            target_rff: true random fourier features calculated by GP (shape must match recon_rff from forward pass)
        """
        recon_rff = vae_outputs['recon_rff']
        mu = vae_outputs['mu']
        logvar = vae_outputs['logvar']
        
        #-MSE Recon Loss-#
        #-VAE kernel dream (recon_rff) against GP kernel fourier features-#
        recon_loss = F.mse_loss(recon_rff, target_rff, reduction='mean')

        #-KL Divergence-#
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / recon_rff.size(0)

        beta = beta_divergence_factor_override if beta_divergence_factor_override is not None else self.beta
        
        return recon_loss + beta * kl_loss
    
    def get_target_rff(self, z, omega, pi):
        """
        z: [B, D]
        omega: [K, M, D]
        pi: [B, K]
        """
        proj = (z.unsqueeze(1).unsqueeze(1) * omega.unsqueeze(0)).sum(dim=-1)
        feat_cos = torch.cos(2 * math.pi * proj)
        feat_sin = torch.sin(2 * math.pi * proj)
        features_k = torch.cat([feat_cos, feat_sin], dim=-1)
        target_rff = (features_k * pi.unsqueeze(-1)).sum(dim=1) #-[B, 2M]
        target_rff = target_rff / math.sqrt(self.config.M)
        
        return target_rff



