import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import math
from typing import Optional, Annotated
import torch.nn.utils.spectral_norm as sn
from typing import Tuple, Optional, TypeAlias, Tuple, Union, NamedTuple, Optional
import torch.nn.functional as F
import torch.distributions as dist
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, ConfigDict

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

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "input_dim": 128,
                "k_atoms": 50,
                "M": 512,
                "latent_dim": 32,
                "beta": 4.0
            }
        }
    )

from deepkernels.losses.simple import SimpleLoss
from deepkernels.models.parent import BaseGenerativeModel

class EncoderOutput(NamedTuple):
    alpha: torch.Tensor
    alpha_mu: torch.Tensor
    alpha_factor: torch.Tensor
    alpha_diag: torch.Tensor
    log_pi: torch.Tensor
    pi: torch.Tensor
    z: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    ls: torch.Tensor
    real_x: torch.Tensor

class ConvolutionalLoopEncoder(BaseGenerativeModel):
    """
    Role: Compress to latent space (latent_dim=16)
    input_features: The number of variables in your time series (e.g., 1 for univariate)
    bottleneck_dim: The exact dimension your KernelNetwork expects (64)
    """
    def __init__(self,
                 config: Optional[VAEConfig]=None,
                 input_dim=30,
                 latent_dim=16,
                 k_atoms=30,
                 bottleneck_dim=64,
                 dropout=0.05
        ):
        super().__init__()
        self.config = config or VAEConfig()
        
        self.input_dim = input_dim
        self.spectral_input_dim=2048
        self.latent_dim = latent_dim
        self.k_atoms = k_atoms
        self.M = 128
        self.jitter = 1e-6
        self.bottleneck_dim = bottleneck_dim or 64
        self.rank = 3
        

        # --- Fusion Layer ---
        # Concatenates GRU State (64) + Spectral State (64) + Prev Pi (30) -> 158
        self.fusion_net = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Linear(158, 64)),
            nn.LayerNorm(64),
            nn.Tanh()
        )

        #-wide base filter for rough trends-#
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        
        #- Convolutional layers capture sequentially tighter & higher frequencies)
        self.stage1 = ConvolutionalNetwork1D(32, 64, kernel_size=5, stride=2)
        self.stage2 = ConvolutionalNetwork1D(64, 128, kernel_size=3, stride=2)
        self.stage3 = ConvolutionalNetwork1D(128, 256, kernel_size=3, stride=2)
        
        #- Global average pooling -- makes bottleneck invariant to the exact sequence length
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_mu = nn.Linear(256, bottleneck_dim)
        self.fc_logvar = nn.Linear(256, bottleneck_dim)
        
        self.latent_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.latent_logvar = nn.Linear(bottleneck_dim, latent_dim)
    
    
    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        """
        Args:
            x: Input features [Batch, seq_len, 30] or recon_x
            pi: Dirichlet mixture weights [Batch, seq_len, 30]. 
            spectral_bottleneck: [Batch, seq_len, 64]
        """
        #-for first iter: Handle missing args and scale by 2.1x-#
        batch_size = x.size(0)
        device = x.device
        empty_tensor = torch.empty(0, device=device)
        
        pi = params.get('pi', None)
        bottleneck = params.get('bottleneck', None)
        alpha = params.get("alpha", None)
        alpha_mu = params.get('alpha_mu', None)
        alpha_factor = params.get("alpha_factor", None)
        alpha_diag = params.get("alpha_diag", None)
        ls = params.get('ls', None)
        jitter=1e-6

        if vae_out is not None:
            alpha = vae_out.get('alpha') if isinstance(vae_out, dict) else getattr(vae_out, 'alpha', None)
            
            if alpha is not None:
                alpha_mu = vae_out.get('alpha_mu') if isinstance(vae_out, dict) else getattr(vae_out, 'alpha_mu', None)
                alpha_factor = vae_out.get('alpha_factor') if isinstance(vae_out, dict) else getattr(vae_out, 'alpha_factor', None)
                alpha_diag = vae_out.get('alpha_diag') if isinstance(vae_out, dict) else getattr(vae_out, 'alpha_diag', None)
                
            if pi is None:
                pi = vae_out.get('pi') if isinstance(vae_out, dict) else getattr(vae_out, 'pi', None)
            if bottleneck is None:
                bottleneck = vae_out.get('bottleneck') if isinstance(vae_out, dict) else getattr(vae_out, 'bottleneck', None)
        if pi is None:
            pi = torch.full((batch_size, self.k_atoms), 1.0/self.k_atoms, device=device)
            pi = pi + (torch.randn_like(pi) * 0.01)
            pi = F.softmax(pi, dim=-1)
            
        if bottleneck is None:
            bottleneck = torch.zeros(batch_size, self.bottleneck_dim, device=device, dtype=x.dtype)
            
        if alpha is None:
            alpha_mu = torch.zeros(batch_size, self.k_atoms, device=device)
            alpha_factor = torch.randn(batch_size, self.k_atoms, self.rank, device=device) * 0.1
            alpha_diag = torch.ones(batch_size, self.k_atoms, device=device)
            alpha = self.lowrankmultivariatenorm(alpha_mu, alpha_factor, alpha_diag)
            
        # --- 4. GPYTORCH FIX: Sanitize remaining Nones into empty tensors ---
        empty_tensor = torch.empty(0, device=device)
        
        alpha_mu = empty_tensor if alpha_mu is None else alpha_mu
        alpha_factor = empty_tensor if alpha_factor is None else alpha_factor
        alpha_diag = empty_tensor if alpha_diag is None else alpha_diag
        ls = empty_tensor if ls is None else ls
        
        
        mu_data, logvar_data = self.run_convolutional_layers(x)

        conv_bottleneck = self.reparameterise(mu_data, logvar_data)

        log_pi = torch.log(pi + jitter)

        weights = torch.cat([conv_bottleneck, log_pi, bottleneck], dim=-1) #-dim 158

        post_bottleneck = self.fusion_net(weights)
        
        mu_z = self.latent_mu(post_bottleneck)
        logvar_z = self.latent_logvar(post_bottleneck)

        z = self.reparameterise(mu_z, logvar_z)
        
        encoder_out = EncoderOutput(
            alpha=alpha,
            alpha_mu=alpha_mu,
            alpha_factor=alpha_factor,
            alpha_diag=alpha_diag,
            log_pi=log_pi,
            pi=pi,
            z=z,
            mu_z=mu_z,
            logvar_z=logvar_z,
            ls=ls,
            real_x=x
        )

        return encoder_out
    
    def run_convolutional_layers(self, x):
        """
        x: Expected shape [Batch, Seq_Len, Features] (Standard PyTorch sequence format)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # [Batch, 1, Features]
        
        x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.global_pool(x).squeeze(-1)
        
        mu = self.fc_mu(x)
        
        logvar = torch.clamp(self.fc_logvar(x), min=-10.0, max=4.0)
        
        return mu, logvar
    

class ConvolutionalNetwork1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # Ensure padding keeps sequence length consistent if stride=1
        padding = kernel_size // 2 
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.SiLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.act2(out)
        return out