import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import math
import torch.nn.utils.spectral_norm as sn
from typing import TypeAlias, Tuple, Union, NamedTuple, Optional
import torch.nn.functional as F
import torch.distributions as dist
import gpytorch

from deepkernels.models.parent import BaseGenerativeModel

from dataclasses import dataclass

@dataclass
class EncoderConfig:
    latent_dim: int = 16
    input_dim: int = 30
    bottleneck_dim: int = 64
    k_atoms: int = 30
    rank: int = 3
    jitter: float = 1e-6
    evidence_dim: int = 58

class EncoderOutput(NamedTuple):
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
    alpha: torch.Tensor

class ConvolutionalLoopEncoder(BaseGenerativeModel):
    """
    Role: Compress to latent space (latent_dim=16)
    input_features: The number of variables in your time series (e.g., 1 for univariate)
    bottleneck_dim: The exact dimension your KernelNetwork expects (64)
    """
    def __init__(self,
                 config:EncoderConfig,
                 **kwargs
        ):
        super().__init__()
        self.config = config if config is not None else EncoderConfig()
        self.jitter = self.config.jitter
        self.latent_dim = self.config.latent_dim
        self.bottleneck_dim = self.config.bottleneck_dim
        self.k_atoms = self.config.k_atoms
        self.rank = self.config.rank
        self.evidence_dim = self.config.evidence_dim
        
        #-global & dynamic:
        self.input_dim = kwargs.get("input_dim", 30)
        self.n_data = kwargs.get('n_data', 87636.0)

        # --- Fusion Layer ---
        # conv_bottleneck (16 or 64) + Spectral_bottleneck (64) + Prev Pi (30) -> 110 or 158 or 222?
        fusion_in_dim = self.bottleneck_dim + self.k_atoms + self.bottleneck_dim

        self.fusion_net = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Linear(fusion_in_dim, 64)),
            nn.LayerNorm(64),
            nn.ELU(inplace=True)
        )

        #-wide base filter for rough trends-#
        self.stem = nn.Sequential(
            nn.Conv1d(self.input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.ELU(inplace=True)
        )
        
        #- Convolutional layers capture sequentially tighter & higher frequencies)
        self.stage1 = ConvolutionalNetwork1D(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.stage2 = ConvolutionalNetwork1D(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.stage3 = ConvolutionalNetwork1D(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        
        #- Global average pooling -- makes bottleneck invariant to the exact sequence length
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc_mu = nn.Linear(256, self.bottleneck_dim)
        self.fc_logvar = nn.Linear(256, self.bottleneck_dim)
        
        self.latent_mu = nn.Linear(self.bottleneck_dim, self.latent_dim)
        self.latent_logvar = nn.Linear(self.bottleneck_dim, self.latent_dim)

        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)
    
    def forward(self, x, vae_out, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, generative_mode:bool=False, **params) -> EncoderOutput:
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        pi = params.get('pi', None)
        if pi is None and vae_out is not None:
            pi = getattr(vae_out, 'pi', None)
        if pi is None:
            pi = torch.full((batch_size, self.k_atoms), 1.0/self.k_atoms, device=device)
            pi = pi + (torch.randn_like(pi) * 0.01)
            pi = F.softmax(pi, dim=-1)
       
        bottleneck = params.get('bottleneck', None)
        if bottleneck is None and vae_out is not None:
            bottleneck = getattr(vae_out, 'bottleneck', None)
        if bottleneck is None:
            bottleneck = torch.zeros(batch_size, self.bottleneck_dim, device=device, dtype=x.dtype)
        
        evidence_dim = self.evidence_dim or 58

        neutral_logit = -4.0
        alpha_mu = params.get('alpha_mu', None)
        if alpha_mu is None and vae_out is not None:
            alpha_mu = getattr(vae_out, 'alpha_mu', None)
            
        
        if alpha_mu is None:
            alpha_mu = torch.full((batch_size, evidence_dim), neutral_logit, device=device)
        
        alpha_diag = params.get('alpha_diag', None)
        if alpha_diag is None and vae_out is not None:
            alpha_diag = getattr(vae_out, 'alpha_diag', None)
        
        if alpha_diag is None:
            alpha_diag = torch.full((batch_size, evidence_dim), -0.5413, device=device)
        
        
        alpha_factor = params.get('alpha_factor', None)
        if alpha_factor is None and vae_out is not None:
            alpha_factor = getattr(vae_out, 'alpha_factor', None)
        
        if alpha_factor is None:
            alpha_factor = torch.zeros(batch_size, evidence_dim, self.rank, device=device)
        

        alpha = params.get('alpha', None)
        if alpha is None and vae_out is not None:
            alpha = getattr(vae_out, 'alpha', None)
        
        if alpha is None:
            alpha = torch.zeros(batch_size, evidence_dim, device=device)
        
        ls = params.get('ls', None)

        if ls is None and vae_out is not None:
            ls = getattr(vae_out, 'ls', None)
        
        if ls is None:
            ls = empty_tensor
        
        
        mu_data, logvar_data = self.run_convolutional_layers(x) #-outputs 64-#

        conv_bottleneck = self.reparameterise(mu_data, logvar_data, eps_min=-3.3, eps_max=3.3)

        log_pi = torch.log(pi + self.jitter) #-30-#

        log_pi = log_pi.to(conv_bottleneck.device)
        bottleneck = bottleneck.to(conv_bottleneck.device) #-64

        weights = torch.cat([conv_bottleneck, log_pi, bottleneck], dim=-1) #-dim 158

        post_bottleneck = self.fusion_net(weights)
        
        mu_z = self.latent_mu(post_bottleneck)
        logvar_z = self.latent_logvar(post_bottleneck)
        logvar_z = torch.clamp(logvar_z, min=-10.0, max=3.0)
        z = self.reparameterise(mu_z, logvar_z, eps_min=-3.3, eps_max=3.3)
        
        return EncoderOutput(
            alpha_mu=alpha_mu,
            alpha_factor=alpha_factor,
            alpha_diag=alpha_diag,
            log_pi=log_pi,
            pi=pi,
            z=z,
            mu_z=mu_z,
            logvar_z=logvar_z,
            ls=ls,
            real_x=x,
            alpha=alpha
        )
    
    def run_convolutional_layers(self, x):
        """
        x: Expected shape [Batch, Seq_Len, Features] (Standard PyTorch sequence format)
        """
       
        if x.dim() == 2:
            if x.shape[1] == self.input_dim:
                x = x.unsqueeze(1)
            else:
                x = x.unsqueeze(-1)
        
        x = x.transpose(1, 2)
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.global_pool(x).squeeze(-1)
        
        mu = self.fc_mu(x)
        
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=3.0)
        return mu, logvar
    

class ConvolutionalNetwork1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2 
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.ELU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.ELU(inplace=True)

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
        
        #if residual.shape[-1] != out.shape[-1]:
         #   diff = out.shape[-1] - residual.shape[-1]
          #  if diff > 0:
           #     residual = F.pad(residual, (0, diff))
           # else:
            #    out = F.pad(out, (0, -diff))
                
        out += residual
        out = self.act2(out)
        
        return out