import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
from deepkernels.models.parent import BaseGenerativeModel
import gpytorch
from typing import Optional, Any, NamedTuple

class GPParams(NamedTuple):
    gates: torch.Tensor
    periodic: torch.Tensor
    linear: torch.Tensor
    matern: torch.Tensor
    rational: torch.Tensor
    polynomial:torch.Tensor
    

class KernelNetworkOutput(NamedTuple):
    dirichlet_features: torch.Tensor
    gp_params: GPParams


class SafeSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) + 1e-4


class KernelNetwork(BaseGenerativeModel):
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.bottleneck_dim = self.kwargs.get("bottleneck_dim", 64)
        self.spectral_emb_dim = self.kwargs.get("spectral_emb_dim", 2048)
        self.individual_kernel_dim_out = self.kwargs.get("individual_kernel_dim_out", 32)
        self.num_primitives = self.kwargs.get("num_primitives", 5)
        
        
        self.primitives_total_dim = self.num_primitives * self.individual_kernel_dim_out # 8 * 32 = 160
        
        self.linear = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.periodic = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.matern = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.rational = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.polynomial = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out, bias=False))
        

        self.spectral_feedback_loop = nn.Sequential(
            torch.nn.utils.spectral_norm(
                nn.Linear(self.primitives_total_dim, self.spectral_emb_dim)), #-160 -> 2048
                SafeSoftplus()
        )

        self.gate_head = nn.Sequential(
            nn.Linear(self.primitives_total_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(64, 8)),
            SafeSoftplus()                  
        )
        

        self.init_weights_nkn()
    
    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only=False, **params):
        """
        inputs the latent bottleneck dim (64)
        """
        #-kernels-#
        matern = self.matern(x)
        linear = self.linear(x)
        periodic = self.periodic(x)
        rational = self.rational(x)
        polynomial = self.polynomial(x)

        kernel_features = torch.cat([linear, periodic, rational, polynomial, matern], dim=-1) #-[B,160]

        features_large = self.spectral_feedback_loop(kernel_features)
        
        gates = self.gate_head(kernel_features)

        gp_params = GPParams(
            gates=gates,
            linear=linear,
            periodic=periodic,
            rational=rational,
            polynomial=polynomial,
            matern=matern
        )

        return KernelNetworkOutput(
            dirichlet_features=features_large,
            gp_params=gp_params
        )
    
    def init_weights_nkn(self):
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.uniform_(self.linear.bias, 0.0, 2.0)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)
        nn.init.orthogonal_(self.matern.weight, gain=1.0)
        nn.init.uniform_(self.matern.bias, -0.1, 0.1)
        nn.init.normal_(self.polynomial.weight, mean=0.005, std=0.01)
        for module in self.gate_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)