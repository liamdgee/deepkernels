import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import gpytorch
from typing import Optional, Any, NamedTuple


from deepkernels.models.parent import BaseGenerativeModel

class GPParams(NamedTuple):
    gates: torch.Tensor
    periodic: torch.Tensor
    linear: torch.Tensor
    matern: torch.Tensor
    rational: torch.Tensor
    polynomial:torch.Tensor
    

class SafeSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) + 1e-4


class KernelNetwork(BaseGenerativeModel):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.bottleneck_dim = self.config.bottleneck_dim
        self.spectral_emb_dim = self.config.spectral_emb_dim
        self.individual_kernel_dim_out = self.config.individual_kernel_dim_out
        self.num_primitives = self.config.num_primitives
        
        
        self.primitives_total_dim = self.num_primitives * self.individual_kernel_dim_out # 8 * 32 = 160
        
        self.linear = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.periodic = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.matern = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.rational = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out))
        self.polynomial = nn.utils.spectral_norm(nn.Linear(self.bottleneck_dim, self.individual_kernel_dim_out, bias=False))
        
        dims = [self.primitives_total_dim, 512, 1024, self.spectral_emb_dim]
        layers = []
        for in_f, out_f in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_f, out_f))
        
            if out_f != dims[-1]:
                layers.append(nn.LayerNorm(out_f))
                layers.append(nn.GELU())
            else:
                layers.append(nn.LayerNorm(out_f))
                layers.append(nn.GELU()) 

        self.spectral_feedback_loop = nn.Sequential(*layers)
        

        self.gate_head = nn.Sequential(
            nn.Linear(self.primitives_total_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(64, 8)),
            SafeSoftplus()                  
        )
        

        self.init_weights_nkn()
    
    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
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

        return gp_params, features_large

    def init_weights_nkn(self):
        nn.init.orthogonal_(self.linear.weight_orig, gain=1.0)
        nn.init.uniform_(self.linear.bias, 0.0, 2.0)
        
        nn.init.orthogonal_(self.periodic.weight_orig, gain=1.41)
        
        nn.init.orthogonal_(self.rational.weight_orig, gain=1.41)
        
        nn.init.orthogonal_(self.matern.weight_orig, gain=1.0)
        nn.init.uniform_(self.matern.bias, -0.1, 0.1)
        
        nn.init.normal_(self.polynomial.weight_orig, mean=0.005, std=0.01)
        
        for module in self.gate_head.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_orig'):
                    nn.init.orthogonal_(module.weight_orig, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=1.0)