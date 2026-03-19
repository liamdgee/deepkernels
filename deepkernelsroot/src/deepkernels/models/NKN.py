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
        
        
        self.primitives_total_dim = self.num_primitives * self.individual_kernel_dim_out # 5 * 32 = 160
        
        self.linear = self._build_primitive(self.bottleneck_dim, self.individual_kernel_dim_out)
        self.periodic = self._build_primitive(self.bottleneck_dim, self.individual_kernel_dim_out)
        self.matern = self._build_primitive(self.bottleneck_dim, self.individual_kernel_dim_out)
        self.rational = self._build_primitive(self.bottleneck_dim, self.individual_kernel_dim_out)
        self.polynomial = self._build_primitive(self.bottleneck_dim, self.individual_kernel_dim_out, is_poly=True)
        
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
        
        gate_last_linear = nn.Linear(64, 8)

        self.gate_head = nn.Sequential(
            nn.Linear(self.primitives_total_dim, 64),
            nn.LayerNorm(64),
            nn.Softsign(),
            P.weight_norm(gate_last_linear),
            SafeSoftplus()                  
        )
        

        self.init_weights_nkn()
    
    def _build_primitive(self, in_dim, out_dim, is_poly=False):
        """Helper to safely init weights BEFORE applying weight_norm"""
        layer = nn.Linear(in_dim, out_dim, bias=not is_poly)
        
        if is_poly:
            nn.init.normal_(layer.weight, mean=0.005, std=0.01)
        else:
            nn.init.orthogonal_(layer.weight, gain=0.8)
            nn.init.zeros_(layer.bias)
            
        return P.weight_norm(layer)

    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        """
        inputs the latent bottleneck dim (64)
        """
        #-kernels-#
        matern = F.softsign(self.matern(x))
        linear = F.softsign(self.linear(x))
        periodic = F.softsign(self.periodic(x))
        rational = F.softsign(self.rational(x))
        polynomial = F.softsign(self.polynomial(x))

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
        # --- Gate Head: Neutral weights, positive bias ---
        for module in self.gate_head.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_orig'):
                    nn.init.orthogonal_(module.weight_orig, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)