import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
from deepkernels.models.parent import BaseGenerativeModel
import gpytorch
from typing import Optional, Any, NamedTuple
from dataclasses import dataclass

class GPParams(NamedTuple):
    gates: torch.Tensor
    ls_rbf: torch.Tensor
    ls_per: torch.Tensor
    p_per: torch.Tensor
    ls_mat: torch.Tensor
    w_sm: torch.Tensor
    mu_sm: torch.Tensor
    v_sm: torch.Tensor
    

class KernelNetworkOutput(NamedTuple):
    dirichlet_features: torch.Tensor
    gp_params: GPParams


class KernelNetwork(BaseGenerativeModel):
    def __init__(self, 
                 bottleneck_dim=64, 
                 num_latents=8, 
                 spectral_emb_dim=2048,
                 gp_dim=1,
                 spectral_micro_mixtures=4):
        super().__init__()

        # --- Dimensions ---
        self.individual_kernel_dim_out = 32
        self.primitive_products = 12
        self.num_primitives = 4 # [linear, periodic, rbf, rational]
        
        self.primitives_total_dim = self.num_primitives * self.individual_kernel_dim_out # 4 * 32 = 128
        self.products_total_dim = self.primitive_products * self.individual_kernel_dim_out # 12 * 24 = 384
        
        self.compression_dim = 128
        
        self.head_input_dim = self.primitives_total_dim + self.compression_dim #-256
        
        self.linear = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))
        self.linear_scale = nn.Parameter(torch.tensor(0.1))
        self.periodic = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))
        self.rbf = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))
        self.matern = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))

        #--all individaul kernels were projected to dim 32 (aggregate: 128)--#

        self.selection_weights = nn.Parameter(torch.randn(self.primitive_products, self.num_primitives)) #-tilde - normal (12, 4)

        self.complex_interactions = nn.utils.spectral_norm(
            nn.Linear(self.products_total_dim, self.compression_dim) #-384 -> 128
        )

        self.spectral_feedback_loop = nn.Sequential(
            torch.nn.utils.spectral_norm(
                nn.Linear(self.head_input_dim, spectral_emb_dim)),
                nn.Sigmoid()
        )

        self.gate_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 16),
            nn.Softplus()                   
        )

        #- KeOps Parameter Generator Heads-#
        #- first 4 params are primitive kernel params, last 3 are specific spectral mixture params-#
        #- gp_dim determines if you use Isotropic (gp_dim=1) or ARD (gp_dim=Input_Dim) for spectral lengthscales
        self.param_heads = nn.ModuleDict({
            'ls_rbf': nn.Linear(self.head_input_dim, gp_dim),
            'ls_per': nn.Linear(self.head_input_dim, gp_dim),
            'p_per':  nn.Linear(self.head_input_dim, gp_dim),
            'ls_mat': nn.Linear(self.head_input_dim, gp_dim),
            'w_sm':   nn.Linear(self.head_input_dim, spectral_micro_mixtures),
            'mu_sm':  nn.Linear(self.head_input_dim, spectral_micro_mixtures),
            'v_sm':   nn.Linear(self.head_input_dim, spectral_micro_mixtures),
        })

        self.init_weights_nkn()
    
    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only=False, **params):
        """
        inputs the latent bottleneck dim (64)
        """
        lin, per, rbf, mat = self.compute_primitives(x)        

        custom_interactions = self.compute_kernel_interactions(lin, per, rbf, mat) #- outputs [B, 12, 128]

        kernel_features = torch.cat([lin, per, rbf, mat, custom_interactions], dim=-1) #-[B,256]

        features_large = self.feed_dirichlet_gate(kernel_features)
        
        gp_params = self.get_keops_gp_params(kernel_features)

        return KernelNetworkOutput(
            dirichlet_features=features_large,
            gp_params=gp_params
        )

    
    def init_weights_nkn(self):
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.uniform_(self.linear.bias, 0.0, 2.0)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)
        nn.init.orthogonal_(self.complex_interactions.weight, gain=1.41)

        for head in self.param_heads.values():
            if isinstance(head, nn.Linear):
                nn.init.orthogonal_(head.weight, gain=1.0)
                nn.init.zeros_(head.bias)
                
        for module in self.gate_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
    
    def compute_primitives(self, x):
        lin = self.linear(x) * self.linear_scale
        per = torch.cos(self.periodic(x))
        rbf = torch.exp(-torch.pow(self.rbf(x), 2))
        #-matern 5/2-#
        dist = torch.abs(self.matern(x))
        sqrt5 = math.sqrt(5.0)
        mat = (1.0 + sqrt5 * dist + (5.0 / 3.0) * torch.pow(dist, 2)) * torch.exp(-sqrt5 * dist)
        return lin, per, rbf, mat
    
    def get_keops_gp_params(self, kernel_features):
        gates = self.gate_head(kernel_features)
        gp_params = {'gates': gates}
        for name, head in self.param_heads.items():
            raw_val = head(kernel_features)
            val = F.softplus(raw_val) + 2e-4
            gp_params[name] = val
        return GPParams(**gp_params)
    
    def compute_kernel_interactions(self, lin, per, rbf, mat):
        stack = torch.stack([lin, per, rbf, mat], dim=-2) 
        mask = torch.sigmoid(self.selection_weights)
        
        stack_safe = torch.abs(stack) + 1e-6
        log_stack = torch.log(stack_safe)
        
        log_product = torch.einsum('...pd, kp -> ...kd', log_stack, mask)
        
        product_features = torch.exp(log_product) 
        interactions_matrix = product_features.flatten(start_dim=-2) 
        return self.complex_interactions(interactions_matrix)
    
    def feed_dirichlet_gate(self, kernel_features):
        return self.spectral_feedback_loop(kernel_features)

    