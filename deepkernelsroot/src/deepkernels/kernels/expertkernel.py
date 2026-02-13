import torch
import torch.nn as nn
import math
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.priors import NormalPrior


class DeepExpertKernel(Kernel):
    def __init__(self, num_experts=30, **kwargs):
        """
        Outputs a Batch of Covariance Matrices for K Experts.
        Shape: [Batch, K, N, N]
        """
        super().__init__(**kwargs)
        self.K = num_experts
        # We tell GPyTorch that this kernel creates an extra batch dimension (K)
        self.batch_shape = torch.Size([num_experts]) 

    def forward(self, x1, x2, diag=False, **params):
        # 1. Extract Spectral Context
        # spectral_means: [K, M, D] (Static) or [B, K, M, D] (Dynamic)
        # bw: [K, M, D]
        spectral_means = params.get('spectral_means') 
        bw = params.get('bw')

        # 2. Reshape Inputs for Broadcasting against K
        # We want the final covariance to be [Batch, K, N, N]
        # x1: [Batch, N, D] -> [Batch, 1, N, 1, D]
        if diag:
             diff = (x1 - x2).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
        else:
             diff = x1.unsqueeze(1).unsqueeze(-2).unsqueeze(-2) - \
                    x2.unsqueeze(1).unsqueeze(-3).unsqueeze(-2)

        # 3. Handle Spectral Params
        # Force [1, K, 1, M, D] (Static) or [B, K, 1, M, D] (Dynamic) alignment
        if spectral_means.dim() == 3: #[K, M, D]
            mu_q = spectral_means.view(1, self.K, 1, -1, spectral_means.size(-1))
            bw_q = bw.view(1, self.K, 1, -1, bw.size(-1))
        else: #[B, K, M, D]
            mu_q = spectral_means.unsqueeze(2)
            bw_q = bw.unsqueeze(2)

        # ------------------------------------------------------------
        # 4. Spectral Computation (Inner Mixture M)
        # ------------------------------------------------------------
        # (diff * bw) -> [B, K, N, N, M, D]
        
        # Squared Distance (Exp component)
        sq_dist = ((diff * bw_q) * 2 * math.pi).pow(2).sum(dim=-1) 
        k_exp = torch.exp(-0.5 * sq_dist)
        
        # Cosine component
        cos_arg = 2 * math.pi * (diff * mu_q).sum(dim=-1)
        k_cos = torch.cos(cos_arg)
        
        # Product & Mean over M
        k_components = k_exp * k_cos
        k_experts = k_components.mean(dim=-1) # [B, K, N, N]

        # ------------------------------------------------------------
        # 5. NO SUMMATION OVER K
        # ------------------------------------------------------------
        # We return the K covariance matrices directly.
        
        if diag:
            return k_experts.squeeze(-1)
            
        return k_experts