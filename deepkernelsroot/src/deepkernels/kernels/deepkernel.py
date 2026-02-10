import torch
import math
from gpytorch.kernels import Kernel

class DeepKernel(Kernel):
    def __init__(self, num_experts=30, **kwargs):
        super().__init__(**kwargs)
        self.K = num_experts
        # Tell GPyTorch this kernel produces a batch of K covariance matrices
        self.batch_shape = torch.Size([num_experts])

    def forward(self, x1, x2, diag=False, **params):
        # 1. Extract Params (Strictly from **kwargs, no data slicing)
        spectral_means = params.get('spectral_means') #-[Batch, K, M, D]
        bw = params.get('bw')                           #-[Batch, K, M, D]

        if spectral_means is None or bw is None:
            raise ValueError("DeepKernel requires 'spectral_means', and 'bw'")

        if diag:
            diff = (x1 - x2).unsqueeze(1).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
        else:
            diff = x1.unsqueeze(1).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2) - \
                   x2.unsqueeze(1).unsqueeze(-3).unsqueeze(-2).unsqueeze(-2)

        # 3. Handle Spectral Params
        # Target: [Batch, K, 1, 1, M, D]
        if spectral_means.dim() == 3: # [K, M, D] (Static)
            mu_q = spectral_means.view(1, self.K, 1, 1, -1, spectral_means.size(-1))
            bw_q = bw.view(1, self.K, 1, 1, -1, bw.size(-1))
        else: # [Batch, K, M, D] (Dynamic)
            mu_q = spectral_means.unsqueeze(2).unsqueeze(2)
            bw_q = bw.unsqueeze(2).unsqueeze(2)

        # ------------------------------------------------------------
        # 4. Spectral Computation (Inner Mixture over M)
        # ------------------------------------------------------------
        # (diff * bw) -> [B, K, N, J, M, D]
        
        # Squared Distance (Exp component)
        sq_dist = ((diff * bw_q) * 2 * math.pi).pow(2).sum(dim=-1) 
        k_exp = torch.exp(-0.5 * sq_dist)
        
        # Cosine component
        cos_arg = 2 * math.pi * (diff * mu_q).sum(dim=-1)
        k_cos = torch.cos(cos_arg)
        
        # Product & Mean over M (Spectral integration)
        k_components = k_exp * k_cos
        k_experts = k_components.mean(dim=-1) # Result: [Batch, K, N, J]


        if diag:
            return k_experts.squeeze(-1) # [Batch, K, N]
            
        return k_experts # [Batch, K, N, J]