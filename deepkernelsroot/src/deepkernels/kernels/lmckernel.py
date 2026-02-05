import torch
import math
from gpytorch.kernels import Kernel

class DeepStatelessEigenKernel(Kernel):
    def __init__(self, num_experts=30, **kwargs):
        """
        A Stateless kernel that inputs dirichlet outputs when 'output_static_bias' is set to False in dirichlet module.
        - Receives output from dirichlet module: (pi, beta, means, scale)
        """
        super().__init__(**kwargs)
        self.K = num_experts
        
    def forward(self, x1, x2, beta, spectral_means, bw, diag=False, **params):
        """
        TensorFlow Diagram:
        graph TD
        subgraph Input
        X1[x1: N, D]
        X2[x2: J, D]
        end

        subgraph Parameters
        Mu[Means: K, M, D]
        Sigma[BW: K, M, D]
        Pi[Pi: N, K]
        Beta[Beta: K]
        end

        Input --> Diff[Distance Calculation<br/>(N, J, 1, 1, D)]
        Mu --> Spectral[Spectral Compute]
        Sigma --> Spectral
        Diff --> Spectral

        subgraph Spectral_Block [Inner Mixture: M]
        Spectral --> Cos[Cosine Term<br/>(N, J, K, M)]
        Spectral --> Exp[Decay Term<br/>(N, J, K, M)]
        Cos --> Prod[Product]
        Exp --> Prod
        Prod --> MeanM[Mean over M<br/>Result: N, J, K]
        end

        subgraph LMC_Block [Outer Mixture: K]
        MeanM --> Gating
        Pi --> Gating
        Beta --> Gating[Apply Weights:<br/>Beta * Pi * Pi]
        Gating --> SumK[Sum over K<br/>Result: N, J]
        end

        SumK --> Final[Covariance Matrix<br/>N, J]

        Args: 
            x1, x2: (Batch, N, D) - Data features
            spectral_means:  (K, M, D)     - Spectral Frequencies (Oscillation)
            bw: (K, M, D)     - Bandwidths (Inverse Lengthscales)
            pi_x1:  (Batch, N, K) - Local Mixing Weights
            pi_x2:  (Batch, J, K) - Local Mixing Weights
            beta:   (K,)          - Global Prevalence
        
        target shape: [N, J, K, M, D]
        """
    
        x1_data = x1[..., :-(self.K)] #-first D columns-#
        x1_weights = x1[..., -(self.K):]
        if x1 is x2:
            x2_data = x1_data
            x2_weights = x1_weights
        else:
            x2_data = x2[..., :-(self.K)]
            x2_weights = x2[..., -(self.K):]
        
        if diag:
            diff = (x1_data - x2_data).view(*x1_data.shape[:-1], 1, 1, 1, x1_data.size(-1))
        else:
            diff = x1_data.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2) - x2_data.unsqueeze(-3).unsqueeze(-2).unsqueeze(-2)
        
        mu_q = spectral_means.view(1, 1, *spectral_means.shape) #-[1, 1, K, M , D]-#
        bw_q = bw.view(1, 1, *bw.shape) #-[1, 1, K, M , D]-#

        # ------------------------------------------------------------
        # Spectral Mixture (inner mixture over M modes)
        # ------------------------------------------------------------
        sq_dist = (diff * bw_q * 2 * math.pi).pow(2)
        k_exp = torch.exp(-0.5 * sq_dist.sum(dim=-1))
        cos_arg = 2 * math.pi * (diff * mu_q).sum(dim=-1)
        k_cos = torch.cos(cos_arg) #[N, J, K, M]
        k_components = k_exp * k_cos #-aggregate over M-#
        k_experts = k_components.mean(dim=-1) #-[N, J, K] -- mean for variance stability

        # ------------------------------------------------------------
        # 3. LMC Mixture (Outer Mixture over K)
        # ------------------------------------------------------------
        w_global = beta.view(1, 1, -1)
        if diag:
            # (N, K) -> (N, 1, K) for k experts-#
            w_local = (x1_weights * x2_weights).unsqueeze(-2) 
        else:
            #--Outer Product: (N, 1, K) * (1, J, K) -> (N, J, K)--#
            w_local = x1_weights.unsqueeze(-2) * x2_weights.unsqueeze(-3)
        
        k_weighted = k_experts * w_global * w_local
        res = k_weighted.sum(dim=-1) #-[N, J]

        if diag:
            return res.squeeze(-1)
        return res