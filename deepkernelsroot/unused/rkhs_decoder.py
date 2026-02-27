import torch
import itertools
from pykeops.torch import LazyTensor
from gpytorch.kernels import Kernel
#lmc adapted keops kernel blueprint-#
class LMCGenerativeKernel(Kernel):
    def __init__(self, num_tasks, rank=2, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.num_tasks = num_tasks
        self.rank = rank
        self.num_all_kernels = 16
        
        # Coregionalization matrices: B = W*W^T + diag(v)
        # We store them as [16, Tasks, Rank]
        self.register_parameter(
            name="W", 
            parameter=torch.nn.Parameter(torch.randn(16, num_tasks, rank))
        )
        self.register_parameter(
            name="v", 
            parameter=torch.nn.Parameter(torch.zeros(16, num_tasks))
        )
        
        # Pruning Buffers
        self.register_buffer("is_diagonal", torch.zeros(16, dtype=torch.bool))
        self.register_buffer("active_mask", torch.ones(16))

    def prune_aggressively(self, correlation_threshold=0.05, gate_threshold=0.01):
        """
        Directs low-contribution kernels to zero and 
        low-correlation kernels to diagonal-only math.
        """
        with torch.no_grad():
            for i in range(self.num_all_kernels):
                # Calculate the B matrix for this kernel: B = W W^T + diag(exp(v))
                W_i = self.W[i] # [Tasks, Rank]
                B_i = W_i @ W_i.T + torch.diag(torch.exp(self.v[i]))
                
                # 1. Check for total inactivity (Gate Pruning)
                # If the max value in B is tiny, deactivate the kernel entirely
                if B_i.abs().max() < gate_threshold:
                    self.active_mask[i] = 0.0
                    continue
                
                # 2. Check for Diagonalization (Correlation Pruning)
                # Compute absolute correlation matrix
                d = torch.diag(B_i).pow(-0.5)
                corr = (d.unsqueeze(1) * B_i * d.unsqueeze(0)).abs()
                
                # If off-diagonals are below threshold, force to diagonal-only math
                off_diag = corr - torch.eye(self.num_tasks, device=corr.device)
                if off_diag.max() < correlation_threshold:
                    self.is_diagonal[i] = True
                    # Clean up W to save potential future compute
                    self.W.data[i].zero_() 

        print(f"Pruning Complete: {self.is_diagonal.sum()} Diagonalized, {(1-self.active_mask).sum()} Deactivated.")

    def forward(self, x1, x2, diag=False, **params):
        # We assume x1 and x2 have task indices in the last column
        t1 = x1[..., -1].long()
        t2 = x2[..., -1].long()
        coords1 = x1[..., :-1]
        coords2 = x2[..., :-1]

        def covar_func(x1_raw, x2_raw, **inner_params):
            # Distance logic (Cached for all components)
            d2 = ((LazyTensor(x1_raw.unsqueeze(-2)) - LazyTensor(x2_raw.unsqueeze(-3)))**2).sum(-1)
            
            # This is where we assemble the 16 components
            # ... (Kernel definitions: k_rbf, k_sm, etc. - see previous code) ...
            kernels = [k_rbf, k_sm, k_per, k_mat, ...] 

            out = 0
            for i in range(16):
                if self.active_mask[i] == 0:
                    continue
                
                # Task Indexing
                # If diagonal, we only care if t1 == t2
                if self.is_diagonal[i]:
                    # Delta function: 1 if tasks are same, else 0
                    B_ij = LazyTensor((t1.unsqueeze(-1) == t2.unsqueeze(-2)).float())
                    # Multiply by the variance (diagonal of B)
                    var_i = torch.exp(self.v[i])[t1]
                    B_ij = B_ij * LazyTensor(var_i.unsqueeze(-1).unsqueeze(-1))
                else:
                    # Full Low-Rank LMC Math
                    W_i = self.W[i]
                    W_t1 = LazyTensor(W_i[t1].unsqueeze(-2)) # [N, 1, Rank]
                    W_t2 = LazyTensor(W_i[t2].unsqueeze(-3)) # [1, N, Rank]
                    B_ij = (W_t1 * W_t2).sum(-1)
                
                out += B_ij * kernels[i]
            
            return out

        return KeOpsLinearOperator(coords1, coords2, covar_func, ...)


import torch
import torch.nn as nn

from src.models.model_config import RootConfig

class ReproducingKernelHilbertSpaceDecoder(nn.Module):
    def __init__(self, config: RootConfig):
        super().__init__()
        self.config = config

        #-dimensions-#
        self.input_dim = self.config.model.gp.bottleneck_dim #-from gp-#
        self.dim_out = self.config.model.rkhs.transformer_out_dim #-matches ViT output-#
        self.n_anchor_points = self.config.model.rkhs.n_anchors
        self.register_buffer('eps', torch.tensor(getattr(config.model, 'eps', 1e-7)))

        #---Learnable Anchor Points on Latent Manifold Z---#
        self.anchors = nn.Parameter(torch.randn(self.n_anchor_points, self.input_dim))

        #---Kernel Hyperparameters (Gaussian RBF Kernel)---#
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))

        #--Projection weights--#
        self.weights_out = nn.utils.spectral_norm(nn.Linear(self.n_anchor_points, self.dim_out, bias=False))
    
    def compute_kernel(self, z):
        """Computes RBF Kernel between two sets of points. z: [batch, n_anchor_points]"""
        lengthscale = torch.exp(self.log_lengthscale) + self.eps
        dist_sq  = torch.cdist(z, self.anchors, p=2)**2 #--power inside bracket calcs euclidian distance
        kernel_scores = torch.exp(-0.5 * dist_sq / (lengthscale**2))
        return kernel_scores
    
    def forward(self, z, scores_out=False):
        """
        Reconstructs original vision transformer feature vectors from latent representations
        h_hat = sum_{i=1}^{n_anchor_points} alpha_i * k(z, anchor_point_i)

        z: [batch, bottleneck_dim] ; anchors: [n_anchor_points, bottleneck_dim] ; dist: [batch, n_anchor_points]
        """

        #---Compute manifold coordinates (kernel score)--
        Kz = self.compute_kernel(z)  #---[B, 256]---#

        #---Nyström Interpolation---#
        h_recon = self.weights_out(Kz) #--[B, 768]

        if scores_out:
            return h_recon, Kz

        return h_recon