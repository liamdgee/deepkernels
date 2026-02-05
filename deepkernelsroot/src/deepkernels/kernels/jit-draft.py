import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import Kernel

class JITCompoundKernel(Kernel):
    """
    Smoothly Gated Compound Kernel (Mixture of Experts).
    
    Designed for HMC Differentiability:
    - Replaces hard kernel switching with soft, differentiable gating.
    - Preserves Positive Definiteness via Outer-Product mixing.
    - Uses JIT compilation to fuse the complex graph.
    
    Math:
    K(x,x') = [g(x)g(x')] * K_global(x,x') + [(1-g(x))(1-g(x'))] * K_local(x,x')
    where g(x) is the gating network output in (0, 1).
    """
    def __init__(self, 
                 global_kernel, 
                 local_kernel, 
                 input_dim, 
                 hidden_dim=32, 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.global_kernel = global_kernel  # e.g., SparseNTK
        self.local_kernel = local_kernel    # e.g., DynamicSpectralMixture
        
        # --- The Gating Network ---
        # Determines which kernel is "active" at a given input location.
        # Uses Tanh for HMC stability (smooth gradients).
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Outputs probability [0, 1]
        )
        
        # Register priors for the gate (Critical for HMC!)
        # Keeps the gate weights from drifting to infinity
        self.register_prior(
            "gate_weight_prior",
            gpytorch.priors.NormalPrior(0.0, 1.0),
            lambda m: [p for p in m.gate_net.parameters()],
            lambda m, v: None
        )

    # --- The JIT Decorator ---
    # In PyTorch 2.0+, this compiles the function into a fused CUDA kernel.
    # This significantly speeds up the HMC Leapfrog steps.
    # If using older PyTorch, remove this line or use @torch.jit.script
    @torch.compile(dynamic=True) 
    def _compute_gated_gram(self, x1, x2, k_global, k_local):
        """
        Fused kernel combination logic.
        Isolated in a helper method to maximize JIT optimization.
        """
        # 1. Compute Gating Values
        # g1: [N, 1], g2: [M, 1]
        g1 = self.gate_net(x1)
        g2 = self.gate_net(x2)
        
        # 2. Form Mixing Weights (Outer Products)
        # weight_global[i, j] = g(x_i) * g(x_j)
        # This structure preserves the PSD property of the resulting matrix.
        w_global = g1 @ g2.transpose(-1, -2) 
        
        # weight_local[i, j] = (1 - g(x_i)) * (1 - g(x_j))
        w_local = (1 - g1) @ (1 - g2.transpose(-1, -2))
        
        # 3. Combine
        return (w_global * k_global) + (w_local * k_local)

    def forward(self, x1, x2, diag=False, **params):
        # 1. Evaluate Sub-Kernels
        # (These might be heavy, so we compute them first)
        k_glob = self.global_kernel(x1, x2, diag=diag, **params)
        k_loc = self.local_kernel(x1, x2, diag=diag, **params)
        
        if diag:
            # Diagonal case: O(N) optimized path
            g1 = self.gate_net(x1).squeeze(-1) # [N]
            w_global = g1 ** 2
            w_local = (1 - g1) ** 2
            return (w_global * k_glob) + (w_local * k_loc)
            
        # 2. JIT-Compiled Combination
        # We pass the pre-computed K matrices to the compiled helper
        return self._compute_gated_gram(x1, x2, k_glob, k_loc)
