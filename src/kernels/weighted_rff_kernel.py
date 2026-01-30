import gpytorch
import torch
import torch.nn as nn
from gpytorch.kernels import Kernel
from src.models.dirichlet_variational import VariationalDirichlet
import math

class CrossCovarRFFKernel(Kernel):
    """
    Computes a matrix valued kernel that maps to a vector-valued reproducing kernel hilbert space
    output shape: [batch, N, N, D, D] or block diagonal: [batch, N*D, N*D]
    """
    def __init__(self, dp_module, **kwargs):
        super().__init__(**kwargs)
        self.dp = dp_module
        self.D = dp_module.D
    
    def _to_feature_space(self, x, weights, omega, bias):
        """
        computes features in the same D dimension
        Args:
            x: [B, N, D]
            omega: [K, M, D]
        Returns:
            features: [B, N, D, K_atoms (n global clusters active) * M_freq (fourier dim)]
        """
        #-standardise-#
        if x.dim() == 2:
            x = x.unsqueeze(1) #-[B, 1, D]-#
        B, N, D = x.shape
        K, M, _ = omega.shape

        #-project elementwise-#
        #-treat each random fourier projection to treat the input dims as independent signals from dp module-#
        #-x: [B, N, D] -> [B, N, D, K, M]-#
        x0 = x.unsqueeze(3).unsqueeze(4)
        #-omega: [K, M, D] -> [1, 1, D, K, M]
        omega0 = omega.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        #-proj: x * omega (elementwise mult) creates rff features for each latent dim-#
        proj = x0 * omega0

        #-biases / activations-#
        bias0 = bias.view(1, 1, 1, K, M)
        z_active = torch.cos(proj + bias0)

        #-weighting for rff projection-#
        monte_carlo_constant = math.sqrt(2.0 / M)

        if weights.dim == 1:
            wscl = weights.sqrt().view(1, 1, 1, K, 1)
        else:
            wscl = weights.sqrt().view(B, 1, 1, K, 1)
        
        weightedfeats = z_active * wscl * monte_carlo_constant

        return weightedfeats.view(B, N, D, -1) #-[B, N, D, feature_dim]-#
    
    def forward(self, x1, x2, diag=False, **params):
        dp_out = params.get('variational_dirichlet_output')
        if dp_out is None:
            raise ValueError("HDP Params required")
        pi, beta, omega, bias = dp_out
        phi_x1 = self._to_feature_space(x1, pi, omega, bias)
        #-outputs [B, N, D, F]-#
        
        if diag:
            return (phi_x1 * phi_x1).sum(-1) #-diagonal variance: [B, N, D-#
        
        if x1.size() == x2.size() and torch.equal(x1, x2):
            phi_x2 = phi_x1
        else:
            phi_x2 = self._to_feature_space(x2, beta, omega, bias)
        
        #--Cross Product / Outer Product Kernel Logic -#
        # phi_x1: [B, N1, D, F] * phi_x2: [B, N2, D, F] -> [B, N1, N2, D, D]
        #- einsum logic {b=batch, i=N1, j=N2, u=D1, v=D2, f=features} -#
        covar_matrix = torch.einsum('biuf, bjvf -> bijuv', phi_x1, phi_x2)
        

        return covar_matrix #-outputs [B, N, N, D, D] for vector-valued GP