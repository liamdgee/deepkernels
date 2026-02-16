#filename: master.py

import torch
import gpytorch
from gpytorch.kernels import Kernel, LinearKernel, ScaleKernel
from src.deepkernels.models.NKN import NeuralKernelNetwork
from src.deepkernels.models.dirichlet import AmortisedDirichlet

#-moving this logic to a nn module-#

class MasterKernel(Kernel): 
    def __init__(self, nkn, dirichlet, **kwargs):
        super().__init__(**kwargs)
        self.nkn = nkn
        self.dirichlet = dirichlet
        
        self.base_kernel = ScaleKernel(LinearKernel()) #-dot product in feature space-#

    def forward(self, x1, x2, diag=False, **params):
        
        feats_t = self.nkn(x1)          # [Batch, K, D_out]
        pi_t = self.dirichlet(x1)[0] #-[batch, k]
        
        if x1 is x2 or (x1.size() == x2.size() and torch.equal(x1, x2)):
            feats_t_plusone = feats_t
            pi_t_plusone = pi_t
        else:
            feats_t_plusone = self.nkn(x2)
            pi_t_plusone = self.dirichlet(x2)[0]

        #-- Elementwise Gating -> h = phi * sqrt(pi) --#
        sqrt_weights_t = torch.sqrt(pi_t).unsqueeze(-1)
        sqrt_weights_t_plusone = torch.sqrt(pi_t_plusone).unsqueeze(-1)
        
        kernel_t = feats_t * sqrt_weights_t
        kernel_t_plusone = feats_t_plusone * sqrt_weights_t_plusone

        #- Flatten and sum over k experts and d dimensions
        #  [Batch, K, D] -> [Batch, K*D] allows LinearKernel to perform all matrix operations quickly
        flat_kernel_t = kernel_t.view(x1.size(0), -1)
        flat_kernel_t_plusone = kernel_t_plusone.view(x2.size(0), -1)
        
        return self.base_kernel(flat_kernel_t, flat_kernel_t_plusone, diag=diag)