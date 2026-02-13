#filename: master.py

import torch
import gpytorch
import torch.nn as nn
from src.deepkernels.models.NKN import NeuralKernelNetwork
from src.deepkernels.models.dirichlet import AmortisedDirichlet

#-moving this logic to a nn module-#

class WeightedNeuralKernel(nn.Module): 
    def __init__(self, nkn, dirichlet):
        super().__init__()
        self.nkn = nkn
        self.dirichlet = dirichlet

    def forward(self, x):
        
        batch = x.size(0)
        feats_t = self.nkn(x)                # [Batch, K, D_out]
        pi, _, _, bw_base, omega = self.dirichlet(x)    #-[batch, k]
        
        #-- KEY: b=batch, k=atoms, d=dim, r=rff_features
        proj = torch.einsum('bd,kmd->bkr', x, omega)
        #-- Elementwise Gating -> h = phi * sqrt(pi) --#
        sqrt_weights_t = torch.sqrt(pi).unsqueeze(-1)
        
        kernel_t = feats_t * sqrt_weights_t

        #  [Batch, K, D] -> [Batch, K*D] allows LinearKernel to perform all matrix operations quickly
        flat_kernel_t = kernel_t.view(x_latent.size(0), -1)
        
        return flat_kernel_t