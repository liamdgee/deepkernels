import math
import itertools
import linear_operator

import os

import pykeops
import linear_operator
from pykeops.torch import LazyTensor
from linear_operator.operators import LinearOperator, KeOpsLinearOperator


pykeops.config.cuda_standalone = True
pykeops.config.use_OpenMP = False


import torch
import torch.nn as nn
import math
import gpytorch
import torch.nn.functional as F


from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean

import torch
import gpytorch
import torch.nn as nn

from typing import NamedTuple

class LMCVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_lmc = None
    
    def __getattribute__(self, name):
        if name == 'lmc_coefficients':
            dynamic_lmc = super().__getattribute__('_dynamic_lmc')
            if dynamic_lmc is not None:
                return dynamic_lmc.mean(dim=0).unsqueeze(0)
        return super().__getattribute__(name)
        

class AcceleratedKernelGP(gpytorch.models.ApproximateGP):
    def __init__(self, likelihood, inducing=None, k_atoms=30, num_latents=8, kernel_features_dim=198):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """
        if inducing is not None:
            base_inducing_points = inducing
        else:
            base_inducing_points = torch.tanh(torch.randn(1024, kernel_features_dim))
            base_inducing_points += torch.randn_like(base_inducing_points) * 0.017
            base_inducing_points[:, 168:198] = 1.0 / 30.0
        
        num_inducing = base_inducing_points.size(0)
        
        latent_batch_shape = torch.Size([num_latents])

        inducing_points = base_inducing_points.repeat(num_latents, 1, 1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing, batch_shape=latent_batch_shape)

        inner_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        variational_strategy = LMCVariationalStrategy(inner_strategy, num_tasks=1, num_latents=num_latents, latent_dim=-1)
        
        super().__init__(variational_strategy)
        self.likelihood = likelihood
        
        self.mean_module = ProbabilisticMixtureMean(k_atoms=k_atoms, num_latents=num_latents)
        
        self.covar_module = GenerativeKernel(batch_shape=latent_batch_shape)

        with torch.no_grad():
            test_x = base_inducing_points.unsqueeze(0).expand(num_latents, -1, -1)
            K_zz = self.covar_module(test_x).evaluate() 
            
            print(f"DEBUG: K_zz Max: {K_zz.max().item():.4f}")
            print(f"DEBUG: K_zz Min: {K_zz.min().item():.4f}")
            print(f"DEBUG: K_zz NaNs: {torch.isnan(K_zz).sum().item()}")
            print(f"DEBUG: K_zz Diag Mean: {K_zz.diagonal(dim1=-2, dim2=-1).mean().item():.4f}")
            diag = K_zz.diagonal(dim1=-2, dim2=-1)
            print(f"DEBUG: K_zz Diag Mean: {diag.mean().item():.4f}")
            off_diag = K_zz - torch.diag_embed(diag)
            print(f"DEBUG: Max Off-Diag: {off_diag.abs().max().item():.4f}")
            unique_rows = torch.unique(base_inducing_points, dim=0).size(0)
            print(f"DEBUG: Unique Inducing Points: {unique_rows} / 1024")
            print(f"--------------------------------\n")
        
    
    def forward(self, x, lmc_learned=None, indices=None, **params):
        """
        x: The PACKED feature tensor [..., N, 168] coming from the decoder
        """
        if lmc_learned is not None:
            if lmc_learned.dim() == 3:
                lmc_learned = lmc_learned.mean(dim=0)
            self.variational_strategy._dynamic_lmc = lmc_learned
        else:
            self.variational_strategy._dynamic_lmc = None
        
        if x.dim() == 2:
            num_latents = self.mean_module.batch_shape[0]
            x = x.unsqueeze(0).expand(num_latents, -1, -1)
        
        params.pop('lmc_learned', None)
        
        mean_x = self.mean_module(x)  
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)