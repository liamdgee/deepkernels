import math
import itertools
import linear_operator


import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean

import torch
import gpytorch
import torch.nn as nn

from typing import NamedTuple

class DynamicStrategy(gpytorch.variational.LMCVariationalStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_lmc = None
        
    @property
    def lmc_coefficients(self):
        """
        Overrides the GPyTorch internal property. 
        If the VAE gave us a matrix, use it. Otherwise, use the standard learned parameter.
        """
        if self._dynamic_lmc is not None:
            return self._dynamic_lmc.mean(dim=0).unsqueeze(0)
        return self.lmc_coefficients_parameter

class AcceleratedKernelGP(gpytorch.models.ApproximateGP):
    def __init__(self, likelihood, inducing=None, k_atoms=30, num_latents=8, kernel_features_dim=168):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """
        #-inducing points-#
        
        base_inducing_points = torch.randn(256, kernel_features_dim)
        base_inducing_points = inducing if inducing is not None else base_inducing_points
        num_inducing = base_inducing_points.size(0)
        
        latent_batch_shape = torch.Size([num_latents])

        inducing_points = base_inducing_points.repeat(num_latents, 1, 1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing, batch_shape=latent_batch_shape)

        inner_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(inner_strategy, num_tasks=1, num_latents=num_latents, latent_dim=-1)
        
        super().__init__(variational_strategy)

        self.likelihood = likelihood
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([8]))
        
        self.covar_module = GenerativeKernel(batch_shape=torch.Size([8]))
    
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
        
        mean_x = self.mean_module(x)
            
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)