
import pykeops
from pykeops.torch import LazyTensor
import math
import itertools
import linear_operator
from linear_operator.operators import LinearOperator, KeOpsLinearOperator
from gpytorch.kernels.keops import RBFKernel

import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"


from deepkernels.models.parent import BaseGenerativeModel
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
            return self._dynamic_lmc
        return self.lmc_coefficients_parameter

class AcceleratedKernelGP(gpytorch.models.ApproximateGP):
    def __init__(self, likelihood, inducing=None, k_atoms=30, num_latents=8, kernel_features_dim=198, **params):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """

        #-inducing points-#
        base_inducing_points = params.get("inducing_points", torch.randn(256, kernel_features_dim))
        base_inducing_points = inducing if inducing is not None else base_inducing_points
        num_inducing = base_inducing_points.size(0)
        
        latent_batch_shape = torch.Size([num_latents])

        inducing_points = base_inducing_points.repeat(num_latents, 1, 1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing, batch_shape=latent_batch_shape)

        inner_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        variational_strategy = DynamicStrategy(inner_strategy, num_tasks=k_atoms, num_latents=num_latents, latent_dim=-1)
        
        super().__init__(variational_strategy)

        self.likelihood = likelihood
        
        self.mean_module = ProbabilisticMixtureMean(batch_shape=latent_batch_shape)
        
        self.covar_module = GenerativeKernel(batch_shape=latent_batch_shape)

    def forward(self, x, lmc_learned=None, diag=None, **kwargs):
        """
        x: The PACKED feature tensor [..., N, 168] coming from the decoder.
        kwargs: Contains 'pi' for the Mixture Mean.
        """

        if lmc_learned is not None:
            if lmc_learned.dim() == 3:
                lmc_learned = lmc_learned.mean(dim=0)
            
            self.variational_strategy._dynamic_lmc = lmc_learned
        else:
            self.variational_strategy._dynamic_lmc = None

        xc = x.contiguous()
        
        mean_x = self.mean_module(xc, **kwargs)
        
        covar_x = self.covar_module(xc, xc, **kwargs)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)