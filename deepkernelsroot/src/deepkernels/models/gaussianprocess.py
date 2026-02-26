import torch
import gpytorch
from gpytorch.kernels.keops import RBFKernel
import torch.nn as nn
from gpytorch.models import ApproximateGP
import torch.nn.functional as F
from typing import Literal
import random

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean

class AcceleratedKernelGP(ApproximateGP):
    def __init__(self, inducing=None, k_atoms=30, num_latents=8, **params):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """
        self.k_atoms = k_atoms
        self.num_latents = num_latents
        
        #-inducing point
        base_inducing_points = params.get("inducing_points", torch.randn(512, 16))
        base_inducing_points = inducing if inducing is not None else base_inducing_points
        num_inducing = base_inducing_points.size(0)
        
        latent_batch_shape = torch.Size([num_latents])
        inducing_points = base_inducing_points.repeat(num_latents, 1, 1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing, batch_shape=latent_batch_shape)

        inner_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=False)
        
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(inner_strategy, num_tasks=self.k_atoms, num_latents=self.num_latents, latent_dim=-1)
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=latent_batch_shape)
        
        self.covar_module = GenerativeKernel(batch_shape=latent_batch_shape)

    def forward(self, x, **kwargs):
        """
        kwargs will contain 'gp_params' (from KernelNetwork) 
        and optionally 'pi' (from Dirichlet)
        """
        
        mean_x = self.mean_module(x)
        
        covar_x = self.covar_module(x, x, **kwargs)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)