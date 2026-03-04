import torch
import gpytorch
import torch.nn as nn
from gpytorch.models import ApproximateGP
import torch.nn.functional as F
from typing import Literal
import random
from typing import NamedTuple

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']

import pykeops
from pykeops.torch import LazyTensor
#### -- pykeops.clean_pykeops()
pykeops.config.cuda_standalone = True
pykeops.config.use_OpenMP = False

import math
import itertools
from gpytorch.kernels import Kernel

import linear_operator
from linear_operator.operators import LinearOperator, KeOpsLinearOperator
from gpytorch.kernels.keops import RBFKernel

class AcceleratedKernelGP(ApproximateGP):
    def __init__(self, inducing=None, likelihood=None, k_atoms=30, num_latents=8, **params):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """
        self.k_atoms = k_atoms
        self.num_latents = num_latents
        self.likelihood = likelihood if likelihood else gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.k_atoms)
        #-inducing points-#
        base_inducing_points = params.get("inducing_points", torch.randn(256, 16))
        base_inducing_points = inducing if inducing is not None else base_inducing_points
        num_inducing = base_inducing_points.size(0)
        
        latent_batch_shape = torch.Size([num_latents])
        inducing_points = base_inducing_points.repeat(num_latents, 1, 1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing, batch_shape=latent_batch_shape)

        inner_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(inner_strategy, num_tasks=self.k_atoms, num_latents=self.num_latents, latent_dim=-1)
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=latent_batch_shape)
        
        self.covar_module = GenerativeKernel(batch_shape=latent_batch_shape)

    def forward(self, x, **kwargs):
        """
        kwargs will contain 'gp_params' (from KernelNetwork) 
        and optionally 'pi' (from Dirichlet)
        """
        diag = kwargs.get("diag", False)

        xc = x.contiguous()
        
        mean_x = self.mean_module(xc)
        
        covar_x = self.covar_module(xc, xc, **kwargs)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)