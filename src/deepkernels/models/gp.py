import logging
from typing import NamedTuple, Optional
import math
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class AcceleratedKernelGP(gpytorch.models.ApproximateGP):
    """Minimal Variational GP for the public repo."""
    def __init__(self, likelihood, num_inducing=32, input_dim=30 * 7):
        # Input dim is 30 * 7 because pack_features concats 7 feature tensors
        #-- this is pseudo logic for state space gaussian processes-#
        inducing_points = torch.randn(num_inducing, input_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)