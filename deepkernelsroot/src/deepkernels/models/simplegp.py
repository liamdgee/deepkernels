import torch
import gpytorch
from gpytorch.kernels.keops import RBFKernel
import torch.nn
from gpytorch.models import ApproximateGP

from deepkernels.models.parent import BaseGenerativeModel

class KeOpsSimpleGP(ApproximateGP):
    def __init__(self, **params):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """

        inducing_points = params.get("inducing_points", torch.randn(512, 64))

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=False
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            RBFKernel(ard_num_dims=inducing_points.size(-1))
        )

        self.covar_module.base_kernel.lengthscale = 1.0
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)