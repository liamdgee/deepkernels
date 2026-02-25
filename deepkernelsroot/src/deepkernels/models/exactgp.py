import torch
import gpytorch
import torch.nn as nn

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean


class Simple(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, keops_kernel, k_atoms=30):
        super().__init__(None, None, likelihood)
        
        self.mean_module = ProbabilisticMixtureMean(k_atoms=k_atoms)
        self.covar_module = GenerativeKernel()

    def forward(self, x, **kwargs):
        """
        kwargs will contain 'gp_params' (from KernelNetwork) 
        and optionally 'pi' (from Dirichlet)
        """
        
        mean_x = self.mean_module(x, **kwargs)
        
        covar_x = self.covar_module(x, x, **kwargs)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)