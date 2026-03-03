import torch
import gpytorch
import torch.nn as nn

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean


class Simple(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, k_atoms=30, num_latents=8):
        """
        'Physics Anchor' with KeOps Optimised Kernel (Exact Version)
        No inducing points required.
        
        Args:
            train_x: Dummy index or raw features (Shape: [N, ...])
            train_y: Target values (Shape: [num_latents, N] or [N])
            likelihood: gpytorch.likelihoods.GaussianLikelihood (or Multitask)
            num_latents: The number of distinct NKN graphs to broadcast
        """
        self.k_atoms = k_atoms
        self.num_latents = num_latents
        self.latent_batch_shape = torch.Size([self.num_latents])
        self.likelihood = likelihood if likelihood else gpytorch.likelihoods.GaussianLikelihood(batch_shape=self.latent_batch_shape)

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self.latent_batch_shape)
        self.covar_module = GenerativeKernel(batch_shape=self.latent_batch_shape)

    def forward(self, x, **kwargs):
        """
        kwargs MUST contain 'gp_params' from your KernelNetwork.
        During evaluation, it may also contain 'gp_params_eval'.
        """
        xc = x.contiguous()
        mean_x = self.mean_module(xc) #-outputs [num_latents, N]
        covar_x = self.covar_module(xc, xc, **kwargs) #-outputs KeOps LazyTensor shape: [num_latents, N, N]
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

