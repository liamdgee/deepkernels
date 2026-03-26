import math
import itertools
import linear_operator
import pykeops
from linear_operator.operators import LinearOperator, KeOpsLinearOperator, AddedDiagLinearOperator, DiagLinearOperator, MatmulLinearOperator, SumLinearOperator
import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F


from deepkernels.kernels.keops import GenerativeKernel, ProbabilisticMixtureMean
from typing import NamedTuple, Optional
class LMCVariationalStrategy(gpytorch.variational.LMCVariationalStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'lmc_coefficients' in self._parameters:
            del self._parameters['lmc_coefficients']
        
        self.register_parameter(
            name="raw_fsa_scale", 
            parameter=torch.nn.Parameter(torch.tensor([-2.75]))
        )

        self.register_parameter(
            name="raw_noise_floor", 
            parameter=torch.nn.Parameter(torch.tensor([-4.0]))
        )
        self._dynamic_lmc: Optional[torch.Tensor] = None
        print(f"DEBUG: Initial LMC Shape: {self.lmc_coefficients.shape}")
    
    @property
    def lmc_coefficients(self) -> torch.Tensor:
        if self._dynamic_lmc is not None:
            # We return it raw. The __call__ will handle the [8, 1, 1] reshape.
            return self._dynamic_lmc
        
        device = self.raw_fsa_scale.device
        return torch.ones(8, dtype=torch.float64, device=device) / 8.0
    
    @lmc_coefficients.setter
    def lmc_coefficients(self, value: torch.Tensor):
        if value is not None:
            self._dynamic_lmc = value.to(dtype=torch.float64)
        else:
            self._dynamic_lmc = None
    
    def __call__(self, x, prior=False, **kwargs):
        latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        w = self.lmc_coefficients
        mixed_mean = (w.unsqueeze(-1) * latent_dist.mean).sum(dim=0)
        w_sq = (w ** 2).reshape(-1, 1, 1)
        mixed_covar = (w_sq * latent_dist.lazy_covariance_matrix).sum(dim=0)
        
        fsa_weight = torch.nn.functional.softplus(self.raw_fsa_scale)
        noise_floor = torch.nn.functional.softplus(self.raw_noise_floor) + 1e-4

        with torch.no_grad():
            full_kernel_diag = self.gaussianprocess.covar_module(x).diagonal(dim1=-2, dim2=-1)
            if full_kernel_diag.dim() > 1 and full_kernel_diag.size(0) == 8:
                w = self.lmc_coefficients
                full_kernel_diag = (w.unsqueeze(-1) * full_kernel_diag).sum(dim=0)
            approx_diag = mixed_covar.diagonal(dim1=-2, dim2=-1)
            residual = (full_kernel_diag - approx_diag).clamp(min=1e-4)
        
        diag = (residual * fsa_weight) + noise_floor
        
        final_covar = linear_operator.operators.AddedDiagLinearOperator(
            mixed_covar, 
            linear_operator.operators.DiagLinearOperator(diag)
        )
        return gpytorch.distributions.MultivariateNormal(mixed_mean, final_covar)
    
class AcceleratedKernelGP(gpytorch.models.ApproximateGP):
    def __init__(self, likelihood, num_inducing=512, k_atoms=30, num_latents=8, kernel_features_dim=198):
        """
        'Physics Anchor' with KeOps Optimised Kernel
        Args:
            inducing_points: (M, D) tensor of initial inducing point locations.
        """
        base_inducing_points = torch.randn(512, kernel_features_dim, dtype=torch.float64) * 0.0541377
        base_inducing_points += torch.randn_like(base_inducing_points) * 0.025137
        dirichlet_init = torch.full((512, 30), 1.0 / 30.0, dtype=torch.float64)
        dirichlet_init += torch.randn_like(dirichlet_init) * 1e-6
        base_inducing_points[:, 168:198] = dirichlet_init

        num_inducing = base_inducing_points.size(0)
        latent_batch_shape = torch.Size([num_latents])
        inducing_points = base_inducing_points.repeat(num_latents, 1, 1)
        inducing_points += torch.randn_like(inducing_points) * 1e-4

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing, batch_shape=latent_batch_shape)

        inner_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        variational_strategy = LMCVariationalStrategy(inner_strategy, num_tasks=1, num_latents=num_latents, latent_dim=-1)
        
        
        super().__init__(variational_strategy)
        object.__setattr__(self.variational_strategy, 'gaussianprocess', self)
        self.likelihood = likelihood
        self.mean_module = ProbabilisticMixtureMean(k_atoms=k_atoms, num_latents=num_latents)
        self.covar_module = GenerativeKernel(batch_shape=latent_batch_shape)
    
    def forward(self, x, **params):
        """
        x: The PACKED feature tensor [..., N, 168] coming from the decoder
        """
        xc = x.contiguous()
        mean_x = self.mean_module(xc)  
        covar_x = self.covar_module(xc, **params)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)