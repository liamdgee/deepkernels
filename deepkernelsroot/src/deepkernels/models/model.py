from src.deepkernels.models.dirichlet import AmortisedDirichlet, HDPConfig
from src.deepkernels.models.spectral_VAE import SpectralVAE
from src.deepkernels.models.encoder import RecurrentEncoder
from src.deepkernels.models.linear_decoder import BayesDecoder
from src.deepkernels.kernels.deepkernel import DeepKernel
from src.deepkernels.models.deepkernels import DeepKernelProcess
from typing import Tuple, Optional, TypeAlias, Tuple, Union
import torch
import gpytorch
import torch.nn as nn
from torch.distributions import Dirichlet
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import IndependentMultitaskVariationalStrategy, VariationalStrategy

from gpytorch.means import MultitaskMean, LinearMean, ConstantMean
import math
from gpytorch.models import ApproximateGP
import torch.nn.functional as F
from gpytorch.kernels import LinearKernel, ScaleKernel, MultitaskKernel
from gpytorch.variational import (
    CholeskyVariationalDistribution, 
    VariationalStrategy, 
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy
)
import torch.nn as nn
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.models import ApproximateGP


class GenerativeKernelProcess(ApproximateGP):
    """
    Data Flow:
    1. real_x -> *skip* -> gp output layer (all means are interprettable as mean module never enters latent space.)
    2. Input (real_x) -> 'RecurrentEncoder' -> Latent (z)
    3. Latent (z) -> 'AmortisedDirichlet' Module for nonparametric clustering -> gradients flow back to 'Recurrent Encoder' via amortised inference
    3. Latent (z) -> VAE Decoder -> Reconstruction (x_hat) [Regularization 2]
    4. Latent (z) -> VAE_decoder -> Primitive Kernel Deconstruction inside gp kernel
    5. latent(z) -> linear model of coregionalisation gaussian process in function space  -> Prediction (y_hat)

    yhat_k_{k=1, 2, ... , 30} ~ sum_{{weight{q_k} * GP(mu_real_{k}, sigma_latent_{q_k})}} for q in rank
    
    where mu_real_{i} is deterministic per cluster
    """
    def __init__(self, steps, y_target, n_inducing=1024, k_atoms=30, feature_dim=7680):
        super().__init__()

        self.vae = SpectralVAE()
        steps = steps or 3
        self.y_target = y_target
        self.n_tasks = k_atoms or 30
        self.k_atoms = k_atoms or 30
        self.feature_dim = feature_dim or 7680
        self.num_latents = 3
        
        init_inducing = self.init_inducing_with_fft(y_target=y_target, n_inducing=n_inducing, feature_dim=feature_dim)

        #-Variational dist-#
        #-batch: torch.Size([num_tasks]) so we have distinct variational parameters (m, S) for the latent functions of each task
        
        dist = CholeskyVariationalDistribution(
            n_inducing, 
            batch_shape=torch.Size([self.num_latents])
        )
        
        #-Linear Model of Coregionalisation variational strategy-#
        strategy = LMCVariationalStrategy(
            VariationalStrategy(
                init_inducing, 
                dist, 
                learn_inducing_locations=True,), 
                num_tasks=self.k_atoms, 
                num_latents=self.num_latents, 
                latent_dim=-1)

        super().__init__(strategy)

        self.mean_module = ConstantMean()

        self.covar_module = LinearKernel()

        if y_target.dim() > 1 and y_target.shape[-1] == self.n_tasks:
            task_means = y_target.mean(dim=0) #-[num_tasks]-#
            self.mean_module.base_means[0].constant.data.copy_(task_means)
        else:
            self.mean_module.base_means[0].constant.data.fill_(y_target.mean().item())
    
        
    
    def forward(self, x, steps):
        loop_results = self.vae(x, steps=steps)
        phi = loop_results['spectral_features_per_step'][-1]
        x_hat = self.mean_module(x)
        y_hat = self.covar_module(phi)

        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(x_hat, y_hat))
    
    def init_inducing_with_fft(y_target: torch.Tensor = None, n_inducing: int = 1024, feature_dim: int = 7680):
        """
        Initializes inducing point values based on the FFT of the target signal.
        
        Args:
            y_target: [N_data] tensor of training targets used for initialization (Assuming somewhat evenly spaced or interpolated)
            n_inducing: Number of inducing points (must match model)
            M: Number of RFF components (M from dirichlet or encoder -- these will match)
            feature_dim: K clusters (30) * M fourier features (128) * 2 = 7680
        """
        #-fast fourier transform of target variable y-#
        if y_target is None:
            return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        
        yflat = y_target.flatten().cpu()
        if yflat.abs().sum() < 1e-6:
             return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        
        fourier_vals = torch.fft.rfft(yflat)
        jitter = 1e-6
        eps = 1e-9
        
        #-Construct a Probability Distribution from FFT magnitudes to sample weights from-#
        density = torch.abs(fourier_vals)
        if density.shape[0] > 0:
            density[0] = 0 #-remove DC component-#
        
        cdf = density.sum()
        if cdf < eps:
             return torch.randn(n_inducing, feature_dim) / math.sqrt(feature_dim)
        
        p = density / cdf
        
        # 3. Sample- generate probabilistic indices in feature space-#
        indices = torch.multinomial(p, feature_dim, replacement=True)
        
        # Retrieve the actual magnitudes for those sampled indices
        samples = density[indices]
        
        #-random signs for uniformity-#
        binary_mask = torch.bernoulli(torch.full((feature_dim,), 0.5))
        plus_or_minus_one = 2 * binary_mask - 1
        
        sigma_y = y_target.std() + jitter
        sigma_samples = samples.std() + jitter

        sqrt_feature_dim_scale = sigma_y / (sigma_samples * math.sqrt(feature_dim))
        weights_flat = samples * plus_or_minus_one * sqrt_feature_dim_scale

        #-expand to inducing-#
        inducing = weights_flat.unsqueeze(0).repeat(n_inducing, 1)
        inducing_jitter = torch.randn_like(inducing) * (sqrt_feature_dim_scale * sigma_samples * 0.13)
        inducing = inducing + inducing_jitter
        
        return inducing