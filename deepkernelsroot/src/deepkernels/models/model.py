from deepkernels.models.dirichlet import AmortisedDirichlet
from deepkernels.models.spectral_VAE import SpectralVAE
from deepkernels.models.encoder import RecurrentEncoder
from deepkernels.kernels.deepkernel import DeepKernel, DynamicMixtureMean

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

from deepkernels.losses import kl


class GenerativeKernelProcess(ApproximateGP):
    """
    Data Flow: 
    1. real_x in
    2. Input (real_x) -> 'RecurrentEncoder' -> Latent (z)
    3. Latent (z) -> 'AmortisedDirichlet' Module for nonparametric clustering -> gradients flow back to 'Recurrent Encoder' via amortised inference
    3. Latent (z) -> VAE Decoder -> Reconstruction (x_hat) [Regularization 2]
    4. Latent (z) -> VAE_decoder -> Primitive Kernel Deconstruction inside gp kernel
    5. latent(z) -> linear model of coregionalisation gaussian process in function space  -> Prediction (y_hat)

    yhat_k_{k=1, 2, ... , 30} ~ sum_{{weight{q_k} * GP(mu_latent_{q_k}, sigma_latent_{q_k})}} for q in rank
    """
    def __init__(
            self,
            inducing_points,
            steps=3,
            num_latents=6, 
            n_inducing=1024, 
            k_atoms=30, 
            feature_dim=7680
        ):

        self.steps = steps
        self.k_atoms = k_atoms
        self.num_latents = num_latents

        #-Variational dist-#
        #-batch: torch.Size([num_tasks]) so we have distinct variational parameters (m, S) for the latent functions of each task
        
        dist = CholeskyVariationalDistribution(
            n_inducing, 
            batch_shape=torch.Size([self.num_latents])
        )
        
        inner_strategy = VariationalStrategy(self, inducing_points, dist, learn_inducing_locations=True)
        
        #-Linear Model of Coregionalisation variational strategy-#
        variational_strategy = LMCVariationalStrategy(inner_strategy, num_tasks=self.k_atoms, num_latents=self.num_latents, latent_dim=-1)

        super().__init__(variational_strategy)

        self.mean_module = DynamicMixtureMean()
        
        #-covar module (custom dynamic with deep feature extraction 
        # and kernel decomposition using similar methods to the 
        # automatic statistician)

        self.covar_module = DeepKernel(num_latents=num_latents)
        self.vae = SpectralVAE()
        self.dirichlet = self.vae.dirichlet
        self.encoder = self.vae.encoder
    
    
    def forward(self, x, **params):
        """beware that steps is encoded in init for the vae call"""
        self.vae.encoder(x)
        self.vae.dirichlet(x)
        loop_results = self.vae(x, steps=self.steps)
        phi = loop_results['spectral_features_per_step'][-1]
        pi = loop_results['simplex_sample_out']
        prediction = self.variational_strategy(phi, pi=pi)
        loop_results.append(prediction)
        return loop_results