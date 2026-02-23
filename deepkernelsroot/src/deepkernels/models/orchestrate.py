from deepkernels.models.dirichlet import AmortisedDirichlet
from deepkernels.models.spectral_VAE import SpectralVAE
from deepkernels.models.encoder import RecurrentEncoder
from deepkernels.models.model import GenerativeKernelProcess
from deepkernels.kernels.deepkernel import DeepKernel, DynamicMixtureMean
from deepkernels.models.decoder import SpectralDecoder

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
from gpytorch.mlls import AddedLossTerm
from torch.distributions import kl_divergence

from gpytorch.priors import HorseshoePrior, HalfCauchyPrior
from deepkernels.models.parent import BaseGenerativeModel


class GenerativeModel(BaseGenerativeModel):
    def __init__(self,
                 train_loader,
                 train_x,
                 train_y,
                 encoder: Optional[nn.Module]=None, 
                 dirichlet: Optional[gpytorch.Module]=None, 
                 vae: Optional[nn.Module]=None, 
                 decoder: Optional[nn.Module]=None, 
                 gp: Optional[gpytorch.models.ApproximateGP]=None, 
                 kernel: Optional[gpytorch.kernels.Kernel]=None, 
                 mean:Optional[gpytorch.means.Mean]=None, 
                 k_atoms:int=30, 
                 num_latents:int=6, 
                 latent_dim:int=16, 
                 input_dim:int=30,
                 M_fourier_features:int=128,
                 real_batch_size:int=128,
                 num_inducing: int=1024,
                 feature_dim: int=2048
    ):
        super().__init__()
        
        batch_x, batch_y = next(iter(train_loader))

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, 
            self.model, 
            num_data=self.num_data
        )

        self.encoder = RecurrentEncoder()
        self.dirichlet = AmortisedDirichlet()
        self.vae = SpectralVAE()
        self.decoder = SpectralDecoder()
        self.kernel = DeepKernel()

        super().__init__()
        
        inducing_points = self.init_inducing_with_fft(train_y, num_inducing, feature_dim)

        self.model = GenerativeKernelProcess(inducing_points=inducing_points)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_latents
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
        