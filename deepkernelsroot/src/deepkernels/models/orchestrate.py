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


class GenerativeModel(gpytorch.Module):
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
        
        batch_x, batch_y = next(iter(train_loader))

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, 
            self.model, 
            num_data=self.num_data
        )

        self.encoder = encoder or RecurrentEncoder()
        self.dirichlet = dirichlet or AmortisedDirichlet()
        self.vae = vae or SpectralVAE()
        self.decoder = decoder or SpectralDecoder()
        self.kernel = kernel or DeepKernel()


        super().__init__()
        
        inducing_points = self.init_inducing_with_fft(train_y, num_inducing, feature_dim)

        self.gp = gp or GenerativeKernelProcess(inducing_points=inducing_points)

        # 3. INIT THE LIKELIHOOD
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_latents
        )

        # 4. MOVE TO GPU (Orchestrator handles hardware)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()


    @staticmethod
    def init_inducing_with_fft(
        y_target, 
        n_inducing, 
        feature_dim
    ):
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