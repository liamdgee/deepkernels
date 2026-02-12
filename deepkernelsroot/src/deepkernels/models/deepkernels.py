#filename: deepkernels.py
import torch
import gpytorch
from gpytorch.means import MultitaskMean, LinearMean
import math
from gpytorch.models import ApproximateGP
import torch.nn.functional as F
import torch.nn as nn
from gpytorch.kernels import LinearKernel, ScaleKernel, MultitaskKernel
from gpytorch.variational import (
    CholeskyVariationalDistribution, 
    VariationalStrategy, 
    LMCVariationalStrategy
)
import torch.nn as nn
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.models import ApproximateGP


from src.deepkernels.models.model import GenerativeKernelProcess
from src.deepkernels.kernels.deepkernel import DeepKernel

class DeepKernelProcess(gpytorch.models.ApproximateGP):
    def __init__(
            self, 
            orchestrator_module,
            config=None, 
            y_target: torch.Tensor=None, 
            n_tasks: int=30, 
            rank: int = 4, 
            n_inducing: int=1024,
            feature_dim: int=7680, 
            encoder_input_dim: int = 44, 
            fourier_dim=256
        ):
        """
        Args:
            num_inducing: Number of variational inducing points (SVGP).
            feature_dim: The flattened size of the spectral features (K * M * 2).
            num_tasks: Number of Dirichlet clusters (K).
            rank: Rank of the low-rank task covariance matrix (coregionalization).
            n_latents = latent dims in LMC -- often set proportional to K clusters

            Linear Model of Coregionalisation Structure:
            - Base: Deep Linear Kernel (Shared features across tasks)
            - Coregionalization: MultitaskKernel (Learns task correlation B)
        """
        self.n_tasks = n_tasks
        self.rank = rank
        self.M = fourier_dim
        self.encoder_input_dim = encoder_input_dim
        self.orchestrator = GenerativeKernelProcess()
        
        target = None
        if isinstance(y_target, torch.Tensor):
            target = y_target.mean(dim=-1) if y_target.dim() > 1 else y_target
        elif hasattr(config, 'y_target') and isinstance(config.y_target, torch.Tensor):
            target = self.config.y_target.mean(dim=-1) if y_target.dim() > 1 else y_target
        
        #-inducing points-#
        inducing = self.init_inducing_with_fft(y_target=target, n_inducing=n_inducing, feature_dim=feature_dim)

        #-Variational dist-#
        #-batch: torch.Size([num_tasks]) so we have distinct variational parameters (m, S) for the latent functions of each task
        dist = CholeskyVariationalDistribution(n_inducing, batch_shape=torch.Size([]))
        
        #-Linear Model of Coregionalisation variational strategy-#
        strategy = VariationalStrategy(self, inducing, dist, learn_inducing_locations=True)
        
        super().__init__(strategy)
   
        self.mean_module = MultitaskMean( torch.nn.ModuleList(
            [LinearMean(input_size = encoder_input_dim) for _ in range(n_tasks)],
            input_size=self.encoder_input_dim, num_tasks=self.n_tasks
        ))
    
        self.covar_module = MultitaskKernel(nn.ModuleList(
            DeepKernel(input_dim=feature_dim, n_experts=self.n_tasks,
                        features_per_expert=self.M)
        ))

        #-Expected shape of y_target: [N, num_tasks]-#
        if target is not None: 
            if target.dim() > 1 and target.shape[-1] == self.n_tasks:
                task_means = target.mean(dim=0) #-[num_tasks]-#
                self.mean_module.base_means[0].constant.data.copy_(task_means)
            else:
                self.mean_module.base_means[0].constant.data.fill_(target.mean().item())

    def forward(self, x, z):
        """
        Args:
            spectral_features: [Batch, K * M * 2] output from SpectralVAE
        """
        # The GP simply weighs the spectral features to predict y
        feats, covar = self.orchestrator(x)

        feats_batch = self.make_expert_tensor(feats)

        x_hat = self.mean_module(feats)

        y_hat = self.covar_module(covar)
        
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(x_hat, y_hat))
    
    @staticmethod
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
    
    def make_expert_tensor(flat_features, num_clusters=30):
        
        batch_size = flat_features.size(0)
        unflattened = flat_features.view(batch_size, num_clusters, -1)
        expert_tensor = unflattened.permute(1, 0, 2)
        
        return expert_tensor