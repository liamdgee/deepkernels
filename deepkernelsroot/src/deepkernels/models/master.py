import torch
import gpytorch
from gpytorch.means import ConstantMean
import math
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import ApproximateGP
import torch.nn.functional as F
import torch.nn as nn

from src.deepkernels.models.dirichlet import AmortisedDirichlet, HDPConfig
from src.deepkernels.kernels.lmckernel import DeepStatelessEigenKernel
from src.deepkernels.models.beta_vae import SpectralVAE, VAEConfig
from src.deepkernels.models.encoder import RecurrentEncoder
from src.deepkernels.models.linear_decoder import BayesDecoder
from src.deepkernels.models.NKN import NeuralKernelNetwork
from src.deepkernels.kernels.master import MasterKernel

#-orchestration GP class-#
class DeepGaussianProcess(gpytorch.models.ApproximateGP):
    def __init__(self, 
                 inducing_points, 
                 k_experts=30, 
                 encoder=None, 
                 vae_decoder=None, 
                 nkn=None, 
                 dirichlet=None, 
                 latent_dim=64, 
                 hidden_dim=128, 
                 input_dim=256, 
                 rff_dim=256
        ):
        """
        The Orchestrator
        - num_experts (int): K
        - encoder: x -> q(z|x)
        - decoder: z -> p(x|z) (Your BayesDecoder)
        - nkn_head: z -> GP Features (Structural Primitives)
        - dirichlet: z -> Mixture Weights & Spectral Params
        - inducing_points (Tensor): Initial inducing points [K, M, D]
        """
        distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_shape=torch.Size([k_experts])
        )

        strategy = VariationalStrategy(
            self, 
            inducing_points, 
            distribution, 
            learn_inducing_locations=True
        )
        
        super().__init__(strategy)
        
        #-dims:-#
        self.latent_dim = latent_dim or 64
        self.hidden_dim = hidden_dim or 128
        self.input_dim = input_dim or 256 #-original data inpuit-#
        self.rff_dim = rff_dim or 256
        self.k_experts = k_experts or 30

        self.encoder = encoder if encoder else RecurrentEncoder()
        self.dirichlet = dirichlet if dirichlet else AmortisedDirichlet()
        self.vae_decoder = vae_decoder if vae_decoder else BayesDecoder(self.latent_dim, self.input_dim)
        self.topic_decoder = nn.Linear(self.k_experts, self.input_dim, bias=False)
        self.nkn = nkn if nkn else NeuralKernelNetwork()
        

        #-mean module-#
        self.mean_module = gpytorch.means.LinearMean(input_size=self.latent_dim)
        
        #-Cov module (stateless)-#
        self.covar_module = gpytorch.kernels.ScaleKernel(
            MasterKernel(num_experts=self.k_experts, batch_mode=True), 
            batch_shape=torch.Size([self.k_experts])
        )

        self.auxiliary_loss = torch.tensor(0.0)

    def forward(self, x):
        #-latent projection-#
        z_dist = self.encoder(x)

        #-reparameterise-#
        z = z_dist.rsample()

        #-vae_loss-#
        vae_recon = self.vae_decoder(z)

        #-spectral params and nonparametric clustering-#
        pi, beta, means, bw, omega = self.dirichlet(z, rff_kernel=False)

        #-gp features from nkn-#
        kernel_features = self.nkn(z)

        #-recon_loss-#
        topic_recon = self.topic_decoder(pi)
        
        #-gp pass-#
        gp_features = kernel_features.expand(self.k_experts, *kernel_features.shape)
        
        #-gp mean-#
        mean_x = self.mean_module(gp_features)

        #-deep custom covariance kernel-#
        covar_x = self.covar_module(
            gp_features,
            spectral_means=means, 
            bw=bw
        )
        
        gp_regression_output = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return gp_regression_output, vae_recon, z_dist, topic_recon, pi