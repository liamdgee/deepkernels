import torch
import gpytorch
from gpytorch.means import ConstantMean
import math

from src.deepkernels.models.dirichlet import AmortisedDirichlet, HDPConfig
from src.deepkernels.kernels.lmckernel import DeepStatelessEigenKernel
from src.deepkernels.models.beta_vae import SpectralVAE, VAEConfig
from src.deepkernels.models.encoder import Encoder
from src.deepkernels.models.decoder import NystromDecoder
from src.deepkernels.kernels.deepkernel import DeepKernel

#-custom mean class-#
class SlicedMean(gpytorch.means.Mean):
    def __init__(self,input_dim, base_mean=None):
        super().__init__()
        self.base_mean = base_mean if base_mean is not None else ConstantMean()
        self.input_dim = input_dim

    def forward(self, x):
        real_x = x[..., :self.input_dim]
        return self.base_mean(real_x)

#-orchestration GP class-#
class DeepGaussianProcess(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_experts, vit_module=None, dirichlet_module=None, vae_module=None, input_dim=128):
        """
        Args:
            inducing_points (Tensor): Shape (M, D + K). 
                                      Must be pre-augmented with initial mixing weights!
            dirichlet_module (nn.Module): Your external network.
            num_experts (int): K
            input_dim (int): D (feature dimension of raw data)
            likelihood: (Optional) Used for predictive strategy, not strictly needed in __init__ 
                        for SVGP but good practice.
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)) #- inducing points are learned

        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        super().__init__(variational_strategy)
        
        self.encoder = Encoder(config=VAEConfig())
        self.vae = NystromDecoder(config=VAEConfig())
        self.dirichlet = dirichlet_module if dirichlet_module is not None else AmortisedDirichlet(config=HDPConfig())
        self.input_dim = input_dim
        self.num_experts = num_experts

        #-mean module-#
        self.mean_module = SlicedMean(input_dim=self.input_dim)
        
        #-Cov module (stateless)-#
        self.covar_module = DeepKernel(num_experts=self.num_experts)

        self.auxiliary_loss = torch.tensor(0.0)
    
    def _comp_target_rff(self, z, spectral_means, bw, pi):
            omega = spectral_means * bw 

            proj = (z.unsqueeze(1).unsqueeze(1) * omega.unsqueeze(0)).sum(dim=-1)
            feat_cos = torch.cos(2 * math.pi * proj)
            feat_sin = torch.sin(2 * math.pi * proj)
            features_k = torch.cat([feat_cos, feat_sin], dim=-1)
            
            #-Weighted Sum across Experts (Amortization)
            target_rff = (features_k * pi.unsqueeze(-1)).sum(dim=1) # [B, 2M]
            #-scale-#
            M = spectral_means.size(1)
            target_rff = target_rff / math.sqrt(M)
            
            return target_rff.detach()

    def forward(self, x):
        #-if inducing points-#
        if x.dim() == 2 and x.size(-1) == (self.input_dim + self.num_experts):
            x_aug = x
            real_x = x[..., :self.input_dim]
            _, beta, spectral_means, bw = self.dirichlet(real_x)
            
        #-if raw data-#
        else:
            z = self.encoder(x)
            pi, beta, spectral_means, bw = self.dirichlet(z)
            if self.training:
                target_rff_tensor = self._comp_target_rff(z, spectral_means, bw, pi)
                vae_out = self.decoder(z)
                vae_loss = self.decoder.loss(vae_out, target_rff = target_rff_tensor)
                self.auxiliary_loss = vae_loss
            
            #-concat and feed back to kernel-#
            x_aug = torch.cat([z, pi], dim=-1)
        
        #-mean prediction-#
        mean_x = self.mean_module(x_aug)
        
        #-covariance prediction-#
        covar_x = self.covar_module(
            x_aug, 
            beta=beta, 
            spectral_means=spectral_means, 
            bw=bw
        )
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    
class DeepSpectralObjective(gpytorch.mlls.MarginalLogLikelihood):
    def __init__(self, likelihood, model, num_data, beta_vae=1.0):
        """
        A composite loss that combines:
        1. GP Marginal Log Likelihood (ELBO) - Maximizes GP fit
        2. VAE Loss - Regularizes the latent space (Recon + KL)
        """
        super().__init__(likelihood, model, num_data, beta=1.0)
        self.base_mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data)
        self.beta_vae = beta_vae

    def forward(self, function_dist, target, *args, **kwargs):
     
        gp_elbo = self.base_mll(function_dist, target, *args, **kwargs)
        
     
        vae_loss = self.model.auxiliary_loss
        
       
        total_objective = gp_elbo - (self.beta_vae * vae_loss)
        
        return total_objective