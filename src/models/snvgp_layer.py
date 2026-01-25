from src.models.model_config import RootConfig
import gpytorch
import torch
import math
import torch.nn as nn

class SpectralVariationalGaussianProcess(gpytorch.models.ApproximateGP):
    """Bottlenecked Random Fourier GP - optimised for SGLD given a spectrally normalised input"""
    def __init__(self, config: RootConfig):
        self.config = config
        self.K = config.model.dirichlet.n_global
        self.M = config.model.gp.fourier_dim
        self.Fdim = self.K * self.M * 2 #--2 times for both cos and sin projections in rff space--#
        self.bottleneck_dim = config.model.gp.bottleneck_dim #--latent weights default to 1024-#

        proj = torch.empty(self.bottleneck_dim, self.Fdim)
        nn.init.orthogonal_(self.random_orthogonal_proj)
        self.register_buffer('random_orthogonal_proj', proj)

        #--Cholesky variational distribution over weight space-#
        #-posterior: Q(w) ~ N(m, L)-#
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=self.bottleneck_dim)

        #--Bayesian Linear Regression on projected vectors in bottleneck space-#
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points=self.random_orthogonal_proj,
            variational_distribution=variational_distribution,
            learn_inducing_locations=False
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.means.LinearKernel()
        self.register_buffer('scaling_constant', torch.tensor(math.sqrt(2.0 / self.M)))
        self.eps = getattr(config.model, 'eps', 1e-6)

    def _spectral_map(self, x, **kwargs):
        """
        Projects x into the HDP-defined spectral manifold
        """
        ##-x: [x_batch, latent_dim]
        ##-mu_atom: [k, M, latent_dim]-
        ##-weights_in: [x_batch, k] or [k] if no local clusters
        mu_atom = kwargs['mu_atom']
        log_sigma_atom = kwargs['log_sigma_atom']
        weights_in = kwargs['weights']
        
        self.M_in = mu_atom.size(1) #-dynamic dimension from dirichlet module-# 
        batch_dim = x.size(0)
        
        #-- Kernel Sampling (reparameterisation trick)-#
        sigma = torch.exp(log_sigma_atom)
        eps = torch.randn_like(mu_atom)
        omega = mu_atom + sigma * eps #-- out: [K, M, latent_dim]
        
        #- Project HDP atoms to [K*M, latent_dim]--#
        omega_flat = omega.view(-1, x.size(-1)) #--[K*M, latent_dim]
        proj = x @ omega_flat.t() #-[x_batch, K*M]-
        
        #-Trigonometric Mapping for random fourier proj--#
        phi_cos = torch.cos(proj) * self.scaling_constant
        phi_sin = torch.sin(proj) * self.scaling_constant

        #-Expansion of Dirichlet mixture weighting (weight space): expands K weights -> K * M weights-#
        if weights_in.dim() == 1:
            #-- 1 set of weights for whole batch: shape = [K]--#
            wmap = weights_in.view(-1, 1).repeat(1, self.M_in).view(-1) #-[K*M]-#
            wscl = torch.sqrt(wmap + self.eps).unsqueeze(0)
        else:
            #--Weights differ per sample: shape: [batch, K]-#
            #--flow: [batch, K, 1] -> [batch, K, M] -> [batch, K*M]
            wmap = weights_in.unsqueeze(-1).repeat(1, 1, self.M_in).view(batch_dim, -1)
            wscl = torch.sqrt(wmap + self.eps)
        
        #-Apply mixture weights-#
        phi_cos = phi_cos * wscl
        phi_sin = phi_sin * wscl
        
        #-concat-#
        phi = torch.cat([phi_cos, phi_sin], dim=-1)

        return phi

    def forward(self, x, **kwargs):
        mu_atom_hdp = kwargs['mu_atom']
        log_sigma_atom_hdp = kwargs['log_sigma_atom']
        weights = kwargs['weights']
        phi_x = self._spectral_map(x, mu_atom=mu_atom_hdp, log_sigma_atom=log_sigma_atom_hdp, weights=weights) #-Computes high-dim spectral features--#
        return self.variational_strategy(phi_x)
        