import torch
import torch.nn as nn
import math
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.priors import NormalPrior
from linear_operator.operators import RootLinearOperator, MatmulLinearOperator

from src.deepkernels.models.beta_vae import SpectralVAE
from src.deepkernels.models.dirichlet import AmortisedDirichlet

class ScalableDynamicEigenKernel(Kernel):
    """
    O(N) Scalable Version of DynamicEigenKernel.
    Uses Explicit Feature Maps to approximate the non-stationary kernel.
    """
    def __init__(self, input_dim, n_mixtures=4, hidden_dim=32, vae=SpectralVAE, num_rff_features=128, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.K = n_mixtures
        self.hidden_dim = hidden_dim
        self.M = num_rff_features 
        self.jitter = 1e-6

        # -- Hypernetwork --
        self.vae = vae(input_dim=self.input_dim, k_atoms=self.K, hidden_dim=self.hidden_dim)

        # -- Static Parameters --
        self.register_parameter("raw_mixture_means", torch.nn.Parameter(torch.zeros(self.K, self.input_dim)))

        #--priors-#
        self.register_prior("mixture_mean_prior", NormalPrior(0.0, 1.0),"raw_mixture_means")

        # -- RFF Noise (Fixed) --#
        self.register_buffer("omega_noise", torch.randn(self.K, self.M, self.input_dim))
        self.register_buffer("bias_noise", torch.rand(self.K, self.M) * 2 * math.pi)

    @property
    def mixture_means(self):
        return self.raw_mixture_means

    def _generate_linear_features(self, x):
        """
        Computes explicit features z(x) for variational inference
        """
        vae_out = self.vae(x)
        w = vae_out['alpha']
        l = vae_out['ls']
        S = w.sum(dim=-1, keepdim=True) #-total var-#
        pi = torch.clamp(w / (S + self.jitter), min=self.jitter) #-mixing weights-#
        #-construct non-stationary harmonic frequencies (Gibbs RFF) -- scaled by bandwidth / inv ls
        inv_lengthscale = 1.0 / (l + self.jitter) 
        
        # Omega = Mean + (Noise * InvLengthscale)
        # We need to broadcast carefully.
        # omega_noise: [Q, M, D] -> [1, Q, M, D]
        noise = self.omega_noise.unsqueeze(0)
        # inv_lengthscale: [N, Q, D] -> [N, Q, 1, D]
        inv_ls = inv_lengthscale.unsqueeze(2)
        # mixture_means: [Q, D] -> [1, Q, 1, D]
        means = self.mixture_means.view(1, self.K, 1, self.input_dim)
        
        #-dynamic frequencies modelled as gaussian-#
        omega_x = means + (noise * inv_ls)
        
        proj = (x.view(-1, 1, 1, self.input_dim) * omega_x).sum(dim=-1)
        proj = proj + self.bias_noise.unsqueeze(0)
        
        #-scale-#
        mc_const = math.sqrt(2.0 / self.M) 
        scale = torch.sqrt(S.unsqueeze(2) * pi.unsqueeze(2)) * mc_const
        
        z_cos = torch.cos(proj) * scale
        z_sin = torch.sin(proj) * scale
        
        #-flatten-#
        feats = torch.cat([z_cos.flatten(1), z_sin.flatten(1)], dim=-1)
        return feats

    def forward(self, x1, x2, diag=False, **params):
        z1 = self._generate_linear_features(x1)
        if x1 is x2:
            z2 = z1
        else:
            z2 = self._generate_linear_features(x2)
        if diag:
            return (z1 * z2).sum(dim=-1)
        
        if x1 is x2:
            return RootLinearOperator(z1)
        else:
            return MatmulLinearOperator(z1, z2.transpose(-1, -2))