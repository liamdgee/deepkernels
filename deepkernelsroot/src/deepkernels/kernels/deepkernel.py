import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch
import math
import gpytorch
from gpytorch.kernels import Kernel
from linear_operator.operators import LowRankRootLinearOperator, RootLinearOperator, MatmulLinearOperator
from src.deepkernels.models.encoder import RecurrentEncoder

class DynamicMixtureMean(gpytorch.means.Mean):
    def __init__(self, k_atoms=30):
        super().__init__()
        self.k_atoms = k_atoms
        self.register_parameter(name="cluster_constants", parameter=torch.nn.Parameter(torch.zeros(k_atoms)))

    def forward(self, x, pi=None):
        """
        Args:
            x: Raw input [Batch, D]
            pi: Mixture weights [Batch, K] from the VAE/Encoder
        """
        if pi is None:
            return torch.zeros(x.size(0), device=x.device)
        
        return (pi * self.cluster_constants).sum(dim=-1)

class DeepKernel(Kernel):
    """
    A unified Deep Kernel module.
    1. Neural Phase: Transforms latent 'z' into high-dimensional task features.
    2   . Kernel Phase: Maps task features into an approximate RBF kernel using Orthogonal Random Features.
    """
    def __init__(self, num_latents=6, **kwargs):
        super().__init__(**kwargs)
        
        # --- Config ---
        self.num_latents = num_latents
        self.input_dim = 7680
        self.H = 16
        self.n_experts = 30
        self.output_dim_per_cluster = 256
        self.deep_feat_dim = 30 * 256
        self.n_samples = 512

        # ==========================================
        # Neural Network Architecture
        # ==========================================
        self.linear = sn(nn.Linear(self.input_dim, self.H))
        self.linear_scale = nn.Parameter(torch.tensor(0.13))
        self.periodic = sn(nn.Linear(self.input_dim, self.H))
        self.rbf = sn(nn.Linear(self.input_dim, self.H))
        self.rational = sn(nn.Linear(self.input_dim, self.H))
        self.constant = nn.Parameter(torch.randn(1, self.H) * 0.13)
        self.primitive_norm = nn.LayerNorm(self.H * 9)

        self.mixer = nn.Sequential(
            sn(nn.Linear(self.H * 9, self.H * 12)),
            nn.SiLU(),
            nn.LayerNorm(self.H * 12),
        )

        #-Parallel Task Heads-#

        self.parallel_task_layers = nn.ModuleList([
            sn(nn.Linear(self.H * 12, self.output_dim_per_latent)) 
            for _ in range(num_latents)
        ])

        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        self.log_amplitude = nn.Parameter(torch.zeros(self.n_experts))
        self._init_weights()

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def _init_weights(self):
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)

        for module in self.mixer.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('silu'))

        for layer in self.parallel_task_layers:
            nn.init.orthogonal_(layer.weight, gain=1.0)
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    
    def _compute_primitives_and_interactions(self, x):
        """
        Runs the Neural Network: x -> Primitives -> Mixer -> Task Heads -> Flattened Features
        """
        # A. Primitives
        lin_out = self.linear(x) * self.linear_scale
        period_out = torch.cos(self.periodic(x))
        rbf_out = torch.exp(-torch.pow(self.rbf(x), 2))
        rat_out = 1.0 / (1.0 + torch.pow(self.rational(x), 2))
        const_out = self.constant.expand(x.size(0), -1)
        
        # B. Interactions
        seasonal = lin_out * period_out
        local = lin_out * rbf_out
        decay = period_out * rbf_out
        tails = period_out * rat_out
        
        combined = torch.cat([lin_out, period_out, rbf_out, rat_out, const_out, seasonal, local, decay, tails], dim=-1)
        mixed = self.mixer(self.primitive_norm(combined))

        #-heads-#
        latent_features = torch.stack([head(mixed) for head in self.parallel_task_layers], dim=0)
        return latent_features / math.sqrt(self.output_dim_per_cluster) #-in this iteration, cluster means latent in kernel logic-#
    
    def forward(self, x1, x2, pi=None, diag=False, **params):
        # 1. Get Latent Features: [num_latents, Batch, 256]
        z1 = self._compute_primitives_and_interactions(x1)
        if x2 is None or torch.equal(x1, x2):
            z2 = z1
            symmetric = True
        else:
            z2 = self._compute_primitives_and_interactions(x2)
            symmetric = False
        
        amp = self.n_experts / self.num_latents

        z1 = z1 * amp
        if not symmetric:
            z2 = z2 * amp
        if diag:
            return (z1 * z2).sum(-1) # [6, Batch]
            
        if symmetric:
            return RootLinearOperator(z1)
        else:
            # Result: MatmulLinearOperator of shape [6, Batch1, Batch2]
            return MatmulLinearOperator(z1, z2.transpose(-1, -2))