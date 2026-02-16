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
from linear_operator.operators import RootLinearOperator, MatmulLinearOperator
from gpytorch.means import Mean

class DynamicMixtureMean(Mean):
    def __init__(self, num_latents=6, k_atoms=30, **kwargs):
        super().__init__(**kwargs)
        self.num_latents = num_latents
        self.k_atoms = k_atoms
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(self.k_atoms, self.num_latents))
        )
    
    def forward(self, x, **params):
       
        pi = params.get("pi", None)

        if pi is None:
            return torch.zeros(x.shape[0], self.num_latents, device=x.device)
        
        # (Batch, 30) @ (30, 6) -> (Batch, 6)
        return pi @ self.cluster_constants



class DeepKernel(Kernel):
    """
    A unified Deep Kernel module.
    1. Neural Phase: Transforms latent 'z' into high-dimensional task features.
    2  Kernel Phase: Maps task features into an approximate Linear kernel with a matmul linear operator
    """
    has_lengthscale = False

    def __init__(self, 
                 input_dim=7680, 
                 k_atoms=30, 
                 num_latents=6,
                 latent_dim=16, 
                 num_rff=128, 
                 data_dim=30, 
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # --- Config ---
        self.input_dim = input_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.k_atoms = k_atoms
        self.output_dim_per_latent = num_rff * 2 #- 6 latents are assumed equivalent to 30 clusters-#
        self.input_dim = 2048 #-dirichlet spectral embedding dim-#
        self.spectral_emb = num_rff * num_latents #-768

        # ==========================================
        # Neural Network Architecture
        # ==========================================

        self.mixer = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Linear(self.input_dim, self.spectral_emb)),
            nn.SiLU(),
            nn.LayerNorm(self.spectral_emb)
        )

        self.heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(self.spectral_emb, self.output_dim_per_latent) #-dim 128 for each latent gp-#
            ) 
            for _ in range(self.num_latents)
        ])


        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        
        self.register_parameter(name="raw_log_amplitude", parameter=torch.nn.Parameter(torch.zeros(num_latents, 1, 1)))
        self.register_constraint("raw_log_amplitude", gpytorch.constraints.Positive())
        
        self._init_weights()

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def _init_weights(self):
        

        for module in self.mixer.modules():
            if isinstance(module, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(module.weight, gain=gain)
        
        for head in self.heads.modules():
            if isinstance(head, nn.Linear):
                nn.init.orthogonal_(head.weight, gain=1.41)
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    
    

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # 1. Get Latent Features: [num_latents, Batch, 256]
        
        bw_learned = params.get("bandwidth", None)

        if bw is not None:
            x1 = x1
        
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
        
        scale = self.outputscale.sqrt().view(1, 1, 1) 
        z1 = z1 * scale
        if not symmetric:
            z2 = z2 * scale
            
        if symmetric:
            return RootLinearOperator(z1)
        
        else:
            # Result: MatmulLinearOperator of shape [6, Batch1, Batch2]
            return MatmulLinearOperator(z1, z2.transpose(-1, -2))