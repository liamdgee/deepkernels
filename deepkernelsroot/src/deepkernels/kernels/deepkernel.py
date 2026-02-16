import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch
import math
import gpytorch
from gpytorch.kernels import Kernel, LinearKernel
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
        latent_means = pi @ self.cluster_constants
        return latent_means.t()



class DeepKernel(Kernel):
    """
    master kernel
    """
    has_lengthscale = False

    def __init__(self, 
                 input_dim=2048, 
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
    
        self.register_parameter(
            name="raw_outputscale", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        
        # 2. PER-LATENT AMPLITUDE (Renamed for clarity)
        # We initialize to 0.0 -> Softplus(0.0) approx 0.69 amplitude
        self.register_parameter(
            name="raw_latent_amplitude", 
            parameter=torch.nn.Parameter(torch.zeros(num_latents, 1, 1))
        )
        self.register_constraint("raw_latent_amplitude", gpytorch.constraints.Positive())
        
        # 3. INVERSE BANDWIDTH (Your Spectral Filter)
        self.register_parameter(
            name="raw_inv_bandwidth", 
            parameter=torch.nn.Parameter(torch.zeros(1, input_dim))
        )
        self.register_constraint("raw_inv_bandwidth", gpytorch.constraints.Positive())

        self._init_weights()

    @property
    def outputscale(self):
        # CORRECT: Access constraint directly on self
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    
    @property
    def latent_amplitude(self):
        # Renamed property to match the parameter logic
        return self.raw_latent_amplitude_constraint.transform(self.raw_latent_amplitude)

    @property
    def inv_bandwidth(self):
        return self.raw_inv_bandwidth_constraint.transform(self.raw_inv_bandwidth)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)
    

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        #-Get Latent Features: [num_latents, Batch, 256]
        bw = self.inv_bandwidth
        x1 = x1 * bw
        
        hidden_x1 = self.mixer(x1) #-shared-# [b, 768]

        z1_list = [head(hidden_x1) for head in self.heads]
        z1 = torch.stack(z1_list, dim=0)

        amp = self.latent_amplitude.sqrt()
        z1 = z1 * amp
        
        if x2 is None or torch.equal(x1, x2):
            return RootLinearOperator(z1) #-[6, b, b]
            
        else:
            x2 = x2 * bw
            hidden_x2 = self.mixer(x2)
            
            z2_list = [head(hidden_x2) for head in self.heads]
            z2 = torch.stack(z2_list, dim=0)
            z2 = z2 * amp
            
            if diag:
                return (z1 * z2).sum(-1)
            else:
                # Matmul: (6, Batch, D) @ (6, D, Batch) -> (6, Batch, Batch)
                return MatmulLinearOperator(z1, z2.transpose(-1, -2))