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
                 num_latents=8,
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
        # 1. The Omega Bank (Frequencies per cluster)
        # ==========================================
        # Shape: [k_atoms, input_dim, num_rff]
        self.register_parameter(
            name="raw_omega", 
            parameter=nn.Parameter(torch.randn(k_atoms, input_dim, num_rff))
        )
        
        # Optional: Biases for the phase shift
        self.register_parameter(
            name="phase_bias", 
            parameter=nn.Parameter(torch.rand(k_atoms, 1, num_rff) * 2 * math.pi)
        )

        # ==========================================
        # 2. The Multi-Latent Heads
        # ==========================================
        # Maps the dynamic RFF features to the multi-latent GP outputs
        self.heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(self.output_dim_per_latent, self.output_dim_per_latent) 
            ) 
            for _ in range(self.num_latents)
        ])
    
        self.register_parameter(
            name="raw_outputscale", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        
        self.register_parameter(
            name="raw_latent_amplitude", 
            parameter=torch.nn.Parameter(torch.zeros(num_latents, 1, 1))
        )
        self.register_constraint("raw_latent_amplitude", gpytorch.constraints.Positive())
        
        self.register_parameter(
            name="raw_inv_bandwidth", 
            parameter=torch.nn.Parameter(torch.zeros(1, input_dim))
        )
        self.register_constraint("raw_inv_bandwidth", gpytorch.constraints.Positive())

        self._init_weights()

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    
    @property
    def latent_amplitude(self):
        return self.raw_latent_amplitude_constraint.transform(self.raw_latent_amplitude)

    @property
    def inv_bandwidth(self):
        return self.raw_inv_bandwidth_constraint.transform(self.raw_inv_bandwidth)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
    

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        pi = params.get("pi", None)
        if pi is None:
            pi = torch.full((x1.size(0), self.k_atoms), 1.0/self.k_atoms, device=x1.device)
        #-Get Latent Features: [num_latents, Batch, 256]
        dynamic_omega = torch.einsum('bk, kdf -> bdf', pi, self.raw_omega)
        dynamic_phase = torch.einsum('bk, kof -> bof', pi, self.phase_bias)

        # ---------------------------------------------------------
        # 3. True Random Fourier Features (RFF) Projection
        # ---------------------------------------------------------
        # Projection: x1 -> [Batch, 1, input_dim] @ [Batch, input_dim, num_rff] -> [Batch, 1, num_rff]
        proj_x1 = torch.bmm(x1.unsqueeze(1), dynamic_omega) + dynamic_phase
        proj_x1 = proj_x1.squeeze(1)

        # Apply trigonometric activations to enter the spectral domain
        rff_x1 = torch.cat([torch.cos(proj_x1), torch.sin(proj_x1)], dim=-1)
        
        # Scale by 1/sqrt(num_rff) to maintain variance
        rff_x1 = rff_x1 / math.sqrt(self.num_rff)

        # ---------------------------------------------------------
        # 4. Latent Heads & Operators
        # ---------------------------------------------------------
        z1_list = [head(rff_x1) for head in self.heads]
        
        z1 = torch.stack(z1_list, dim=0) # [num_latents, Batch, 256]

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