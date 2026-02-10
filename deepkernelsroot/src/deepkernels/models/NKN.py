import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

class NeuralKernelNetwork(nn.Module):
    """
    The 'Neural Kernel' Head.
    It takes the stable, flat latent code 'z' and explodes it into 
    structural primitives (Linear, Periodic, Multiplicative) for the GP.
    """
    def __init__(self, input_dim, n_experts, features_per_expert, hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.H = hidden_dim if hidden_dim else 32
        self.n_experts = n_experts
        self.output_dim = features_per_expert #-gp feature dim-#
        
        #-Linear Primitive-=global trends-#
        #kernel: linear
        self.linear = sn(nn.Linear(self.input_dim, self.H))
        self.linear_scale = nn.Parameter(torch.tensor(0.13))
        
        #- Periodic Primitive - texture -#
        #- kernel: spectral mixture
        self.periodic = sn(nn.Linear(self.input_dim, self.H))

        # --- 3. RBF Primitive (Local Smoothness) ---
        #- kernel: RBF (sq exponential)
        self.rbf = sn(nn.Linear(self.input_dim, self.H))

        # --- 4. Rational Primitive (Multi-Scale/Heavy Tail) ---
        # Kernel: Rational Quadratic
        # Activation: 1 / (1 + (Wx + b)^2) -> Cauchy/Inverse Multiquadric
        self.rational = sn(nn.Linear(self.input_dim, self.H))

        # --- 5. Constant Primitive (Bias) ---
        # Kernel: Constant
        self.constant = nn.Parameter(torch.randn(1, self.H) * 0.13)
        
        self.primitive_norm = nn.LayerNorm(self.H * 9)

        #-kernel mixer-#
        # Mixer: 5 (Bases) + 4 (Interactions) = 9
        self.mixer = nn.Sequential(
            sn(nn.Linear(self.H * 9, self.H * 12)),
            nn.SiLU(),
            nn.LayerNorm(self.H * 12),
        )
        
        self._init_weights()

        self.dense_multitask_proj = sn(nn.Linear(self.H * 12, self.output_dim * self.n_experts))
    
    def _init_weights(self):
        # 1. Linear Primitive (Identity)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        
        # 2. Periodic Primitive (Cosine) -> gain = sqrt(2)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        
        # 3. RBF Primitive (Gaussian) - centres of rbfs
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        
        # 4. Rational Primitive (Cauchy)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)

        # 5. The Mixer
        # Standard Orthogonal Init for the MLP part
        for module in self.mixer.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('silu'))

        nn.init.orthogonal_(self.dense_multitask_proj.weight, gain=1.0)
        
    def forward(self, z):
        #-primitive kernels-#
        lin_out = self.linear(z) * self.linear_scale
        period_out = torch.cos(self.periodic(z))
        rbf_out = torch.exp(-torch.pow(self.rbf(z), 2))
        rat_out = 1.0 / (1.0 + torch.pow(self.rational(z), 2))
        const_out = self.constant.expand(z.size(0), -1) #-count = 5/9-#

        #-interactions-#:
        seasonal = lin_out * period_out
        local = lin_out * rbf_out
        decay = period_out * rbf_out
        tails = period_out * rat_out

        combined = torch.cat([lin_out, period_out, rbf_out, rat_out, const_out, seasonal, local, decay, tails], dim=-1)

        combined = self.primitive_norm(combined)

        mixed = self.mixer(combined)

        #-flatten [batch, K *output dim]-#
        flat = self.dense_multitask_proj(mixed)

        #-shape out: [Batch, K, output_dim]
        return flat.view(-1, self.n_experts, self.output_dim)