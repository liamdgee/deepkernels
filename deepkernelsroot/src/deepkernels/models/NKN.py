import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

class NeuralKernelNetwork(nn.Module):
    """
    #-currently nonfunctional-#
    The 'Neural Kernel' Head.
    It takes the stable, flat latent code 'z' and explodes it into 
    structural primitives (Linear, Periodic, Multiplicative) for the GP.
    """
    def __init__(self, batch_dim=256, n_experts=30, features_per_expert=64, hidden_dim=16):
        super().__init__()
        self.batch_dim = batch_dim or 256
        self.input_dim = batch_dim * hidden_dim
        self.H = hidden_dim
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
        
    def forward(self, z, omega):
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
        flat_features = self.all_latent_heads(mixed)

        reshaped = flat_features.view(
            x.size(0), 
            self.num_latents, 
            self.output_dim_per_cluster
        )

        # 3. Permute to bring latents to the front (The "GP Batch" dimension)
        # [Batch, num_latents, 256] -> [num_latents, Batch, 256]
        latent_features = reshaped.permute(1, 0, 2)
        
        # 4. Normalize for RFF stability
        return latent_features / math.sqrt(self.output_dim_per_cluster)
