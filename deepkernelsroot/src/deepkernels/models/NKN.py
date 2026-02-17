import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
from deepkernels.models.parent import BaseGenerativeModel

class KernelNetwork(BaseGenerativeModel):
    def __init__(self):
        super().__init__()

        num_latents = 8
        bottleneck_dim = 64

        # --- Dimensions ---
        self.individual_kernel_dim_out = 32
        self.primitive_products = 12
        self.num_primitives = 4 # [linear, periodic, rbf, rational]

        ###self.omega = nn.Parameter(torch.randn(k_atoms, fourier_dim, latent_dim))
        
        self.primitives_total_dim = self.num_primitives * self.individual_kernel_dim_out # 4 * 32 = 128
        self.products_total_dim = self.primitive_products * self.individual_kernel_dim_out # 12 * 24 = 384
        
        self.compression_dim = 128
        
        self.head_input_dim = self.primitives_total_dim + self.compression_dim #-256
        
        self.linear = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))
        self.linear_scale = nn.Parameter(torch.tensor(0.1))
        
        self.periodic = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))
        self.rbf = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))
        self.rational = nn.utils.spectral_norm(nn.Linear(bottleneck_dim, self.individual_kernel_dim_out))

        #--all individaul kernels were projected to dim 32 (aggregate: 128)--#

        self.selection_weights = nn.Parameter(torch.randn(self.primitive_products, self.num_primitives)) #-tilde - normal (12, 4)

        self.complex_interactions = nn.utils.spectral_norm(
            nn.Linear(self.products_total_dim, self.compression_dim) #-384 -> 128
        )

        self.latent_kernel_heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(self.head_input_dim, self.individual_kernel_dim_out)
            ) 
            for _ in range(num_latents)
        ])

        emb_dim=248

        self.spectral_feedback_loop = torch.nn.utils.spectral_norm(
                nn.Linear(self.head_input_dim, emb_dim))
    
    def _init_weights(self):
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.uniform_(self.linear.bias, 0.0, 2.0)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)
        nn.init.orthogonal_(self.complex_interactions.weight, gain=1.41)

        for module in self.latent_kernel_heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)

    def forward(self, x, output_matrices=False):
        """This inputs the latent bottleneck dim (64)"""
        lin = self.linear(x) * self.linear_scale
        per = torch.cos(self.periodic(x))
        rbf = torch.exp(-torch.pow(self.rbf(x), 2))
        rat = 1.0 / (1.0 + torch.pow(self.rational(x), 2))

        custom_interactions = self._compute_kernel_interactions(lin, per, rbf, rat) #- outputs [B, 12, 96]

        #-cat [B, 4, 128] + [B, 12, 128] -> [B, 256]
        kernel_features = torch.cat([lin, per, rbf, rat, custom_interactions], dim=-1)
        
        #-outputs 8 latent gp kernels of [B, 32]
        latent_kernels = [latent(kernel_features) for latent in self.latent_kernel_heads]
        
        stacked_features = torch.stack(latent_kernels) #-[8 batch, 32]
        
        if output_matrices:
            cov_matrices = torch.einsum('lbd, lcd -> lbc', stacked_features, stacked_features)
            return cov_matrices #-[8, B, B]
        
        return self.spectral_feedback_loop(kernel_features)
    
    def _compute_kernel_interactions(self, lin, per, rbf, rat):
        "input stack: [B, 4, 32]"
        stack = torch.stack([lin, per, rbf, rat], dim=1)
        mask = torch.sigmoid(self.selection_weights).unsqueeze(-1)
        stack_safe = torch.abs(stack) + 1e-6
        log_stack = torch.log(stack_safe)
        # Weighted Einsum in log space: b p d (batch, primitives, dim), k p   (products, primitives) -> b k d (batch, products, dim)
        log_product = torch.einsum('bpd, kp -> bkd', log_stack, mask.squeeze(-1))
        product_features = torch.exp(log_product) #-[B, 12, 32]
        interactions_matrix = product_features.flatten(start_dim=1) #-[B, 384]
        return self.complex_interactions(interactions_matrix) #-[B, products, 128]