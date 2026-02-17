import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
from deepkernels.models.parent import BaseGenerativeModel

class KernelNetwork(BaseGenerativeModel):
    def __init__(self, bottleneck_dim=64, num_latents=8, spectral_emb_dim=2048):
        super().__init__()

        # --- Dimensions ---
        self.individual_kernel_dim_out = 32
        self.primitive_products = 12
        self.num_primitives = 4 # [linear, periodic, rbf, rational]
        
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

        self.spectral_feedback_loop = nn.Sequential(
            torch.nn.utils.spectral_norm(
                nn.Linear(self.head_input_dim, spectral_emb_dim)),
                nn.Sigmoid()
        )

        self.init_weights_nkn()
    
    
    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), **params):
        """
        This inputs the latent bottleneck dim (64)
        """
        lin, per, rbf, rat = self.compute_primitives(x)        

        custom_interactions = self.compute_kernel_interactions(lin, per, rbf, rat) #- outputs [B, 12, 128]

        kernel_features = torch.cat([lin, per, rbf, rat, custom_interactions], dim=-1) #-[B,256]
        
        return self.feed_dirichlet_gate(kernel_features)
    
    