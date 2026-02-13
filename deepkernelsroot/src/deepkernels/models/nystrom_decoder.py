import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import math

class NystromDecoder(nn.Module):
    def __init__(self, 
                 config,
                 latent_dim=None,
                 hidden_dim=None,
                 output_dim=None, # Input dim of the original data (D)
                 num_landmarks=None, # M (Number of spectral features)
                 num_kernels=None, # K (Number of mixture components)
                 depth=3):
        
        super().__init__()
        
        # --- Config & Hyperparams ---
        self.config = config
        self.latent_dim = latent_dim or config.latent_dim
        self.hidden_dim = hidden_dim or config.hidden_dim
        self.output_dim = output_dim or config.input_dim
        self.num_landmarks = num_landmarks or config.M
        self.num_kernels = num_kernels or config.k_atoms
        self.depth = depth

        # --- Stability: Fixed RFF Frequencies (Omega) ---
        self.register_buffer(
            "omega", 
            torch.randn(self.num_kernels, self.num_landmarks, self.output_dim)
        )
        
        # --- 1. Latent Projection (z -> hidden) ---
        self.z_proj = P.spectral_norm(nn.Linear(self.latent_dim, self.hidden_dim))
        
        # --- 2. Deep Geometric Backbone ---
        layers = []
        for _ in range(self.depth):
            layers.append(P.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)))
            layers.append(nn.LayerNorm(self.hidden_dim)) # Crucial for deep gradient flow
            layers.append(nn.SiLU()) # SiLU is smoother than ReLU, better for spectral reconstruction
            
        self.backbone = nn.Sequential(*layers)

        # --- 3. Heads ---
        self.rkhs_head = nn.Sequential(
            P.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.num_landmarks) 
        )

        self.alpha_head = nn.Linear(self.hidden_dim, self.num_kernels)
        self.ls_head = nn.Linear(self.hidden_dim, self.num_kernels * self.output_dim)

    def forward(self, z):
        """
        Args:
            z: Latent samples [Batch, latent_dim] (Sampled by the Encoder!)
        """
   
        h = self.z_proj(z)
        h = F.silu(h)
        
      
        h_geo = self.backbone(h)
      
        recon_rff = self.rkhs_head(h_geo)
      
        alpha_logits = self.alpha_head(h_geo)
        alpha = F.softmax(alpha_logits, dim=-1) #-sums to 1-#
        
        ls = F.softplus(self.ls_head(h_geo)) + 1e-5
        ls = ls.view(-1, self.num_kernels, self.output_dim)

        return {
            "recon_rff": recon_rff, #-decoder dream of rff kernel-#
            "alpha": alpha,
            "ls": ls,
            "z_ctx": h_geo 
        }

    def compute_loss(self, vae_outputs, x_true):
        """
        Stable Spectral Loss.
        Instead of reconstructing pixels, we reconstruct the RFF embedding of x.
        
        Args:
            vae_outputs: Dict from forward()
            x_true: Original input data [Batch, D]
        """
        pred_rff = vae_outputs['recon_rff']
        alpha = vae_outputs['alpha']
        ls = vae_outputs['ls']
        
        # 1. Calculate TRUE RFF target from the actual input x
        # We use the fixed 'omega' buffer and predicted 'ls' to scale frequencies
        with torch.no_grad():
            # Scale frequencies by lengthscale (simulating stationary kernel)
            # We average over K kernels for the target to keep it simple, 
            # or use the dominant alpha. Here we assume K=1 for target generation or mean.
            # Simplified for stability: Use raw omega
            target_proj = torch.matmul(x_true, self.omega[0].T) # [B, M]
            target_cos = torch.cos(2 * math.pi * target_proj)
            target_sin = torch.sin(2 * math.pi * target_proj)
            target_rff = torch.cat([target_cos, target_sin], dim=-1) # [B, 2M]
            
            # Normalize
            target_rff = target_rff / math.sqrt(self.num_landmarks)

        # 2. Spectral MSE Loss
        recon_loss = F.mse_loss(pred_rff, target_rff, reduction='mean')
        
        return recon_loss