import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm
from deepkernels.models.parent import BaseGenerativeModel

class SpectralDecoder(BaseGenerativeModel):
    def __init__(self, 
                 input_dim=30,       # Output shape (reconstruction)
                 spectral_dim=256,   # Features per cluster (M*2)
                 num_clusters=30,
                 spectral_emb_dim=2048,
                 input_dim_data=30,
                 bottleneck_dim=64,
                 num_experts=8,
                 k_atoms=30,
                 hidden_dims=None,
                 dropout=0.1):
        super().__init__()
        
        spectral_compressions = [1024, 512, 128, 64]
        current_dim = spectral_emb_dim
        layers = []
        layers.append(nn.LazyLinear(current_dim))
        for compression in spectral_compressions:
            layers.append(torch.nn.utils.spectral_norm(nn.Linear(current_dim, compression)))
            layers.append(nn.LayerNorm(compression))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            current_dim = compression
        
        #-current_dim is now 64
        self.compression_network = nn.Sequential(*layers)

        self.ls_head_recon = nn.Sequential(
            nn.Linear(4, input_dim_data),
            nn.Tanh()            
        )
        self.pi_head_recon = nn.Sequential(
            nn.Linear(30, input_dim_data),
            nn.LayerNorm(input_dim_data),
            nn.Sigmoid()
        )
        self.data_head_recon = nn.Sequential(
            nn.LayerNorm(30),
            nn.utils.spectral_norm(nn.Linear(30, input_dim_data))
        )

        self.expert_variational_head = torch.nn.utils.spectral_norm(nn.Linear(bottleneck_dim, num_experts * k_atoms))
        self.expert_logit_head = torch.nn.utils.spectral_norm(nn.Linear(bottleneck_dim, num_experts * k_atoms))

        self.mu_alpha = nn.Linear(bottleneck_dim, k_atoms)
        self.chol_alpha = nn.Linear(bottleneck_dim, k_atoms * 3)
        self.diag_alpha = nn.Linear(bottleneck_dim, k_atoms)

        self.scale_head = torch.nn.utils.spectral_norm(nn.Linear(bottleneck_dim, num_experts))
    
    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)
    
    def dirichlet_sample(self, alpha):
        alpha = torch.clamp(alpha, min=1e-3, max=100.0)
        q_alpha= torch.distributions.Dirichlet(alpha)
        pi_sample = q_alpha.rsample()
        return pi_sample


    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        """
        Args:
            spectral_features: [Batch, K*M*2] OR [Batch, K, M*2]
        Returns:
            recon_x: [Batch, input_dim]
        """
        
        spectral_bottleneck = self.compression_network(x)
        
        bottleneck = torch.tanh(spectral_bottleneck)

        recon, trend, amp, residuals = self.disentangle(bottleneck)

        var_out = self.expert_variational_head(bottleneck).view(batch_shape, self.num_experts, self.k_atoms)
        latent_functions_per_expert = var_out.permute(1, 0, 2)

        mu, chol, diag = self.get_alpha_mvn_heads_decoder(bottleneck)

        alpha_logits = self.multivariate_projection(mu, chol, diag)

        self.log_alpha_kl_low_rank(mu, chol, diag)

        pi = self.dirichlet_sample(alpha_logits)

        bandwidth_modulator_per_expert = torch.sigmoid(self.scale_head(bottleneck)) * 2.0

        mixture_means_per_expert = self.expert_logit_head(bottleneck).view(batch_shape, self.num_experts, self.k_atoms).permute(1, 0, 2)

        return {
            "alpha_logits": alpha_logits,
            "mixture_means": mixture_means_per_expert,
            "latent_experts": latent_functions_per_expert,
            "recon": recon,
            "bandwidth_mod": bandwidth_modulator_per_expert,
            "pi": pi,
            "amp": amp,
            "trend": trend,
            "res": residuals
        }