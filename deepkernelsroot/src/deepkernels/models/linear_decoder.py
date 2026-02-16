import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

from deepkernels.models.parent import BaseGenerativeModel

#-where decoder_input_dim = k_atoms * M_inducing_points * 2
class BayesParametricDecoder(BaseGenerativeModel):
    def __init__(
            self, 
            input_dim_data=30, 
            feature_dim=2048, 
            latent_dim=16, 
            rff_dim=128, 
            k_atoms=30, 
            num_experts=6
    ):
        """
        Reconstructs the original input space from the latent code z.
        latent_dim: dim of z
        only takes in params that are in latent_dim ! no feature matrices! they go the kernel!!
        """
        super().__init__()
        bottleneck_dim = 64
        self.decode = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, bottleneck_dim)),
            nn.LayerNorm(bottleneck_dim),
            nn.Tanh()
        )
        
        gp_in = k_atoms * num_experts

        self.variational_heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(bottleneck_dim, k_atoms) #-64 -> 30
            ) 
            for _ in range(num_experts)
        ])
        
        self.sigma_min = self.register_buffer("raw_sigma_minimum", torch.tensor(0.00001))

        self.pi_head = nn.Linear(bottleneck_dim, gp_in)

        self.logit_heads = nn.ModuleList([
            torch.nn.utils.spectral_norm(
                nn.Linear(gp_in, k_atoms) #-dim 128 for each latent gp-#
            ) 
            for _ in range(num_experts)
        ])


        self.mu_alpha = nn.Linear(bottleneck_dim, k_atoms)
        self.chol_alpha = nn.Linear(bottleneck_dim, k_atoms * 3)
        self.diag_alpha = nn.Linear(bottleneck_dim, k_atoms)

        self.recon_head = nn.Linear(bottleneck_dim, input_dim_data)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)

    def forward(self, z):
        #-recon z-
        bottleneck = self.decode(z)

        recon_x = self.recon_head(bottleneck)

        variational_heads = [head(bottleneck) for head in self.variational_heads]

        latent_functions_per_expert = torch.stack(variational_heads, dim=0)

        mu_alpha_out = self.mu_alpha(bottleneck)
        chol_alpha = self.chol_alpha(bottleneck).view(-1, 30, 3)
        diag_alpha_raw = self.diag_alpha(bottleneck)
        diag_alpha = self.apply_softplus(diag_alpha_raw)

        alpha_logits = self.multivariate_projection(mu_alpha_out, chol_alpha, diag_alpha)

        pi_mixture_means = [head(alpha_logits) for head in self.logit_heads]
        mixture_means_per_expert = torch.stack(pi_mixture_means, dim=0)

        return alpha_logits, mixture_means_per_expert, latent_functions_per_expert, recon_x

    def apply_softplus(self, x, jitter=1e-6):
        return torch.nn.functional.softplus(x) + jitter
    
    def multivariate_projection(self, mu, factor, diag):
        """for alpha params: input projections from three alpha heads"""
        mvn = torch.distributions.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=factor,
            cov_diag=diag
        )
        
        logits = mvn.rsample()
        alpha = F.softplus(logits) + 1e-6
        
        return alpha
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        return mu