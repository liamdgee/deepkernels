import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

#-where decoder_input_dim = k_atoms * M_inducing_points * 2
class BayesDecoder(nn.Module):
    def __init__(self, latent_dim, batch_dim, sigma_min=1e-5):
        """
        Reconstructs the original input space from the latent code z.
        latent_dim: dim of z
        """
        super().__init__()
        self.mu_fn = nn.Linear(latent_dim, batch_dim)
        self.logvar_fn = nn.Linear(latent_dim, batch_dim)
        self.sigma_min = sigma_min

        # --- 1. Orthogonal Init for the Mean (Structure) ---
        # Acts like a random PCA projection at step 0
        nn.init.orthogonal_(self.mu_fn.weight)
        nn.init.constant_(self.mu_fn.bias, 0)

        # --- 2. Zero Init for the Variance (Stability) ---
        # Initialize logvar to 0 so variance starts at ~0.69 (softplus(0))
        # This prevents division-by-zero explosions in the first epoch
        nn.init.constant_(self.logvar_fn.weight, 0)
        nn.init.constant_(self.logvar_fn.bias, 0)

    def forward(self, z):
        mu = self.mu_fn(z)
        var = F.softplus(self.logvar_fn(z)) + self.sigma_min
        return mu, var

    def get_reconstruction_loss(self, z, x_target):
        """
        This is the 'Expected Log Likelihood' term for the VAE. - Gaussian negative log likelihood.
        """
        mu_x, var_x = self.forward(z)
        recon_loss = 0.5 * (torch.log(var_x) + (x_target - mu_x)**2 / var_x)
        return recon_loss.sum(dim=-1).mean()