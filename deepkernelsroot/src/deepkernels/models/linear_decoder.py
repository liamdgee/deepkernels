import torch
import torch.nn as nn
import torch.nn.functional as F

#-where decoder_input_dim = k_atoms * M_inducing_points * 2
class BayesDecoder(nn.Module):
    def __init__(self, latent_dim=64, original_dim=256, sigma_min=1e-5):
        """
        Reconstructs the original input space from the latent code z.
        latent_dim: dim of z
        original_dim: dim of your input data x
        """
        super().__init__()
        self.mu_fn = nn.Linear(latent_dim, original_dim)
        self.logvar_fn = nn.Linear(latent_dim, original_dim)
        self.sigma_min = sigma_min

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