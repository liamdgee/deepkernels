import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P


class SpectralEncoder(nn.Module):
    def __init__(self, input_dim, n_mixtures, hidden_dim=32, depth=4, use_spectral_norm=True, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.hidden_dim = hidden_dim
        self.device = kwargs.get('device')
        self.jitter = 1e-6
        self.depth = depth

        layers = []

        dims = [self.input_dim] + [self.hidden_dim] * (self.depth - 1)

        for dim in range(len(dims) - 1):
            indim = dims[dim]
            outdim = dims[dim+1]
            linear = nn.Linear(indim, outdim)
            if use_spectral_norm:
                linear = P.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(outdim))
            
        self.network = nn.Sequential(*layers)

        self.alpha_head = nn.Linear(hidden_dim, n_mixtures)
        
        self.lengthscale_head = nn.Linear(hidden_dim, n_mixtures * input_dim)

        if use_spectral_norm:
            self.alpha_head = P.spectral_norm(self.alpha_head)
            self.lengthscale_head = P.spectral_norm(self.lengthscale_head)

      
    def forward(self, x):
        features = self.network(x)

        #---Concentrations for variational dirichlet module--#
        alpha_raw = self.alpha_head(features)
        alpha = F.softplus(alpha_raw) + self.jitter

        #--lengthscales for kernel (reshapes [batch, mixtures, dims])--#
        ls_raw = self.lengthscale_head(features)
        ls = F.softplus(ls_raw) + self.jitter
        ls = ls.view(-1, self.n_mixtures, self.input_dim)

        return alpha, ls
