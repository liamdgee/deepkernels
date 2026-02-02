import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    def __init__(self, input_dim, n_mixtures, hidden_dim=32, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.hidden_dim = hidden_dim
        self.device = kwargs.get('device')
        self.jitter = 1e-6

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh() #-bounds feature vectors btwn (-1, 1)-#
        )

        self.dirichlet_alpha_head = nn.Linear(self.hidden_dim, self.n_mixtures)

        self.kernel_lengthscale_head = nn.Linear(self.hidden_dim, self.input_dim * self.n_mixtures)
    
    def forward(self, x):
        features = self.mlp(x)

        #---Concentrations for variational dirichlet module--#
        alpha = F.softplus(self.dirichlet_alpha_head(features)) + self.jitter

        #--lengthscales for kernel (reshapes [batch, mixtures, dims])--#
        lengthscales = F.softplus(self.kernel_lengthscale_head(features))
        lengthscales = lengthscales.view(-1, self.n_mixtures, self.input_dim)

        return alpha, lengthscales
