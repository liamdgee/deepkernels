import math

import gpytorch
import torch
import torch.nn as nn
from deepkernels.models.dummy_model_config import RootConfig

class HierarchicalDirichletProcess(nn.Module):
    def __init__(self, config: RootConfig):
        super().__init__()
        self.config = config if config is not None else RootConfig()
        self.k_atoms = self.config.n_global
        self.j_tables = self.config.n_local
        self.latent_dim = self.config.latent_dim

        #---Spectral Atoms---#
        self.M = self.config.fourier_dim
        self.mu_atom = nn.Parameter(torch.randn(self.k_atoms, self.M, self.latent_dim) * 0.1)
        self.log_sigma_atom = nn.Parameter(torch.full((self.k_atoms, self.M, self.latent_dim), -2.0)) #---exp(-2.0) approx equal to 0.13

         #---Learnable Weights for non-parametric clustering---#
        #---'v' params determine final cluster assignments---#
        self.v_k = nn.Parameter(torch.randn(self.k_atoms - 1)) #--Global--#
        self.v_j = nn.Parameter(torch.randn(self.j_tables, self.k_atoms - 1)) #--local--#

        alpha0 = self.config.alpha
        gamma0 = self.config.gamma
        
        #--Learnable Params with gradient flow to tune cluster sparsity---#
        if self.config.learnable_params:
            self.alpha = nn.Parameter(torch.tensor(alpha0))
            self.gamma = nn.Parameter(torch.tensor(gamma0)) 
        #---Fixed Priors (fixed with buffer state)---#
        #---No gradient flow to the below params---#
        else:
            self.register_buffer('alpha', torch.tensor([float(alpha0)]))
            self.register_buffer('gamma', torch.tensor([float(gamma0)]))
        
        #--Observation noise Prior--#
        self.register_buffer('sigma_noise', torch.tensor(self.config.sigma_noise))

    def _break_stick(self, v_raw, concentration):
        """GEM Construction"""
        v = torch.sigmoid(v_raw - torch.log(concentration))
        cum_prod = torch.cumprod(1 - v, dim=-1)
        pi = torch.zeros(len(v) + 1, device=v.device)
        pi[0] = v[0]
        pi[1:-1] = v[1:] * cum_prod[:-1]
        pi[-1] = cum_prod[-1]
        return pi
    
    def get_weights(self, local_idx):
        """Compute hierarchical weights using fixed alpha/gamma state"""
        beta = self._break_stick(self.v_k, self.gamma)
        pi_raw = self._break_stick(self.v_j[local_idx], self.alpha)
        pi_j = pi_raw * beta
        weights = pi_j / (pi_j.sum() + 1e-8)
        return weights
    
    def forward(self, local_idx):
        return self.get_weights(local_idx)