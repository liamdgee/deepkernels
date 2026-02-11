import torch
import gpytorch
import torch.nn as nn
from torch.distributions import kl_divergence as kl
from gpytorch.mlls import AddedLossTerm

class KLDivergence(AddedLossTerm):
    """loss term for hierarchical dirichlet process -- KL(q(rho)|p(rho))"""
    def __init__(self, qdist, pdist):
        self.qdist = qdist
        self.pdist = pdist
    def loss(self):
        return kl(self.qdist, self.pdist).sum()