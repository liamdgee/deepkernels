import torch
import torch.nn as nn
import gpytorch
from gpytorch.mlls import AddedLossTerm

class InverseWishartPenalty(AddedLossTerm):
    def __init__(self, X, cov_mat, scale_mat, **kwargs):
        self.X = X
        self.cov_mat = cov_mat
        self.scale_mat = scale_mat
        self.device = kwargs.get('device')
        self.eps = 1e-4
    
    def loss(self):
        p = self.cov_mat.shape[-1].to(self.device)
        Binv = torch.inverse(self.cov_mat + torch.eye(p, device=self.device) * self.eps)
        logdet = torch.logdet(self.cov_mat + torch.eye(p, device=self.device) * self.eps)
        trace = torch.trace(torch.matmul(self.scale_mat, Binv))

        return 0.5 * trace + 0.5 * (self.X + p + 1) * logdet