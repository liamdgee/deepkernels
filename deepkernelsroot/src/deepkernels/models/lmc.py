import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from src.losses.inverse_wishart import InverseWishartPenalty
from gpytorch.lazy import KroneckerProductLinearOperator
from gpytorch.lazy import RootLinearOperator
from gpytorch.lazy import KroneckerProductLazyTensor
from gpytorch.lazy import RootLazyTensor
from gpytorch.lazy import DiagLazyTensor
from gpytorch.lazy import NonLazyTensor
from gpytorch.lazy import BlockDiagLazyTensor


class LinearCoregionalisation(gpytorch.Module):
    def __init__(self, num_tasks, rank=2, iw_strength=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.rank = rank
        self.inv_wishart_strength = iw_strength
        self.jitter = 1e-6
        
        #--goal: learn coregionalisation matrix 'B' -- B = W W^T + diag(v)--#
        self.lmc_w = nn.Parameter(torch.randn(num_tasks, rank) * 0.1)
        self.lmc_var = nn.Parameter(torch.zeros(num_tasks)) # softplus(0) = 0.69
        
        # --- Inverse Wishart Prior Buffers (psi: mean 0 (tasks are indepenjdent, nu: degrees freedom)) ---
        self.register_buffer("pr_psi", torch.eye(num_tasks))
        self.register_buffer("pr_nu", torch.tensor(num_tasks + 2.0))
    
    def _calc_b_mat(self):
        W = self.lmc_w
        v = F.softplus(self.lmc_var)
        B = torch.matmul(W, W.t()) + torch.diag(v + self.jitter)
        return B
    
    def forward(self, check_prior=True):
        """
        register loss in model.foward() stage in training
        """
        B = self._calc_b_mat()
        if check_prior:
            iw_loss = InverseWishartPenalty(B, self.pr_nu, self.pr_psi)
            self.update_added_loss_term("lmc_inv_wishart_prior", iw_loss)
            
        return B
    
    def correlate_multitask_distributions(self, latent_dist):
        """
        Input: latent_dist (Batch: [Num_Tasks], Event: [N])
        Output: MultitaskMultivariateNormal (Batch: [], Event: [N, Num_Tasks])
        """
        #-fetch coregionalisation matrix-#
        B = self._calc_b_mat()
        B_lazy = gpytorch.lazy.NonLazyTensor(B)
        latent_mean = latent_dist.mean.t() # [N, T]
        #-Cholesky linear mixing-#
        L = torch.linalg.cholesky(B)
        #-assume gp means are standard normal -- [N, T] @ [T,T] -> [N, T] (chol transform)-#
        mixed_mean = torch.matmul(latent_mean, L.t())
        #-construct covar module-#
        #-K_f = B x K_latent (where x = kronecker product)-#
        #-manually inject task correlation after reshaping latent_dist to a multitask obj-#
        #-assume k_latent is shared across tasks and use cxovar of first task as shared K_xx-#
        K_xx = latent_dist.lazy_covariance_matrix[0]
        
        #-Compute Kronecker Product-#
        covar = gpytorch.lazy.KroneckerProductLazyTensor(B_lazy, K_xx)
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mixed_mean, covar)