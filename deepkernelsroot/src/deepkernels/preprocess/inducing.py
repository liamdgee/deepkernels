import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Annotated, Union
import torch

class InducingConfig(BaseModel):
    n_inducing: int = 512
    tolerance_threshold: float = 2e-6
    kernel_lengthscale: float = 1.0
    eps: float = 1e-10

class InducingPointSelect(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            config: Optional[InducingConfig] = None,
            n_inducing: Optional[int] = 512,
            tolerance_threshold: Optional[float] = 1e-6,
            kernel_lengthscale: Optional[float] = 1.0,
            eps: float = 1e-10,
            **kwargs
        ):
        self.config = config if config else InducingConfig()

        self.n_inducing = n_inducing if n_inducing is not None else self.config.n_inducing or 512
        self.tolerance_threshold = tolerance_threshold if tolerance_threshold is not None else self.config.tolerance_threshold or 1e-6
        self.kernel_lengthscale = kernel_lengthscale if kernel_lengthscale is not None else self.config.kernel_lengthscale or 1.0
        self.eps = eps or self.config.eps or 1e-10

        self.X_sq_norms_ = None
        self.inducing_indices_ = []
        self.n_inducing_actual_ = None
        self.inducing_points_ = None

    def fit(self, X, y=None):
        """
        Greedy selection using RBF kernel: K(x, x) = 1 
        -- picks point with highest uncertainty measured by kernel diag values
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        m = min(self.n_inducing, n_samples)

        self.X_sq_norms_ = np.sum(X**2, axis=1)

        diags = np.ones(n_samples) #-RBF kernel diag always equals 1.0-#

        L = np.zeros((n_samples, m))

        for i in range(m):
            #-Pivot Selection-#
            piv_idx = np.argmax(diags)
            max_error = diags[piv_idx]
            
            #-Convergence Check and prune-#
            if max_error < self.tolerance_threshold:
                L = L[:, :i]
                break
            
            self.inducing_indices_.append(piv_idx)
            
            #-Compute Kernel-#
            xpiv = X[piv_idx]
            xpiv_norm = self.X_sq_norms_[piv_idx]
            sq_dist = self.X_sq_norms_ + xpiv_norm - 2 * (X @ xpiv)
            sq_dist = np.maximum(sq_dist, self.eps)
            k_star = np.exp(-0.5 * sq_dist / (self.kernel_lengthscale**2))
            
            #-Schur Complement Update (Cholesky Logic)-#
            root_err = np.sqrt(max_error)
            if i == 0:
                L[:, i] = k_star / root_err
            else:
                projection = L[:, :i] @ L[piv_idx, :i]
                L[:, i] = (k_star - projection) / root_err
            
            #-Deflation-#
            diags -= L[:, i]**2
            diags = np.maximum(diags, self.eps)

        self.inducing_points_ = X[self.inducing_indices_].copy()
        self.n_inducing_actual_ = len(self.inducing_indices_)
        
        return self

    def transform(self, X):
        #-Project X onto RBF-projected inducing points-#
        check_is_fitted(self)
        X = check_array(X)
        return self._compute_kernel(X, self.inducing_points_)
    
    def get_inducing_tensor(self):
        """Helper to get clean torch tensor for GPyTorch"""
        check_is_fitted(self, ['inducing_points_'])
        return torch.tensor(self.inducing_points_, dtype=torch.float32)

    def _compute_kernel(self, X, inducing):
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(inducing**2, axis=1, keepdims=True)
        #--(N, 1) + (1, M) - (N, M) -> Broadcasts to (N, M)-#
        sq_dist = X_sq + Y_sq.T - 2 * (X @ inducing.T)
        sq_dist = np.maximum(sq_dist, 0)
        return np.exp(-0.5 * sq_dist / (self.kernel_lengthscale**2))