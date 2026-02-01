import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

class InducingPointSelect(BaseEstimator, TransformerMixin):
    def __init__(self, n_inducing=700, tolerance_threshold=1e-6, kernel_lengthscale=1.0):
        self.n_inducing = n_inducing
        self.tol = tolerance_threshold
        self.lengthscale = kernel_lengthscale
        self.inducing_points_ = None
        self.chol_factor_ = None
        self.eps = 1e-10
    
    def _rbf_diag(self, X):
        return np.ones(X.shape[0])
    
    def _kernel_column(self, X, piv_idx):
        """pv_idx is the index corresponding to max diagonals (pivots)"""
        xpiv = X[piv_idx : piv_idx + 1]
        sqdist = np.sum(X**2, axis=1) + np.sum(xpiv**2, axis=1) - 2 * np.dot(X, xpiv.T).flatten()
        return np.exp(-0.5 * sqdist / (self.lengthscale**2))

    def fit(self, X, y=None):
        """
        Greedy selection using RBF kernel: K(x, x) = 1 
        -- picks point with highest uncertainty measured by kernel diag values
        """
        X = check_array(X)
        n_samples = X.shape[0]
        m = min(self.n_inducing, n_samples)
        diags = self._rbf_diag(X)
        pivs = []
        L = np.zeros((n_samples, m))

        for i in range(m):
            piv_idx = np.argmax(diags)
            error = diags[piv_idx] #-error refers to max error-#

            if error < self.tol:
                L = L[:, :i] #-early convergence-#
                break
            
            pivs.append(piv_idx)
            k_star = self._kernel_column(X, piv_idx)

            #--schur complement for chol factor-#
            if i == 0:
                L[:, i] = k_star / np.sqrt(error)
            
            else:
                L[:, i] = (k_star - (L[:, :i] @ L[piv_idx, :i])) / np.sqrt(error)
            
            diags -= L[:, i]**2 #-deflation-#
            diags = np.maximum(diags, self.eps)

        self.inducing_points_ = X[pivs].copy()
        self.chol_factor_ = L
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        #-Project X onto RBF-projected inducing points-#
        check_is_fitted(self)
        X = check_array(X)
        return self._compute_kernel(X, self.inducing_points_)

    def _compute_kernel(self, X, inducing):
        """pairwise second order euclidean distance"""
        sqdist = np.sum(X**2, axis=1, keepdims=True) + np.sum(inducing**2, axis=1) - 2 * (X @ inducing.T)
        return np.exp(-0.5 * sqdist / (self.lengthscale**2))