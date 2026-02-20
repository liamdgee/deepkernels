# filename: dirichlet_clusters.py
        #-stick breaking logic (Numerically Stable)-#
        qv_global = torch.sigmoid(qz_global)
        
        # log(1 - v) is exactly -softplus(z_global)
        log_one_minus_v = -F.softplus(qz_global)
        
        # cumprod in log-space is just cumsum
        log_cumprod_one_minus_v = torch.cumsum(log_one_minus_v, dim=-1)
        
        # Exponentiate back to linear space only at the very end
        cumprod_one_minus_v = torch.exp(log_cumprod_one_minus_v)

        pad = torch.ones_like(cumprod_one_minus_v[..., :1])
        previous_remaining = torch.cat([pad, cumprod_one_minus_v[..., :-1]], dim=-1)

        beta_k = qv_global * previous_remaining
        beta_last = cumprod_one_minus_v[..., -1:]
        beta = torch.cat([beta_k, beta_last], dim=-1)
#changes for logistic normal to sb 

import torch
import torch.nn as nn

class KumaraswamyStickBreaking(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def kumaraswamy_rsample(self, a, b):
        """
        Samples from the Kumaraswamy distribution using the inverse CDF method.
        a, b: Parameters of the distribution, outputted by the encoder.
        """
        # Draw uniform noise
        u = torch.rand_like(a)
        u = torch.clamp(u, self.eps, 1.0 - self.eps)
        
        # Inverse CDF reparameterization trick
        v = (1.0 - (1.0 - u).pow(1.0 / b)).pow(1.0 / a)
        
        # Clamp to prevent strictly 0 or 1 values for log stability
        return torch.clamp(v, self.eps, 1.0 - self.eps)

    def forward(self, a, b):
        """
        Takes parameters a and b of shape (batch_size, K-1)
        Returns the simplex vector pi of shape (batch_size, K)
        """
        # 1. Sample the breaking fractions
        v = self.kumaraswamy_rsample(a, b)
        
        # 2. Compute log(v) and log(1-v)
        log_v = torch.log(v)
        log_1_minus_v = torch.log(1.0 - v)
        
        # 3. Compute the log of the remaining stick length
        # We prepend a zero because the first stick has no prior breaks
        pad = torch.zeros_like(log_1_minus_v[..., :1])
        log_remaining = torch.cat([pad, log_1_minus_v], dim=-1).cumsum(dim=-1)
        
        # 4. Compute log probabilities for each component
        # The final stick portion uses the remaining stick, so we append log(1.0) = 0 to log_v
        log_v_extended = torch.cat([log_v, torch.zeros_like(log_v[..., :1])], dim=-1)
        log_pi = log_v_extended + log_remaining
        
        # Return to linear space on the simplex
        return torch.exp(log_pi)

# Example usage:
# Assuming an encoder outputs `a` and `b` for K-1 components
batch_size = 32
K = 10 

# Dummy outputs from an encoder network (must be strictly positive, e.g., via Softplus)
a = torch.rand(batch_size, K-1) + 0.1 
b = torch.rand(batch_size, K-1) + 0.1

sbp_layer = KumaraswamyStickBreaking()
pi = sbp_layer(a, b)

# pi is now shape (32, 10), sums to 1 across dim=-1, and is fully differentiable

#--- Dependencies---#
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---Class Definition: Hierarchical Dirichlet Process Clustering Algorithm---#
class HDPContextualiser(BaseEstimator, TransformerMixin):
    def __init__(self, init_components=5, step_size=2, max_scan=50, gamma=0.1, alpha=1.0, coverage_limit=0.95, interaction_type='product', random_state=42):
        self.init_components = init_components
        self.step_size = step_size
        self.max_scan = max_scan
        self.gamma = gamma #---Global Concentration Param---#
        self.alpha = alpha #---Local Concentration Param---#
        self.coverage_limit = coverage_limit
        self.interaction_type = interaction_type
        self.random_state = random_state
        self.model_ = None
        self.active_components_ = 0
        self.global_stick_order_ = None
    
    def fit(self, X, y=None):
        kt = self.init_components
        logger.info(f"Starting Hierarchical DP Scan. Init: {kt}, Limit: {self.max_scan}")
        while kt <= self.max_scan:
            model = BayesianGaussianMixture(
                n_components=kt,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.gamma,
                covariance_type='diag',
                random_state=self.random_state,
                n_init=1
            )
            model.fit(X)
            weights = model.weights_
            sort_indices = np.argsort(weights)[::-1]
            sorted_weights = weights[sort_indices]
            cumulative_weights = np.cumsum(sorted_weights)
            coverage_idx = np.searchsorted(cumulative_weights, self.coverage_limit)
            logger.debug(f"Scan K={kt}; Density Explained={(cumulative_weights[coverage_idx]):.2f}")
            if coverage_idx < kt - 1:
                self.model_ = model
                self.active_components_ = coverage_idx + 1
                self.global_stick_order_ = sort_indices
                logger.info(f"Converged! ~ {self.active_components_} latent groups explain {self.coverage_limit*100}% of density")
                return self
            kt += self.step_size
        
        logger.warning(f"HDP failed to converge within max scan ({self.max_scan}). Defaulting to last model with K={kt}.")
        self.model_ = model
        self.active_components_ = kt
        self.global_stick_order_ = np.argsort(model.weights_)[::-1]
        return self
    
    def _joint_probs_to_dirichlet_sticks(self, probs):
        """
        Converts posteriors into stick-breaking weights using Bayesian posterior expectation
        """
        p_sorted = probs[:, self.global_stick_order_]
        sticks = np.zeros_like(p_sorted)
        for k in range(self.active_components_):
            n_k = p_sorted[:, k] #--Mass in cluster k--#
            if k < p_sorted.shape[1] - 1:
                n_gt_k = p_sorted[:, (k+1):].sum(axis=1) #--Mass in  subsequent tail clusters--#
            else:
                n_gt_k = np.zeros_like(n_k)
            numerator = n_k
            denom = n_k + n_gt_k + self.alpha
            vk = np.clip(numerator / denom, 0, 1)
            sticks[:, k] = vk
        return sticks
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            logger.warning("Input X is NOT a pandas DataFrame. Column names will default to integer index.")
            X = pd.DataFrame(X)
        probs = self.model_.predict_proba(X)
        sticks = self._joint_probs_to_dirichlet_sticks(probs)
        generated_features = []
        for k in range(self.active_components_):
            cluster_idx = self.global_stick_order_[k]
            cluster_name = f"Stick_{k}_(Global_Cluster_{cluster_idx})"
            v_vec = sticks[:, k]
            if self.interaction_type == 'product':
                weighted = X.multiply(v_vec, axis=0)
                weighted.columns = [f"{col}_{cluster_name}" for col in X.columns]
                generated_features.append(weighted)
            elif self.interaction_type == 'append_only':
                factor_series = pd.Series(v_vec, index=X.index, name=cluster_name)
                generated_features.append(factor_series)
        if not generated_features:
            logger.warning("No features generated during transform -- No active components")
            return X
        X_c = pd.concat(generated_features, axis=1)
        return pd.concat([X, X_c], axis=1)
