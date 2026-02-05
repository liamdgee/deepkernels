# filename: dirichlet_clusters.py

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
