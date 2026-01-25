#--Dependencies--#
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

#--Class: Feature Engineering Pipeline--#
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.scaler_ = None
        self._eps = self.config.features.scaling.eps
        self._base_term_names = [item[1] for item in self.config.features.transforms]
        self._interaction_bases = [item[0] for item in self.config.features.transforms]
        self.interaction_names_ = [item[0] for item in self.config.features.interactions]

    def _apply_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log/log1p identities defined in YAML."""
        df = X.copy()
        for src, (func, feat) in self.config.features.transforms.items():
            if func == "log1p":
                df[feat] = np.log1p(df[src])
            elif func == "log":
                df[feat] = np.log(df[src] + self._eps)
            else:
                df[feat] = df[src]
        return df

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives multi-order interaction terms"""
        for feat, components in self.config.features.interactions:
            df[feat] = df[components].prod(axis=1)
        return df

    def fit(self, X, y=None):
        """Learns scaling parameters using sklearn scalers"""
        #--Orchestration--#
        df_with_base_terms = self._apply_transforms(X)
        df_with_interaction_terms = self._create_interactions(df_with_base_terms)
        
        #--Scaler as per config---#
        method = self.config.features.scaling.method
        if method == "power":
            self.scaler_ = PowerTransformer(method='yeo-johnson')
        elif method == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = StandardScaler()
        
        #-fit on newly engineered terms under "_create_interactions_()"
        if self.interaction_names_:
            self.scaler_.fit(df_with_interaction_terms[self.interaction_names_])

        return self

    def transform(self, X):
        """Orchestration module"""
        df = self._apply_transforms(X)
        df = self._create_interactions(df)
        
        if self.scaler_ is not None and self.interaction_names_:
            df[self.interaction_names_] = self.scaler_.transform(df[self.interaction_names_])
            
        return df