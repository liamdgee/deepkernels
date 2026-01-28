#filename: lasso_features.py

#-Dependencies-
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from scipy.stats import spearmanr
import logging
from pydantic import BaseModel, Field
from sklearn.utils.validation import check_is_fitted

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngConfig(BaseModel):
    spearman_corr_threshold: float = Field(0.9, ge=0, le=1.0)
    lasso_cv: int = Field(5, gt=1)
    lasso_max_samples: int = Field(24000, gt=1, le=24998)
    vif_threshold: float = Field(8.5, gt=1.0, le=10.0)
    interaction_only: bool = True
    random_state: int = 42

class LassoFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeatureEngConfig):
        self.config = config
        self.scaler_ = RobustScaler()

        self.shipped_features_ = None
        self.selected_features_ = None
    
    def _vif_prune(self, X: pd.DataFrame) -> pd.DataFrame:
        Xv = X.copy()

        #--Short circuit--#

        if 'const' not in Xv.columns:
            Xv.insert(0, 'const', 1)
        
        while True:
            cols = Xv.columns
            vals = Xv.values
            ncol = vals.shape[1]

            if ncol <= 2:
                break

            vifs = []

            for idx in range(1, ncol):
                #--Target (current col)
                y = vals[:, idx]
                
                #-Predict using remaining columns--#
                mask = np.ones(ncol, dtype=bool)
                mask[idx] = False
                Xr = vals[:, mask] #--r for remaining cols--#

                #--Vectorised OLS--#
                coef, *diagnostics = np.linalg.lstsq(Xr, y, rcond=None)

                rss = np.sum((y - Xr @ coef) ** 2)
                tss = np.sum((y - np.mean(y)) **2)
                tss = np.clip(tss, a_min=1e-8, a_max=None)

                rsquared = 1 - (rss / tss)
                vif = 1 / (1 - rsquared)
                vifs.append(vif)
            
            vif_max = max(vifs)
            if vif_max > self.config.vif_threshold:
                idx_max = vifs.index(vif_max) + 1
                drop_cols_ = cols[idx_max]

                logger.info(f"-- Pruning {drop_cols_} cols -- variance inflation factor (max): {vif_max:.2f}")

                Xv = Xv.drop(columns=[drop_cols_])
            else:
                break

        return Xv.drop(columns=['const'])

    def _spearman_thinner(self, X):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > self.config.spearman_corr_threshold)]
        kept_cols = [c for c in X_df.columns if c not in drop_cols]
        return X_df[kept_cols]
    
    def _lasso_selector(self, X, y):
        X_df = pd.DataFrame(X)
        X_scld = pd.DataFrame(self.scaler_.fit_transform(X_df), columns=X_df.columns, index=X_df.index)
        if len(X_df) > self.config.lasso_max_samples:
            sample_idx = np.random.RandomState(self.config.random_state).choice(len(X_df), self.config.lasso_max_samples, replace=False)
            X_train = X_scld.loc[sample_idx]
            y_train = y.loc[sample_idx]
        else:
            X_train = X_scld
            y_train = y
        lasso = LassoCV(cv=self.config.lasso_cv, random_state=self.config.random_state, n_jobs=-1)
        lasso.fit(X_train, y_train)
        coef_series = pd.Series(lasso.coef_, index=X_df.columns)
        selected_features = coef_series[coef_series != 0].index.tolist()
        return X_df[selected_features]
    
    def _create_interactions(self, X):
        X_df = pd.DataFrame(X)
        poly = PolynomialFeatures(degree=2, interaction_only=self.config.interaction_only, include_bias=False)
        X_poly = poly.fit_transform(X_df)
        shipped_features_ = poly.get_feature_names_out(X_df.columns)
        X_poly_df = pd.DataFrame(X_poly, columns=shipped_features_, index=X_df.index)
        return X_poly_df
    
    def fit(self, X, y):
        """Sequential Selection: Spearman -> Lasso -> interaction_terms -> vif_factor"""
        X_df = pd.DataFrame(X).copy() 
        y_series = pd.Series(y).copy()
        
        #--Spearman thinner to filter collinear input cols--#
        X_thin = self._spearman_thinner(X_df)
        
        # 2. Lasso Selection (Sparsity filter)
        X_lasso = self._lasso_selector(X_thin, y_series)
        self.selected_features_ = X_lasso.columns.tolist()

        if not self.selected_features_:
            logger.warning("Lasso selected 0 features -- shipping empty features")
            self.shipped_features_ = []
            return self
        
        #--Create Interaction Terms--#
        X_interactions = self._create_interactions(X_lasso)
        
        # 4. VIF Prune (Multicollinearity filter for interactions)
        X_final = self._vif_prune(X_interactions)
        
        # 5. THE LOCK: Record exactly which features survived the gauntlet
        self.shipped_features_ = X_final.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies learned feature transform"""

        check_is_fitted(self, ['shipped_features_', 'selected_features_'])
        
        X_df = pd.DataFrame(X)
        
        #--1st order interactions--#
        X_base = X_df[self.selected_features_]
        X_poly = self._create_interactions(X_base)
        
        #--Reindex--#
        return X_poly.reindex(columns=self.shipped_features_, fill_value=0)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'shipped_features_')
        return np.array(self.shipped_features_)