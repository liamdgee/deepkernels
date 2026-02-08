#filename: lasso_features.py

#---Dependencies--#
import sklearn
sklearn.set_config(transform_output="pandas")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from scipy.stats import spearmanr
import logging
from pydantic import BaseModel, Field
from sklearn.utils.validation import check_is_fitted
from typing import Dict, List, Tuple, Union, Optional, TypeAlias, Literal, Annotated
from sklearn.model_selection import TimeSeriesSplit

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LassoConfig(BaseModel):
    spearman_corr_threshold: Annotated[float, Field(ge=0, le=1)] = 0.9
    lasso_cv: Annotated[int, Field(gt=1)] = 5
    lasso_max_samples: Annotated[int, Field(gt=1, le=24977)] = 24000 #-less than 25k for computational efficiency (24977 is an arbitrarily chosen numerical safeguard)
    vif_threshold: Annotated[float, Field(gt=1, le=17.0)] = 9.0
    interaction_only: bool = True
    random_state: int = 42
    power_scaler_override: bool = False
    ts_split_as_cv_strategy: bool = True
    cv_split: Annotated[int, Field(gt=1)] = 5

    #-core-#
    num_cols: list[str] = ['amountsought', 'animus_scaled', 'black_bifsg_pct', 'black_fs_pct', 'black_g_pct', 'black_s_pct', 'black_sg_pct', 'dissim_scaled', 'iat_score_f_scaled', 'isolation_scaled', 'ln_tenure', 'log_amountsought', 'num_apps', 'num_loans', 'share_black_pop_geba', 'share_pop_black', 'total_percap_inc']
    cat_cols: list[str] = ['bank', 'cdfi', 'creditunion', 'fintech',  'mdi', 'factoringccmca']
    id_cols: list[str] = ['lender_clean', 'time', 'unique_borrower']
    y_target: Union[list[str], str] = ['lmean_rejected']

class LassoFeatures(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            config: Optional[LassoConfig]=None,
            ts_split_as_cv_strategy: bool = True,
            cv_split: Optional[int] = 5, #-use for time evolving target variable-#
            lasso_max_samples: Annotated[int, Field(le=24977, gt=1)] = 24000, 
            vif_threshold: float = 9.0,
            interaction_only: bool = True, #-only computes polynomial interactions for engineered terms-#
            random_state: int = 42,
            spearman_corr_threshold: Annotated[float, Field(ge=0, le=1)] = 0.9,
            standard_scaler_override: bool = False,
            num_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None,
            id_cols: Optional[List[str]] = None,
            y_target: Optional[Union[List[str], str]] = None,
            **kwargs
        ):

        self.config = config if config else LassoConfig()
        self.ts_split_as_cv_strategy = ts_split_as_cv_strategy
        self.lasso_max_samples = lasso_max_samples or self.config.lasso_max_samples
        self.vif_threshold = vif_threshold or self.config.vif_threshold or 10.0
        self.random_state = random_state or self.config.random_state or 42
        self.interaction_only = interaction_only if interaction_only else True
        self.spearman_corr_threshold = spearman_corr_threshold or self.config.spearman_corr_threshold or 0.9
        
        self.scaler_ = StandardScaler() if standard_scaler_override else RobustScaler()

        self.num_cols = num_cols if num_cols is not None else self.config.num_cols
        self.cat_cols = cat_cols if cat_cols is not None else self.config.cat_cols
        self.id_cols = id_cols if id_cols is not None else self.config.id_cols
        self.y_target = y_target if y_target is not None else self.config.y_target
        

        raw_split = cv_split or self.config.cv_split
        if self.ts_split_as_cv_strategy:
            self.cv_obj_ = TimeSeriesSplit(n_splits=raw_split)
        else:
            self.cv_obj_ = raw_split
        
        self.poly_ = None
        self.shipped_features_ = None
        self.selected_lasso_features_ = None
    
    @staticmethod
    def sort_by_time(X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], time_col: Optional[str] = 'time') -> tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
        """
        run this first if needing to sort df by time for ts cv split!
         Logic:
            - If time_col exists, sort by it
            - If not, sort by index and log a warning (since this may not be ideal for time series data)
         This ensures that the temporal order of data is preserved for time series cross-validation strategies.
         If y is provided as a DataFrame, it will be reindexed to match the sorted X.
         Returns sorted X and y.
         """
        X_sorted = X.copy()
        time_col = time_col if time_col else 'time'

        if time_col:
            if time_col not in X_sorted.columns:
                logger.warning("No time column found in X")
            if not pd.api.types.is_datetime64_any_dtype(X_sorted[time_col]):
                logger.info(f"Converting '{time_col}' to datetime...")
                X_sorted[time_col] = pd.to_datetime(X_sorted[time_col])
            X_sorted = X_sorted.sort_values(by=time_col)
        else:
            X_sorted = X_sorted.sort_index()
            logger.warning("No time col found")
        
        if isinstance(y, pd.DataFrame):
            y_sorted = y.reindex(X_sorted.index)
        else:
            y_sorted = y.reindex(X_sorted.index)

        return X_sorted, y_sorted
    
    def _vif_prune(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Optimised VIF calculation using Matrix Inversion.
        
        Logic:
            The diagonal elements of the inverted correlation matrix are the VIFs.
            This allows us to calculate VIF for ALL features in one matrix operation,
            replacing the need to run N separate OLS regressions per iteration.
        """
        df_to_prune = X.copy()
        dropped = True
        while dropped:
            dropped = False
            cols = df_to_prune.columns
            if len(cols) < 2:
                break
            
            #-corr over cov because vif is scale invariant-#
            corr_mat = df_to_prune.corr(method='pearson').values
            
            try:
                corr_mat_inv = np.linalg.inv(corr_mat)
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix encountered in VIF. Dropping last column to resolve.")
                df_to_prune = df_to_prune.drop(columns=[cols[-1]])
                dropped = True
                continue
            
            #-extract vifs (diag values)-#
            vifs = np.diag(corr_mat_inv)
            vif_idx = np.argmax(vifs)
            max_vif_found = vifs[vif_idx]
            
            if max_vif_found > self.vif_threshold:
                drop_cols = cols[vif_idx]
                logger.info(f"Pruning {drop_cols} (VIF: {max_vif_found:.2f})")
                df_to_prune = df_to_prune.drop(columns=[drop_cols])
                dropped = True
                
        return df_to_prune

    def _spearman_thinner(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes highly correlated features before computationally heavy steps
        """
        corr_mat = X.corr(method='spearman').abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)) #-upper triangle select-#
        cols_dropped = [col for col in upper.columns if any(upper[col] > self.spearman_corr_threshold)]
        if cols_dropped:
            logger.info(f"Spearman Thinner dropping {len(cols_dropped)} cols: {cols_dropped}")
        return X.drop(columns=cols_dropped)
    
    def _lasso_selector(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Uses L1 regularization to select the most predictive base features
        Args:
            X: pandas data frame for feature selection and pruning
            y: pandas series / data frame of chosen target variable (sometimes identified as y_target)
        """
        if isinstance(y, pd.DataFrame): y = y.squeeze()
        if not isinstance(y, pd.Series): y = pd.Series(y, index=X.index)
        
        X = pd.DataFrame(X)
        if X.empty:
            logger.warning("Input to Lasso Selector is empty. Skipping selection.")
            return X

        X_scaled_vals = self.scaler_.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)

        if self.ts_split_as_cv_strategy:
            lasso = LassoCV(cv=self.cv_obj_, random_state=self.random_state, n_jobs=-1, max_iter=8000)
            lasso.fit(X_scaled, y)

        else:
            if len(X_scaled) > self.lasso_max_samples:
                sample_idx = np.random.RandomState(self.random_state).choice(len(X_scaled), self.lasso_max_samples, replace=False)
                X_train = X_scaled.iloc[sample_idx]
                y_train = y.iloc[sample_idx]
            else:
                X_train = X_scaled
                y_train = y
            
            lasso = LassoCV(cv=self.cv_obj_, random_state=self.random_state, n_jobs=-1, selection='random')
            lasso.fit(X_train, y_train)
        
        coef = pd.Series(lasso.coef_, index=X.columns)
        selected = coef[coef != 0].index.tolist()
        
        if not selected:
             logger.warning("Lasso dropped ALL features. Returning full original set as fallback.")
             return X

        return X[selected]
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame] = None):
        """Sequential Selection: Spearman -> Lasso -> interaction_terms -> vif_factor"""
        if y is None:
            target = self.y_target[0] if isinstance(self.y_target, list) else self.y_target
            if target in X.columns:
                y = X[target]
                X = X.drop(columns=[target])
            else:
                raise ValueError('Target y is None and cannot be found.')
        
        
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        df = X.copy()

        ignore = [col for col in df.columns if col in self.id_cols or col in self.cat_cols]
        X_features = df.drop(columns=ignore, errors='ignore')
        X_features = X_features.select_dtypes(include=[np.number])

        X_thin = self._spearman_thinner(X_features) #-filter collinear inputs-#
        X_lasso = self._lasso_selector(X_thin, y) #-filter sparsity-#
        self.selected_lasso_features_ = X_lasso.columns.tolist()

        if not self.selected_lasso_features_:
            self.shipped_features_ = []
            return self
        
        self.poly_ = PolynomialFeatures(
            degree=2,
            interaction_only=self.interaction_only,
            include_bias=False
        )

        polynomial_feature_vals = self.poly_.fit_transform(X_lasso)
        selected_poly_features = self.poly_.get_feature_names_out(X_lasso.columns)
        X_poly = pd.DataFrame(polynomial_feature_vals, columns=selected_poly_features, index=X_lasso.index)
        X_final = self._vif_prune(X_poly) #-vif prune on interactions only-#
        self.shipped_features_ = X_final.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies learned feature transform"""
        check_is_fitted(self, ['shipped_features_', 'selected_lasso_features_', 'poly_'])
        
        df = pd.DataFrame(X.copy())
        passthrough_cols = [id for id in df.columns if id in self.id_cols or id in self.cat_cols]
        passthrough = df[passthrough_cols].copy()
        
        missing_base = []
        for col in self.selected_lasso_features_:
            if col not in df.columns:
                df[col] = 0.0
                missing_base.append(col)
        
        X_base = df[self.selected_lasso_features_]

        polynomial_feature_vals = self.poly_.transform(X_base)
        selected_poly_features = self.poly_.get_feature_names_out(X_base.columns)
        X_poly = pd.DataFrame(polynomial_feature_vals, columns=selected_poly_features, index=X_base.index)

        X_selected = X_poly.reindex(columns=self.shipped_features_, fill_value=0)

        X_out = pd.concat([passthrough, X_selected], axis=1)

        return X_out
    

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'shipped_features_')
        return np.array(self.shipped_features_)