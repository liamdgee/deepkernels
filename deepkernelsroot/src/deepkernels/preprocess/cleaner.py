#---Dependencies---#
import sklearn
sklearn.set_config(transform_output="pandas")

import os
import logging
import numpy as np
import pandas as pd
import warnings
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal, Annotated
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--Config--#
class CleanerConfig(BaseModel):
    missingness_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85
    impute_strategy: Literal['mean', 'median', 'mode', 'zero'] = 'mean'
    categorical_threshold: Annotated[float, Field(ge=0, le=1)] = 0.025
    default_to_numeric: Optional[List[str]] = [
    "black_s_pct", "black_g_pct", "black_fs_pct", "black_bifsg_pct", "black_sg_pct",
    "share_pop_black", "share_black_pop_geba",
    "shr_loan_black_final_race", "shr_loan_black_sg_cont", 
    "shr_loan_white_final_race", "shr_loan_white_sg_cont",
    "shr_app_black_sg_cont", "shr_app_white_sg_cont",
    "total_percap_inc", "amountsought",
    "dissim_scaled", "isolation_scaled", "animus_scaled", 
    "iat_score_f_scaled", "mdi"
    ]
    override_to_numeric_cols: Optional[List[str]] = None

    default_id_cols: List[str] = ['unique_borrower', 'lender_clean', 'time']
    override_id_cols: Optional[List[str]] = None

    @property
    def active_id_cols(self) -> List[str]:
        return self.override_id_cols if self.override_id_cols is not None else self.default_id_cols
    
    @property
    def active_numeric_cols(self) -> List[str]:
        return self.override_to_numeric_cols if self.override_to_numeric_cols is not None else self.default_to_numeric



#---Data Cleaning Pipeline Class---#
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            config: Optional[CleanerConfig] = None, 
            id_cols: Optional[List[str]] = None,
            to_numeric: Optional[List[str]] = None, 
            missingness_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85, 
            impute_strategy: Literal['mean', 'median', 'mode', 'zero'] = 'mean', 
            categorical_threshold: Annotated[float, Field(ge=0, le=1)] = 0.025, 
            **kwargs
        ):

        self.config = config or CleanerConfig()
        self.missingness_threshold = missingness_threshold or self.config.missingness_threshold
        self.impute_strategy = impute_strategy or self.config.impute_strategy
        self.categorical_threshold = categorical_threshold or self.config.categorical_threshold
        self.default_to_numeric = [
            "black_s_pct", "black_g_pct", "black_fs_pct", "black_bifsg_pct", "black_sg_pct",
            "share_pop_black", "share_black_pop_geba",
            "shr_loan_black_final_race", "shr_loan_black_sg_cont", 
            "shr_loan_white_final_race", "shr_loan_white_sg_cont",
            "shr_app_black_sg_cont", "shr_app_white_sg_cont",
            "total_percap_inc", "amountsought",
            "dissim_scaled", "isolation_scaled", "animus_scaled", 
            "iat_score_f_scaled", "mdi"
        ]
        
        self.to_numeric = to_numeric if to_numeric is not None else self.default_to_numeric #-for robust testing-#

        self.feature_names_out_ = None
        self.dtype_map_ = None
        self.keep_cols_ = None

        self.id_cols = id_cols if id_cols is not None else self.config.active_id_cols

        self.processor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), self.to_numeric),
                ('keep_ids', 'passthrough', self.id_cols)
            ]
        )
        self.processor.set_output(transform="pandas")
    
    def _assign_time_index(self, df: pd.DataFrame, target_idx: str = 'lender_clean'):
        """
        Unique helper function to assign a time index to specific bisg datasets being used.
        """
        if target_idx not in df.columns:
            warnings.warn(
                f"Column '{target_idx}' not found for sorting. "
                "Defaulting 'time' index to original row order.",
                UserWarning
            )
            df['time'] = range(len(df))
            return df
        sort_key = df[target_idx].astype(str).str.lower().str.strip()
        df['time'] = sort_key.rank(method='first').astype(int)
        df = df.sort_values('time').reset_index(drop=True)

        return df

    def clean(self, df, fit=True):
        """
        Args:
            fit (bool): If True, relearn scaling stats (Training). 
                        If False, use existing stats (Inference).
        """
        df = self._assign_time_index(df)
        
        if fit:
            return self.processor.fit_transform(df)
        else:
            return self.processor.transform(df)
    
    def fit(self, X: pd.DataFrame, y=None):
        X_norm = self._canonicalise_headers(X.copy())
        #--Normalise Numeric cols--#
        for c in self.to_numeric:
            if c in X_norm.columns:
                X_norm[c] = pd.to_numeric(X_norm[c],errors='coerce')
        mask = X_norm.isna().mean() < self.missingness_threshold
        self.keep_cols_ = X_norm.columns[mask].tolist()
        num_cols = X_norm[self.keep_cols_].select_dtypes(include=[np.number]).columns
        if self.impute_strategy == 'mean':
            self.impute_vals_ = X_norm[num_cols].mean().to_dict()
        elif self.impute_strategy == 'median':
            self.impute_vals_ = X_norm[num_cols].median().to_dict()
        elif self.impute_strategy == 'mode':
            self.impute_vals_ = X_norm[num_cols].mode().iloc[0].to_dict()
        elif self.impute_strategy == 'zero':
            logger.warning("All missing values will be replaced with 0")
            self.impute_vals_ = {col: 0 for col in num_cols}
        else:
            raise ValueError(f"impute strategy must be set to 'mean', 'median', 'mode' or 'zero'. No valid strategy was received-- Current strategy: {self.config.impute_strategy}")
        
        obj_cols = X_norm[self.keep_cols_].select_dtypes(include=['object']).columns
        candidate_drop_cols = [col for col in obj_cols if col not in self.id_cols]
        for col in candidate_drop_cols:
            unique_ratio = X_norm[col].nunique() / len(X_norm)
            if unique_ratio > self.categorical_threshold:
                logger.info(f"Dropping high-cardinality col: {col} (Ratio: {unique_ratio:.4f})")
                self.keep_cols_.remove(col)
    
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ['keep_cols_', 'impute_vals_'])
        df = self._canonicalise_headers(X.copy())

        #-to numeric safeguard-#
        for c in self.to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        #--Reindex cols from fit--#
        df = df.reindex(columns=self.keep_cols_)


        #--Normalise non-numeric cols--#
        obj_cols = df.select_dtypes(include=['object']).columns
        cols_to_clean = [col for col in obj_cols if col not in self.id_cols]
        if cols_to_clean:
            df[cols_to_clean] = df[cols_to_clean].apply(lambda x: x.str.strip().str.lower())
            df[cols_to_clean] = df[cols_to_clean].replace(
                ["na", "n/a", "unknown", "nan", "null", "none", ""], np.nan
            )
    
        #--Handle Missingness in numeric cols--#
        for k, v in self.impute_vals_.items():
            if k in df.columns:
                df[k] = df[k].fillna(v)
        
        self.feature_names_out_ = df.columns.tolist()
        
        return df
    
    def _canonicalise_headers(self, df:pd.DataFrame) -> pd.DataFrame:
        """Seperate for column consistency throughout class"""
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        return df
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_out_')
        return np.array(self.feature_names_out_)