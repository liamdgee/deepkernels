#---Dependencies---#
import os
import logging
import numpy as np
import pandas as pd
import warnings
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--Config--#
class CleanerConfig(BaseModel):
    missingness_threshold: float = Field(default=0.8, ge=0, le=1)
    impute_strategy: Literal['mean', 'median', 'mode', 'zero'] = 'mean'
    categorical_threshold: float = 0.025 #--1 in 40 values is unqiue--#
    to_numeric: Optional[List[str]] = [
    "black_s_pct", "black_g_pct", "black_fs_pct", "black_bifsg_pct", "black_sg_pct",
    "share_pop_black", "share_black_pop_geba",
    "shr_loan_black_final_race", "shr_loan_black_sg_cont", 
    "shr_loan_white_final_race", "shr_loan_white_sg_cont",
    "shr_app_black_sg_cont", "shr_app_white_sg_cont",
    "total_percap_inc", "amountsought",
    "dissim_scaled", "isolation_scaled", "animus_scaled", 
    "iat_score_f_scaled", "mdi"
    ]


#---Data Cleaning Pipeline Class---#
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, config: CleanerConfig, missingness_threshold: float = Field(default=0.8, ge=0.01, le=0.995), impute_strategy: Literal['mean', 'median', 'mode', 'zero'] = 'mean', categorical_threshold: float = Field(default=0.025, ge=0.001, le=0.9)):
        self.config = config if config is not None else CleanerConfig
        self.missingness_threshold = missingness_threshold or self.config.missingness_threshold
        self.impute_strategy = impute_strategy or self.config.impute_strategy
        self.categorical_threshold = categorical_threshold or self.config.categorical_threshold
        self.to_numeric = config.to_numeric or []
        self.impute_vals_ = {}
        self.feature_names_out_ = None
        self.dtype_map_ = None
        self.keep_cols_ = None
    
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
        for col in obj_cols:
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
        if not obj_cols.empty:
            df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip().str.lower())
            df[obj_cols] = df[obj_cols].replace(
                ["na", "n/a", "unknown", "nan", "null", "none", ""], np.nan
            )
    
        #--Handle Missingness--#
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