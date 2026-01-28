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
    to_numeric: List[str] = Field(default_factory=list) #--List of cols we want to force to numeric--#


#---Data Cleaning Pipeline Class---#
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, config: CleanerConfig):
        self.config = config
        self.impute_vals = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.dtype_map_ = None
        self.keep_cols_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        X_norm = self._canonicalise_headers(X.copy())
        self.feature_names_in_ = X_norm.columns.tolist()
        mask = X_norm.isna().mean() < self.config.missingness_threshold
        self.keep_cols_ = X_norm.columns[mask].tolist()
        self.dtype_map_ = X_norm[self.keep_cols_].dtypes.to_dict()
        num_cols = X_norm[self.keep_cols_].select_dtypes(include=[np.number]).columns
        if self.config.impute_strategy == 'mean':
            self.impute_vals = X_norm[num_cols].mean().to_dict()
        elif self.config.impute_strategy == 'median':
            self.impute_vals = X_norm[num_cols].median().to_dict()
        elif self.config.impute_strategy == 'mode':
            self.impute_vals = X_norm[num_cols].mode().iloc[0].to_dict()
        elif self.config.impute_strategy == 'zero':
            logger.warning("All missing values will be replaced with 0")
            self.impute_vals = {col: 0 for col in num_cols}
        else:
            raise ValueError(f"impute strategy must be set to 'mean', 'median', 'mode' or 'zero'. No valid strategy was received-- Current strategy: {self.config.impute_strategy}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ['keep_cols', 'impute_vals'])
        df = self._canonicalise_headers(X.copy())

        #--Reindex cols from fit--#
        df = df.reindex(columns=self.keep_cols_)

        #--Schema Guard--#
        for c, dtype in self.dtype_map_.items():
            if c in df.columns:
                try:
                    df[c] = df[c].astype(dtype, errors='ignore')
                except Exception as e:
                    logger.warning(f"Could not enforce {dtype} on col: {c} -- {e}")

        #--Normalise non-numeric cols--#
        obj_cols = df.select_dtypes(include=['object']).columns
        if not obj_cols.empty:
            df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip().str.lower())
            df[obj_cols] = df[obj_cols].replace(
                ["na", "n/a", "unknown", "nan", "null", "none", ""], np.nan
            )
        
        #--Normalise Numeric cols--#
        for c in self.config.to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c],
                    errors='coerce'
                )
    
        #--Handle Missingness--#
        mask = df.isna().mean() < self.config.missingness_threshold
        df = df.loc[:, mask]

        #--Imputation learned in fit--#
        for k, v in self.impute_vals.items():
            if k in df.columns:
                df[k] = df[k].fillna(v)
        
        
        self.feature_names_out = self.keep_cols_
        
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
        return np.append(np.array(self.feature_names_out_))

        

            
        