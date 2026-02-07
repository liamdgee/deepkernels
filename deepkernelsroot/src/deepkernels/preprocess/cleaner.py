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
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--Config--#
class CleanerConfig(BaseModel):
    missingness_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85
    impute_strategy: Literal['mean', 'median', 'mode', 'zero'] = 'mean'
    categorical_threshold: Annotated[float, Field(ge=0, le=1)] = 0.025
    num_cols: list[str] = ['amountsought', 'animus_scaled', 'black_bifsg_pct', 'black_fs_pct', 'black_g_pct', 'black_s_pct', 'black_sg_pct', 'dissim_scaled', 'iat_score_f_scaled', 'isolation_scaled', 'ln_tenure', 'log_amountsought', 'num_apps', 'num_loans', 'share_black_pop_geba', 'share_pop_black', 'total_percap_inc']
    cat_cols: list[str] = ['bank', 'cdfi', 'creditunion', 'fintech',  'mdi', 'factoringccmca']
    id_cols: list[str] = ['lender_clean', 'time', 'unique_borrower']
    y_target: list[str] = ['lmean_rejected']



#---Data Cleaning Pipeline Class---#
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            config: Optional[CleanerConfig] = None, 
            id_cols: Optional[List[str]] = None,
            num_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None,
            y_target: Optional[Union[str, List[str]]] = None,
            missingness_threshold: Annotated[float, Field(ge=0, le=1)] = 0.85, 
            impute_strategy: Literal['mean', 'median', 'mode', 'zero'] = 'mean', 
            categorical_threshold: Annotated[float, Field(ge=0, le=1)] = 0.025, 
            **kwargs
        ):

        self.config = config or CleanerConfig()
        self.missingness_threshold = missingness_threshold if missingness_threshold is not None else self.config.missingness_threshold
        self.impute_strategy = impute_strategy or self.config.impute_strategy
        self.y_target = y_target if y_target is not None else self.config.y_target
        self.id_cols = id_cols if id_cols is not None else self.config.id_cols
        self.categorical_threshold = categorical_threshold or self.config.categorical_threshold or 0.025
        self.num_cols = num_cols if num_cols is not None else self.config.num_cols
        self.cat_cols = cat_cols if cat_cols is not None else self.config.cat_cols

        self.feature_names_out_ = None
        self.impute_vals_ = {}
        self.keep_cols_ = []
    
    def _assign_time_index(self, df: pd.DataFrame):
        """
        Unique helper function to assign a time index to specific bisg datasets being used.
        """
        if 'time' in df.columns:
            return df
        target_idx = None
        if 'lender_clean' in df.columns:
            target_idx = 'lender_clean'
        elif 'unique_borrower' in df.columns:
            target_idx = 'unique_borrower'
        if target_idx is None:
            warnings.warn(
                "Neither 'lender_clean' nor 'unique_borrower' found. "
                "Defaulting 'time' index to original row order.",
                UserWarning
            )
            df['time'] = range(len(df))
        else:
            sort_key = df[target_idx].astype(str).str.lower().str.strip()
            df['time'] = sort_key.rank(method='first').astype(int)
        return df
    
    def fit(self, X: pd.DataFrame, y=None):
        df = self._canonicalise_headers(X.copy())
        df = self._assign_time_index(df)
        existing_nums = [c for c in self.num_cols if c in df.columns]
        for c in existing_nums:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        mask = df.isna().mean() < self.missingness_threshold
        initial_keep = df.columns[mask].tolist()
        current_cols = df[initial_keep]
        obj_cols = current_cols.select_dtypes(include=['object', 'category']).columns
        
        final_keep = []
        for col in initial_keep:
            if col in self.id_cols:
                final_keep.append(col)
                continue
            
            if col in obj_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > self.categorical_threshold:
                    logger.info(f"Dropping high-cardinality col: {col} (Ratio: {unique_ratio:.4f})")
                    continue
            
            final_keep.append(col)
            
        self.keep_cols_ = final_keep
        kept_nums = [c for c in existing_nums if c in final_keep]

        if self.impute_strategy == 'mean':
            self.impute_vals_ = df[kept_nums].mean().to_dict()
        elif self.impute_strategy == 'median':
            self.impute_vals_ = df[kept_nums].median().to_dict()
        elif self.impute_strategy == 'mode':
            modes = df[kept_nums].mode()
            self.impute_vals_ = modes.iloc[0].to_dict() if not modes.empty else {}
        else:
            logger.warning("All missing values will be replaced with 0")
            self.impute_vals_ = {col: 0 for col in kept_nums}
    
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ['keep_cols_', 'impute_vals_'])
        df = self._canonicalise_headers(X.copy())
        df = self._assign_time_index(df)
        df = df.reindex(columns=self.keep_cols_)
        existing_nums = [c for c in self.num_cols if c in df.columns]
        for c in existing_nums:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        obj_cols = df.select_dtypes(include=['object']).columns
        cols_to_clean = [c for c in obj_cols if c not in self.id_cols]
        
        if cols_to_clean:
            df[cols_to_clean] = df[cols_to_clean].apply(
                lambda x: x.str.strip().str.lower()
                if x.dtype == "object" else x
            )
            df[cols_to_clean] = df[cols_to_clean].replace(
                ["na", "n/a", "unknown", "nan", "null", "none", ""], np.nan
            )

        if self.impute_vals_:
            df = df.fillna(self.impute_vals_)
        df = df.fillna('missing')
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