#---Dependencies--#
import sklearn
sklearn.set_config(transform_output="pandas")

import warnings
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Literal, Annotated, Union
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
import re

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#-helper fn-#
class Collector:
    def collect(self, *dfs: pd.DataFrame) -> list[pd.DataFrame]:
        """prepares data for input to harmoniser"""
        return [df for df in dfs if not df.empty]

#--- Pydantic Config Model---# -- to be updated to reflect pydantic configs--
class HarmoniserConfig(BaseModel):
    """
    Strict Configuration for cleaning pipeline.
    Validates parameters using Pydantic as a pre pre-processing step.
    """
    threshold_for_missingness: Annotated[float, Field(le=1.0, ge=0.0)] = 0.92
    numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median'
    mode: Literal['union', 'intersection'] = 'union'
    num_cols = ['amountsought', 'animus_scaled', 'black_bifsg_pct', 'black_fs_pct', 'black_g_pct', 'black_s_pct', 'black_sg_pct', 'dissim_scaled', 'iat_score_f_scaled', 'isolation_scaled', 'ln_tenure', 'log_amountsought', 'num_apps', 'num_loans', 'share_black_pop_geba', 'share_pop_black', 'total_percap_inc']
    cat_cols = ['bank', 'cdfi', 'creditunion', 'fintech',  'mdi', 'factoringccmca']
    id_cols = ['lender_clean', 'time', 'unique_borrower']
    y_target = ['lmean_rejected']

#--- Schema Harmoniser Class ---#
class SchemaHarmoniser(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 config: Optional[HarmoniserConfig] = None, 
                 id_cols: Optional[List[str]] = None,
                 num_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 y_target: Optional[Union[str, List[str]]] = None,
                 numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median',
                 mode: Literal['union', 'intersection'] = 'union',
                 threshold_for_missingness: float = 0.92,
                 **kwargs
        ):

        self.config = config or HarmoniserConfig()

        self.numeric_strategy = numeric_strategy or self.config.numeric_strategy
        self.threshold_for_missingness = threshold_for_missingness if threshold_for_missingness is not None else self.config.threshold_for_missingness
        self.mode = mode if mode else self.config.mode

        self.id_cols = id_cols if id_cols is not None else self.config.id_cols
        self.num_cols = num_cols if num_cols is not None else self.config.num_cols
        self.cat_cols = cat_cols if cat_cols is not None else self.config.cat_cols
        self.y_target = y_target if y_target is not None else self.config.y_target

        #-states-#
        self.feature_names_out_ = None
        self.target_schema_ = None
        self.impute_values_ = {}
    
    def fit(self, dfs_in: List[pd.DataFrame], y=None):
        """Learns master schema across multiple input datasets"""
        dfs = dfs_in.copy()
        #-normalise headers and assign time index-#
        dfs = [self._canonicalise_headers(df.copy()) for df in dfs]
        dfs = [self._assign_time_index(df) for df in dfs]

        #-determine base schema-#
        all_cols = [set(df.columns) for df in dfs]
        if self.mode == 'intersection':
            base_schema = sorted(list(set.intersection(*all_cols)))
        else:
            base_schema = sorted(list(set.union(*all_cols)))
        
        #-calculate missingness-#
        total_rows = 0
        null_count = {col: 0 for col in base_schema}
        
        #-iterative missingness check --vectorised logic upgrade-#
        for df in dfs:
            nrows = len(df)
            total_rows += nrows
            missing = df.isnull().sum()
            for col in base_schema:
                if col in df.columns:
                    null_count[col] += missing[col]
                else:
                    null_count[col] += nrows
        
        schema_out = []
        if total_rows > 0:
            pct_null = {k: v / total_rows for k, v in null_count.items()}

            for keepcols in base_schema:
                if keepcols in self.id_cols or keepcols == 'src_idx':
                    schema_out.append(keepcols)
                    continue

                if pct_null[keepcols] <= self.threshold_for_missingness:
                    schema_out.append(keepcols)
        else:
            schema_out = base_schema.copy()
        
        self.target_schema_ = schema_out

        survivors = [df.reindex(columns=self.target_schema_) for df in dfs]
        valid_survivors = [df for df in survivors if not df.empty and not df.isna().all().all()]
        
        if valid_survivors:
            temp_master = pd.concat(valid_survivors, ignore_index=True)
            num_candidates = temp_master.select_dtypes(include=[np.number]).columns
            num_cols_for_impute = [c for c in num_candidates if c not in self.id_cols and c != 'src_idx']

            if self.numeric_strategy == 'median':
                self.impute_values_ = temp_master[num_cols_for_impute].median().to_dict()
            elif self.numeric_strategy == 'mean':
                self.impute_values_ = temp_master[num_cols_for_impute].mean().to_dict()
            elif self.numeric_strategy == 'mode':
                modes = temp_master[num_cols_for_impute].mode()
                self.impute_values_ = modes.iloc[0].to_dict() if not modes.empty else {}
            else:
                logger.warning("All missing values will be replaced with 0")
                self.impute_values_ = {col: 0 for col in num_cols_for_impute}
        
        self.feature_names_out_ = self.target_schema_
        
        return self

    def transform(self, dfs_in: Union[List[pd.DataFrame], pd.DataFrame], y=None) -> pd.DataFrame:
        """Aligns, tags and merges input datasets into a master data frame"""
        check_is_fitted(self, ['target_schema_', 'impute_values_'])

        if isinstance(dfs_in, pd.DataFrame):
            dfs = [dfs_in]
        else:
            dfs = dfs_in.copy()
        
        processed = []
        
        for tag, df in enumerate(dfs):
            temp_df = self._canonicalise_headers(df.copy())
            temp_df = self._assign_time_index(temp_df)
            temp_df['src_idx'] = tag
            temp_df = temp_df.reindex(columns=self.target_schema_ )

            if self.impute_values_:
                temp_df = temp_df.fillna(self.impute_values_)
            
            processed.append(temp_df)
        
        valid_processed = [df for df in processed if not df.empty and not df.isna().all().all()]
        
        if valid_processed:
            master = pd.concat(valid_processed, ignore_index=True)
            if 'src_idx' in master.columns:
                 master['src_idx'] = master['src_idx'].fillna(-1).astype(int)
            return master
        else:
            logger.warning("Returned empty DataFrame in transform")
            return pd.DataFrame(columns=self.target_schema_)
    
    def _canonicalise_headers(self, df:pd.DataFrame) -> pd.DataFrame:
        """Seperate for column consistency throughout class"""
        df.columns = [self._clean_str(col) for col in df.columns]
        return df
    
    def _clean_str(self, s: str) -> str:
        """Helper to ensure ID strings match canonical format"""
        if not isinstance(s, str): return str(s)
        return re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', s.strip().lower()))
    
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
        
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'target_schema_')
        return np.array(self.target_schema_)


