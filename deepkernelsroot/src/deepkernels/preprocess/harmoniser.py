#---Dependencies--#
import sklearn
sklearn.set_config(transform_output="pandas")

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Literal, Annotated, Union
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
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
    threshold_for_missingness: Annotated[float, Field(le=0.995, ge=0.01)] = 0.92
    numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median'
    mode: Literal['union', 'intersection'] = 'union'
    default_id_cols: List[str] = ['unique_borrower', 'lender_clean', 'time']
    override_id_cols: Optional[List[str]] = None

#--- Schema Harmoniser Class ---#
class SchemaHarmoniser(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 config: Optional[HarmoniserConfig] = None, 
                 id_cols: Optional[List[str]] = None,
                 numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median',
                 mode: Literal['union', 'intersection'] = 'union',
                 threshold_for_missingness: float = 0.92,
                 **kwargs
        ):
        self.config = config if config else HarmoniserConfig()
        self.numeric_strategy = numeric_strategy
        self.threshold_for_missingness = threshold_for_missingness or self.config.threshold_for_missingness
        self.mode = mode or self.config.mode or 'union'
        id_cols_in = id_cols if id_cols is not None else ['lender_clean', 'time', 'unique_borrower']
        self.id_cols = [self._clean_str(strings) for strings in id_cols_in]

        #-states-#
        self.feature_names_out_ = None
        self.target_schema_ = None
        self.impute_values_ = {}
        self.numeric_cols_ = []
        self.processor_ = None
    
    def fit(self, dfs_in: List[pd.DataFrame], y=None):
        """Learns master schema across multiple input datasets"""
        dfs = [self._canonicalise_headers(df.copy()) for df in dfs_in]
        all_cols = [set(df.columns) for df in dfs]

        if self.mode == 'intersection':
            base_schema = sorted(list(set.intersection(*all_cols)))
        else:
            base_schema = sorted(list(set.union(*all_cols)))
        
        total_rows = 0
        null_count = {col: 0 for col in base_schema}
        
        #-iterative missingness check --vectorised logic upgrade-#
        for df in dfs:
            nrows = len(df)
            total_rows += nrows
            nulls = df.isnull().sum()
            for col in base_schema:
                if col in nulls.index:
                    null_count[col] += nulls[col]
                else:
                    null_count[col] += nrows
        
        if total_rows == 0:
            pct_null = {col: 1.0 for col in base_schema}
        else:
            pct_null = {k: v / total_rows for k, v in null_count.items()}
        
        self.target_schema_ = [k for k in base_schema if pct_null[k] <= self.threshold_for_missingness]

        survivors = [df.reindex(columns=self.target_schema_) for df in dfs]
        temp_master = pd.concat(survivors, ignore_index=True)

        all_num_cols = temp_master.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols_ = [col for col in all_num_cols if col not in self.id_cols]

        if self.numeric_strategy == 'median':
            self.impute_values_ = temp_master[self.numeric_cols_].median().to_dict()
        elif self.numeric_strategy == 'mean':
            self.impute_values_ = temp_master[self.numeric_cols_].mean().to_dict()
        elif self.numeric_strategy == 'mode':
            modes = temp_master[self.numeric_cols_].mode()
            self.impute_values_ = modes.iloc[0].to_dict() if not modes.empty else {}
        elif self.numeric_strategy == 'zero':
            logger.warning("All missing values will be replaced with 0")
            self.impute_values_ = {col: 0 for col in self.numeric_cols_}
        else:
            raise ValueError(f"numeric strategy must be set to 'mean', 'median', 'mode' or 'zero'. No valid strategy was received-- Current strategy: {self.numeric_strategy}")
        
        if 'src_idx' not in temp_master.columns:
            temp_master['src_idx'] = 0
        
        self.ids_verified_ = [col for col in (self.id_cols + ['src_idx']) if col in temp_master.columns]
        
        self.processor_ = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), self.numeric_cols_),
                ('keep_ids', 'passthrough', self.ids_verified_)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

        data_to_fit = temp_master.fillna(self.impute_values_)

        self.processor_.fit(data_to_fit)

        self.feature_names_out_ = self.target_schema_
        
        return self
    
    def transform(self, dfs_in: Union[List[pd.DataFrame], pd.DataFrame], y=None) -> pd.DataFrame:
        """Aligns, tags and merges input datasets into a master data frame"""
        check_is_fitted(self, ['target_schema_', 'impute_values_', 'processor_'])
        
        if isinstance(dfs_in, pd.DataFrame):
            dfs_in = [dfs_in]
        
        processed = []
        
        for tag, df in enumerate(dfs_in):
            temp_df = self._canonicalise_headers(df.copy())

            temp_df = temp_df.reindex(columns=self.target_schema_ )

            if self.impute_values_:
                temp_df.fillna(self.impute_values_, inplace=True)
            
            temp_df['src_idx'] = tag
            processed.append(temp_df)
        
        master = pd.concat(processed, ignore_index=True)
        
        return self.processor_.transform(master)
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'processor_')
        return self.processor_.get_feature_names_out()
    
    def _canonicalise_headers(self, df:pd.DataFrame) -> pd.DataFrame:
        """Seperate for column consistency throughout class"""
        df.columns = [self._clean_str(col) for col in df.columns]
        return df
    
    def _clean_str(self, s: str) -> str:
        """Helper to ensure ID strings match canonical format"""
        if not isinstance(s, str): return str(s)
        return re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', s.strip().lower()))
        


