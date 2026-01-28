#---Dependencies--#
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#--- Pydantic Config Model---# -- to be updated to reflect pydantic configs--
class HarmoniserConfig(BaseModel):
    """
    Strict Configuration for cleaning pipeline.
    Validates parameters using Pydantic as a pre pre-processing step.
    """
    threshold_for_missingness: float = Field(0.95, ge=0.0, le=1.0)
    id_cols: List[str] = Field(default_factory=list)
    numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median'
    mode: Literal['union', 'intersection'] = 'union'

#--- Schema Harmoniser Class ---#
class SchemaHarmoniser(BaseEstimator, TransformerMixin):
    def __init__(self, config: HarmoniserConfig):
        self.config = config
        self.feature_names_out_ = None
        self.target_schema_ = None
        self.impute_values_ = {}
    
    def fit(self, dfs_in: List[pd.DataFrame], y=None):
        """Learns master schema across multiple input datasets"""
        logger.info(f"Fitting Harmoniser on {len(dfs)} source datasets.")
        
        dfs = [self._canonicalise_headers(df.copy()) for df in dfs_in]

        #-- Lock target schema--#
        all_cols = [set(df.columns) for df in dfs]
        if self.config.mode == 'intersection':
            self.target_schema_ = sorted(list(set.intersection(*all_cols)))
        else:
            self.target_schema_ = sorted(list(set.union(*all_cols)))
        
        master_lock = pd.concat(dfs, join='outer', ignore_index=True)
        master_lock = master_lock.reindex(columns=self.target_schema_)

        num_cols = master_lock.select_dtypes(include=[np.number]).columns

        if self.config.numeric_strategy == 'median':
            self.impute_values_ = master_lock[num_cols].median().to_dict()
        elif self.config.numeric_strategy == 'mean':
            self.impute_values_ = master_lock[num_cols].mean().to_dict()
        elif self.config.numeric_strategy == 'mode':
            self.impute_values_ = master_lock[num_cols].mode().iloc[0].to_dict()
        elif self.config.numeric_strategy == 'zero':
            logger.warning("All missing values will be replaced with 0")
            self.impute_values_ = {col: 0 for col in num_cols}
        else:
            raise ValueError(f"numeric strategy must be set to 'mean', 'median', 'mode' or 'zero'. No valid strategy was received-- Current strategy: {self.config.numeric_strategy}")
            
        self.feature_names_out_ = self.target_schema_
        return self
    
    def transform(self, dfs: List[pd.DataFrame], y=None) -> pd.DataFrame:
        """Aligns, tags and merges input datasets into a master data frame"""
        processed = []

        check_is_fitted(self, ['target_schema_', 'impute_values_'])
        
        for tag, df in enumerate(dfs):
            temp_df = self._canonicalise_headers(df.copy())

            temp_df = temp_df.reindex(columns=self.target_schema_ )
            
            temp_df['src_idx'] = tag

            processed.append(temp_df)
        
        master = pd.concat(processed, ignore_index=True)

        for k, v in self.impute_values_.items():
            if k in master.columns:
                master[k] = master[k].fillna(v)
        
        return master
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_out_')
        return np.append(np.array(self.feature_names_out_), 'src_idx')
    
    def _canonicalise_headers(self, df:pd.DataFrame) -> pd.DataFrame:
        """Seperate for column consistency throughout class"""
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        return df
        


