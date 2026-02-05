#---Dependencies--#
import sklearn
sklearn.set_config(transform_output="pandas")

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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
    threshold_for_missingness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.92
    numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median'
    mode: Literal['union', 'intersection'] = 'union'
    default_id_cols: List[str] = ['unique_borrower', 'lender_clean']
    override_id_cols: Optional[List[str]] = None

    @property
    def active_id_cols(self) -> List[str]:
        return self.override_id_cols if self.override_id_cols is not None else self.default_id_cols

#--- Schema Harmoniser Class ---#
class SchemaHarmoniser(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 config: Optional[HarmoniserConfig] = None, 
                 id_cols: Optional[List[str]] = None,
                 numeric_strategy: Literal['mean', 'median', 'zero', 'mode'] = 'median',
                 mode: Literal['union', 'intersection'] = 'union',
                 threshold_for_missingness: float = None,
                 **kwargs
        ):
        self.config = config if config else HarmoniserConfig()
        self.numeric_strategy = numeric_strategy
        self.threshold_for_missingness = threshold_for_missingness or self.config.threshold_for_missingness
        self.mode = mode if mode else self.config.mode

        self.feature_names_out_ = None
        self.target_schema_ = None
        self.impute_values_ = {}
        self.processor_ = None

        self.default_id_cols = ['unique_borrower', 'lender_clean']
        self.id_cols = id_cols if id_cols is not None else self.config.active_id_cols
    
    def fit(self, dfs_in: List[pd.DataFrame], y=None):
        """Learns master schema across multiple input datasets"""
        dfs = [self._canonicalise_headers(df.copy()) for df in dfs_in]
        all_cols = [set(df.columns) for df in dfs]

        if not hasattr(self, 'mode') or self.mode is None:
            self.mode = self.config.mode if self.config else 'union'

        if self.mode == 'intersection':
            schema = sorted(list(set.intersection(*all_cols)))
        else:
            schema = sorted(list(set.union(*all_cols)))
        
        concat_for_missingness = pd.concat(dfs)
        pct_null = concat_for_missingness.isnull().mean()
        self.target_schema_ = [c for c in schema if pct_null.get(c, 0) <= self.threshold_for_missingness]

        temp_master = concat_for_missingness.reindex(columns=self.target_schema_)
        num_cols = temp_master.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols_ = [c for c in num_cols if c not in self.id_cols]

        if self.numeric_strategy == 'median':
            self.impute_values_ = temp_master[num_cols].median().to_dict()
        elif self.numeric_strategy == 'mean':
            self.impute_values_ = temp_master[num_cols].mean().to_dict()
        elif self.numeric_strategy == 'mode':
            self.impute_values_ = temp_master[num_cols].mode().iloc[0].to_dict()
        elif self.numeric_strategy == 'zero':
            logger.warning("All missing values will be replaced with 0")
            self.impute_values_ = {col: 0 for col in num_cols}
        else:
            raise ValueError(f"numeric strategy must be set to 'mean', 'median', 'mode' or 'zero'. No valid strategy was received-- Current strategy: {self.numeric_strategy}")
        
        temp_master['src_idx'] = 0
        self.id_cols_with_src_idx_ = self.id_cols + ['src_idx']
        self.processor_ = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), self.numeric_cols_),
                ('keep_ids', 'passthrough', self.id_cols_with_src_idx_)
            ],
            remainder='drop'
        )

        self.processor_.fit(temp_master.fillna(self.impute_values_))

        self.feature_names_out_ = self.target_schema_
        return self
    
    def transform(self, dfs_in: List[pd.DataFrame], y=None) -> pd.DataFrame:
        """Aligns, tags and merges input datasets into a master data frame"""
        check_is_fitted(self, ['target_schema_', 'impute_values_', 'processor_'])
        
        processed = []
        
        for tag, df in enumerate(dfs_in):
            temp_df = self._canonicalise_headers(df.copy())

            temp_df = temp_df.reindex(columns=self.target_schema_ )
            
            temp_df['src_idx'] = tag

            processed.append(temp_df)
        
        master = pd.concat(processed, ignore_index=True)

        for k, v in self.impute_values_.items():
            if k in master.columns:
                master[k] = master[k].fillna(v)
        
        return self.processor_.transform(master)
    
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'processor_')
        return self.processor_.get_feature_names_out()
    
    def _canonicalise_headers(self, df:pd.DataFrame) -> pd.DataFrame:
        """Seperate for column consistency throughout class"""
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        return df
        


