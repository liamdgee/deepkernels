from sklearn.pipeline import Pipeline
from pydantic import Field, BaseModel

from deepkernels.preprocess.cleaner import DataCleaner, CleanerConfig
from deepkernels.preprocess.harmoniser import Collector, HarmoniserConfig, SchemaHarmoniser
from deepkernels.preprocess.novelty import FeatureTransformer, NoveltyConfig

import pandas as pd
import logging
import torch
from typing import Union, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessConfig(BaseModel):
    clean: CleanerConfig = Field(default_factory=CleanerConfig)
    harmonise: HarmoniserConfig = Field(default_factory=HarmoniserConfig)
    feature: NoveltyConfig = Field(default_factory=NoveltyConfig)

class DataOrchestrator:
    def __init__(self, config=None):
        self.config = config if config is not None else ProcessConfig()
        self.cleaner_df1 = DataCleaner(config=self.config.clean)
        self.cleaner_df2 = DataCleaner(config=self.config.clean)
        self.collector = Collector()
    
    def collect(self, df1, df2):
        dfs = self.collector.collect(df1, df2)
        return dfs

    def build_pipe(self, **overrides):

        pipe = [
            ('harmonising', overrides.get('harmonise', SchemaHarmoniser(self.config.harmonise))),
            ('feature_eng', overrides.get('feature', FeatureTransformer(self.config.feature))),
        ]
        return Pipeline(pipe)
    
    @staticmethod
    def sort_by_time(X: pd.DataFrame, y_col: str = 'lmean_rejected', time_col: str = 'time'):
        """Sorts chronologically and extracts the target."""
        X_sorted = X.copy()
        y_sort = None
        if y_col in X_sorted.columns:
            y_sort = X_sorted.pop(y_col)
        else:
            logger.warning(f"Target column '{y_col}' not found in DataFrame.")
        if time_col in X_sorted.columns:
            if not pd.api.types.is_numeric_dtype(X_sorted[time_col]) and not pd.api.types.is_datetime64_any_dtype(X_sorted[time_col]):
                logger.info(f"Converting '{time_col}' to datetime...")
                X_sorted[time_col] = pd.to_datetime(X_sorted[time_col], errors='coerce')
            
            X_sorted = X_sorted.sort_values(by=time_col)
        else:
            logger.warning(f"Time column '{time_col}' not found. Sorting by Index.")
            X_sorted = X_sorted.sort_index()
        if y_sort is not None:
            y_sort = y_sort.reindex(X_sorted.index)

        return X_sorted, y_sort
    
    def normalise_time(self, X: pd.DataFrame, time_col: str = 'time'):
        """Min-Max scales the time column for neural network stability."""
        X_norm = X.copy()
        if pd.api.types.is_datetime64_any_dtype(X_norm[time_col]):
            X_norm[time_col] = X_norm[time_col].astype('int64') // 10**9
            
        t_min = X_norm[time_col].min()
        t_max = X_norm[time_col].max()
        
        X_norm[time_col] = (X_norm[time_col] - t_min) / (t_max - t_min + 1e-8)
        return X_norm

    @staticmethod
    def to_seq_data(tensor_x: torch.Tensor, tensor_y: torch.Tensor, seq_len: int = 64):
        """
        Creates sliding window sequences. 
        Input: [Total_Rows, Features] -> Output: [Num_Windows, Seq_Len, Features]
        """
        logger.info(f"Unfolding sequences of length {seq_len}...")
        
        seq_x = tensor_x.unfold(dimension=0, size=seq_len, step=1).transpose(1, 2)
        
        if tensor_y.dim() == 1:
            tensor_y = tensor_y.unsqueeze(-1)
        seq_y = tensor_y.unfold(dimension=0, size=seq_len, step=1).transpose(1, 2)
        
        return seq_x, seq_y
    
    @staticmethod
    def to_numpy(X):
        return X.to_numpy()
    
    @staticmethod
    def prepare_data(seq_x, seq_y, test_pct=0.2, batch_size=128):
        """
        Splits the sequence blocks and packages them into DataLoaders.
        Note: shuffle=False for time series chronological splitting!
        """
        X_train, X_test, y_train, y_test = train_test_split(
            seq_x.numpy(), seq_y.numpy(), 
            test_size=test_pct, 
            random_state=42, 
            shuffle=False  #-Chronological split
        )
        
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        test_x = torch.tensor(X_test, dtype=torch.float32)
        test_y = torch.tensor(y_test, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def run_pipeline(self, df1: pd.DataFrame, df2:pd.DataFrame, target_col: str, seq_len: int = 64, batch_size:int = 128):
        """Runs the entire pipeline end-to-end smoothly."""
        df1_clean = self.cleaner_df1.fit_transform(df1)
        df2_clean = self.cleaner_df2.fit_transform(df2)
        dfs_in = self.collect(df1_clean, df2_clean)
        pipe = self.build_pipe()
        df_processed = pipe.fit_transform(dfs_in)
        X_sorted, y_sorted = self.sort_by_time(df_processed, y_col=target_col, time_col='time')
        X_norm = self.normalise_time(X_sorted)
        X_tensor = torch.tensor(X_norm.to_numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(y_sorted.to_numpy(), dtype=torch.float32)
        seq_x, seq_y = self.to_seq_data(X_tensor, y_tensor, seq_len=seq_len)
        train_loader, test_loader = self.prepare_data(seq_x, seq_y, batch_size=batch_size)
        
        logger.info("Pipeline execution complete. DataLoaders ready for training.")
        return train_loader, test_loader