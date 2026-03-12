from sklearn.pipeline import Pipeline
from pydantic import Field, BaseModel

from deepkernels.preprocess.cleaner import DataCleaner, CleanerConfig
from deepkernels.preprocess.harmoniser import Collector, HarmoniserConfig, SchemaHarmoniser
from deepkernels.preprocess.novelty import FeatureTransformer, NoveltyConfig
import numpy as np
import pandas as pd
import logging
import torch
from typing import Union, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"


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
        return self.collector.collect(df1, df2)

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
        
        return seq_x.contiguous(), seq_y.contiguous()
    
    @staticmethod
    def prepare_data(seq_x, seq_y, seq_len: int = 64, val_pct=0.1, test_pct=0.1, batch_size=128, num_workers=4):
        """
        Splits sequence blocks chronologically into Train/Val/Test, 
        ensuring strict boundaries to prevent sliding window leakage.
        """
        total_seqs = seq_x.size(0)

        test_size = int(total_seqs * test_pct)
        val_size = int(total_seqs * val_pct)
        test_start = total_seqs - test_size
        val_end = test_start - seq_len 
        val_start = val_end - val_size
        train_end = val_start - seq_len
        if train_end <= 0:
            raise ValueError(f"Dataset too small for seq_len={seq_len} with these split percentages. "
                             f"Total seqs: {total_seqs}, required train_end: {train_end}")
        train_x = seq_x[:train_end]
        val_x = seq_x[val_start:val_end]
        test_x = seq_x[test_start:]
        train_y_point = seq_y[:train_end, -1, :]
        val_y_point = seq_y[val_start:val_end, -1, :]
        test_y_point = seq_y[test_start:, -1, :]
        logger.info(f"Chronological Split (Purged Overlaps) -> Train: {train_x.size(0)} | Val: {val_x.size(0)} | Test: {test_x.size(0)}")
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": num_workers,
            "drop_last": True,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "persistent_workers": True if num_workers > 0 else False  # <--- ADD THIS
        }
        indices_train = torch.arange(train_x.size(0))
        indices_val = torch.arange(val_x.size(0))
        indices_test = torch.arange(test_x.size(0))
        train_loader = DataLoader(TensorDataset(train_x, train_y_point, indices_train), **loader_kwargs)
        val_loader = DataLoader(TensorDataset(val_x, val_y_point, indices_val), **loader_kwargs)
        test_loader = DataLoader(TensorDataset(test_x, test_y_point, indices_test), **loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def run_pipeline(self, df1: pd.DataFrame, df2:pd.DataFrame, float_64:bool=False, target_col: str='lmean_rejected', drop_cols: Optional[list[str]]=None):
        """Runs the entire pipeline end-to-end smoothly."""
        df1_clean = self.cleaner_df1.fit_transform(df1)
        df2_clean = self.cleaner_df2.fit_transform(df2)
        dfs_in = self.collect(df1_clean, df2_clean)
        pipe = self.build_pipe()
        df_processed = pipe.fit_transform(dfs_in)
        X_sorted, y_sorted = self.sort_by_time(df_processed, y_col=target_col, time_col='time')
        
        drops = drop_cols if drop_cols is not None else ["lender_id", "lmean_rejected"]
        drops = [col for col in drops if col in X_sorted.columns]
        
        X_sorted = X_sorted.drop(columns=drops, errors='ignore')
        X_norm = self.normalise_time(X_sorted)

        # --- ADD THIS BLOCK HERE ---
        # Select only numeric columns to avoid the "numpy.object_" er
        X_numeric = X_norm.select_dtypes(include=[np.number])
        
        if X_numeric.shape[1] < X_norm.shape[1]:
            dropped = set(X_norm.columns) - set(X_numeric.columns)
            logger.warning(f"Dropping non-numeric columns before tensor conversion: {dropped}")
        # ---------------------------
        if float_64:
            # Change X_norm to X_numeric here
            X_tensor = torch.tensor(X_numeric.to_numpy(), dtype=torch.float64)
            y_tensor = torch.tensor(y_sorted.to_numpy(), dtype=torch.float64)
        else:
            # And change X_norm to X_numeric here
            X_tensor = torch.tensor(X_numeric.to_numpy(), dtype=torch.float32)
            y_tensor = torch.tensor(y_sorted.to_numpy(), dtype=torch.float32)
        
        return X_tensor, y_tensor