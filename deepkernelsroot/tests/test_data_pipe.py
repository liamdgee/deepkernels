import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from deepkernels.preprocess.pipe import DataOrchestrator

@pytest.fixture
def dummy_dataframe():
    """Creates a mock timeseries dataset with 100 rows."""
    # We scramble the dates to test the chronological sorter
    dates = pd.date_range(start='2023-01-01', periods=100).tolist()
    np.random.shuffle(dates) 
    
    df = pd.DataFrame({
        'time': dates,
        'feature_1': np.random.randn(100),
        'feature_2': np.random.rand(100),
        'lmean_rejected': np.random.randint(0, 2, 100) # Binary target
    })
    return df

@pytest.fixture
def orchestrator():
    # Mock the config to avoid needing the src modules for testing
    mock_config = MagicMock()
    return DataOrchestrator(config=mock_config)

def test_sort_by_time(orchestrator, dummy_dataframe):
    """Tests if the dataframe is correctly sorted chronologically and target extracted."""
    X_sorted, y_sorted = orchestrator.sort_by_time(
        dummy_dataframe, 
        y_col='lmean_rejected', 
        time_col='time'
    )
    
    # Check target extraction
    assert 'lmean_rejected' not in X_sorted.columns
    assert y_sorted is not None
    assert len(y_sorted) == 100
    
    # Check chronological sorting
    assert X_sorted['time'].is_monotonic_increasing

def test_normalise_time(orchestrator, dummy_dataframe):
    """Tests if datetime scaling bounds the time column between 0 and 1."""
    X_sorted, _ = orchestrator.sort_by_time(dummy_dataframe, y_col='lmean_rejected')
    X_norm = orchestrator.normalise_time(X_sorted)
    
    assert X_norm['time'].min() >= 0.0
    assert X_norm['time'].max() <= 1.0

def test_to_seq_data(orchestrator):
    """Tests the PyTorch unfold operation for sliding windows."""
    # 100 rows, 3 features
    tensor_x = torch.randn(100, 3) 
    tensor_y = torch.randn(100)
    seq_len = 10
    
    seq_x, seq_y = orchestrator.to_seq_data(tensor_x, tensor_y, seq_len=seq_len)
    
    # Formula for sliding windows: Total_Rows - Seq_Len + 1
    expected_windows = 100 - 10 + 1 # 91
    
    assert seq_x.shape == (expected_windows, seq_len, 3)
    assert seq_y.shape == (expected_windows, seq_len, 1)

def test_run_pipeline_end_to_end(orchestrator, dummy_dataframe, mocker):
    """
    Mocks the Sklearn pipeline and tests the entire end-to-end data flow,
    proving the DataLoaders output the correct dimensions.
    """
    # 1. Mock the build_pipe to return a passthrough pipeline
    # This skips the custom src modules but tests our orchestration logic!
    passthrough_pipe = Pipeline([('passthrough', FunctionTransformer(lambda x: x))])
    mocker.patch.object(orchestrator, 'build_pipe', return_value=passthrough_pipe)
    
    seq_len = 10
    batch_size = 16
    
    train_loader, test_loader = orchestrator.run_pipeline(
        df_raw=dummy_dataframe, 
        target_col='lmean_rejected', 
        seq_len=seq_len,
        batch_size=batch_size
    )
    
    # Expected sequence blocks: 91 total
    # 80% Train = 72 blocks, 20% Test = 19 blocks
    
    # Check Train Loader
    train_x_batch, train_y_batch = next(iter(train_loader))
    assert train_x_batch.shape == (batch_size, seq_len, 3) # 3 columns: time, feature_1, feature_2
    assert train_y_batch.shape == (batch_size, seq_len, 1)
    
    # Calculate expected dataset sizes
    train_dataset_size = len(train_loader.dataset)
    test_dataset_size = len(test_loader.dataset)
    
    assert train_dataset_size == 72
    assert test_dataset_size == 19