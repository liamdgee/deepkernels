import pytest
import pandas as pd
import numpy as np
from src.preprocess.cleaner import Cleaner

# --- Fixtures ---
@pytest.fixture
def raw_df():
    """Creates a dirty DataFrame covering all edge cases."""
    return pd.DataFrame({
        '  ID  ': [1, 2, 3, 4, 5],
        'Date Column': ['2023-01-01', '01/02/2023', 'invalid_date', '2023.03.01', np.nan],
        'Price': ['$100', '200.50', 'Unknown', '300', np.nan],
        'Category': ['A', 'B', '  A  ', 'c', 'N/A'],
        'All_Nulls': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Mixed_Numeric': [10, '20', 30, '40', 50] # Ints stored as mixed
    })

@pytest.fixture
def cleaner(raw_df):
    return Cleaner(raw_df)

# --- Unit Tests ---

def test_canonicalise_headers(cleaner):
    """Test header cleaning: lowercase, strip, snake_case."""
    cleaner.canonicalise_headers()
    df = cleaner.get_df()
    
    expected_cols = ['id', 'date_column', 'price', 'category', 'all_nulls', 'mixed_numeric']
    assert list(df.columns) == expected_cols

def test_drop_nulls_columns(cleaner):
    """Test dropping columns that are 100% null."""
    # The 'All_Nulls' column is 100% empty, so threshold 0.8 should drop it
    cleaner.drop_nulls(threshold=0.8, axis=1)
    df = cleaner.get_df()
    
    assert 'All_Nulls' not in df.columns
    assert 'ID' in df.columns  # Original columns should remain (if not canonicalised yet)

def test_normalise_strings(cleaner):
    """Test string standardization and null text handling."""
    cleaner.normalise()
    df = cleaner.get_df()
    
    # Check '  A  ' became 'a'
    assert 'a' in df['Category'].values
    assert '  A  ' not in df['Category'].values
    
    # Check 'N/A' became np.nan
    assert df['Category'].isna().sum() > 0

def test_convert_datetimes(cleaner):
    """Test heuristic datetime conversion."""
    cleaner.convert_datetimes()
    df = cleaner.get_df()
    
    # 'Date Column' should be converted to datetime64
    assert pd.api.types.is_datetime64_any_dtype(df['Date Column'])
    
    # The 'invalid_date' string should become NaT (Not a Time)
    assert pd.isna(df['Date Column'].iloc[2])

def test_infer_types_numeric(cleaner):
    """Test that strings looking like numbers ('20') become numbers."""
    cleaner.infer_types()
    df = cleaner.get_df()
    
    # 'Mixed_Numeric' should become integers
    assert pd.api.types.is_numeric_dtype(df['Mixed_Numeric'])
    assert df['Mixed_Numeric'].sum() == 150

def test_fill_nulls_strategy(cleaner):
    """Test imputation strategies."""
    # Setup specific DF for math check
    df_math = pd.DataFrame({'val': [10, 20, np.nan]})
    c = Cleaner(df_math)
    
    # Test Mean Imputation
    c.fill_nulls(strategy='mean')
    result = c.get_df()
    
    # Mean of 10 and 20 is 15
    assert result['val'].iloc[2] == 15.0

def test_full_pipeline_integrity(raw_df):
    """
    Integration Test: Run the full MLOps pipeline.
    This ensures methods chain correctly and output is model-ready.
    """
    c = Cleaner(raw_df)
    
    final_df = (
        c
        .canonicalise_headers()
        .drop_nulls(axis=1, threshold=0.9) # Drop empty cols
        .normalise()                       # Fix strings
        .convert_datetimes()               # Fix dates
        .infer_types()                     # Fix numerics
        .fill_nulls(strategy='zero')       # Fill remaining holes
        .drop_dupes()
        .get_df()
    )
    
    # 1. Check Schema
    assert 'all_nulls' not in final_df.columns
    assert pd.api.types.is_datetime64_any_dtype(final_df['date_column'])
    assert pd.api.types.is_numeric_dtype(final_df['mixed_numeric'])
    
    # 2. Check Data Integrity (No Nulls should remain for Model input)
    # Note: Datetimes might still have NaT if we didn't fill them, but numerics should be filled
    assert final_df['mixed_numeric'].isna().sum() == 0