#filename: preprocess_pipe.py

#---Dependencies (core)---#
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Literal, Optional, Annotated
from pydantic import BaseModel, Field

#-Scikit-Learn Dependencies-#
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---Class: Config--#
class PreprocessConfig(BaseModel):
    numeric_features: List[str] = Field(default_factory=lambda: [
        'ln_tenure', 'log_amountsought', 'log_total_percap_inc', 'dissim_scaled_z',
        'isolation_scaled_z', 'animus_scaled_z', 'iat_score_f_scaled_z',
        'black_sg_pct_logit_z', 'share_black_pop_geba_logit_z',
        'black_s_pct_logit_z', 'share_pop_black_logit_z',
        'black_bifsg_pct_logit_z', 'black_fs_pct_logit_z',
        'black_g_pct_logit_z', 'l1_g', 'lfs', 'lsg', 'l1_bifsg', 'sg_lsg',
        'lg_lsg', 's_lsg', 'bifsg_lsg', 'sg_s', 'lg_s', 'comp_int_1',
        'comp_int_2'
    ])
    categorical_features: List[str] = Field(default_factory=lambda: [
        'is_fintech', 'is_cdfi', 'is_creditunion', 'is_bank',
        'mdi_flag', 'factoringccmca_flag', 'is_ever_ceo_flag',
        'has_masters_flag', 'has_postgrad_flag'
    ])
    task_type: Literal['regression', 'classification'] = 'regression'
    scaler: Literal['robust', 'standard'] = 'robust'
    batch_size: Annotated[int, Field(le=4096, ge=4)] = 128
    device: str = "auto"
    random_state: int = 42
    test_pct: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.2
    y_target: str = 'lmean_rejected'
    id_cols : List[str] = Field(default_factory=lambda: ['lender_clean', 'time'])


#---Class Definition: Torch Preprocessor--#
class TorchPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: Optional[PreprocessConfig] = None,
                 num_overrides: Optional[List[str]] = None,
                 cat_overrides: Optional[List[str]] = None,
                 id_cols: Optional[List[str]] = None,
                 task_type: Optional[Literal['classification', 'regression']] = None, 
                 use_robust_scaler: bool = True, 
                 batch_size: Optional[int] = None, 
                 device: Optional[Literal['auto', 'cpu', 'cuda', 'mps']] = None, 
                 random_state: Optional[int] = None, 
                 test_pct: Optional[float] = None, 
                 y_target: Optional[str] = None):
        
        self.config = config or PreprocessConfig()
        
        self.y_target = y_target if y_target else self.config.y_target
        self.test_pct = test_pct or self.config.test_pct
        self.id_cols = id_cols or self.config.id_cols
        self.numeric_features = num_overrides if num_overrides is not None else self.config.numeric_features
        self.categorical_features = cat_overrides if cat_overrides is not None else self.config.categorical_features
        
        self.device_str = device if device is not None else self.config.device or "auto"
        self.device = self._get_device()

        self.random_state = random_state or self.config.random_state or 42
        self.batch_size = batch_size if batch_size else self.config.batch_size
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.task_type = task_type or self.config.task_type or 'regression'
        
        self.feature_names_out_ = None
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.label_encoder_: Optional[LabelEncoder] = None
    
    def _get_device(self) -> torch.device:
        if self.device_str != "auto":
            return torch.device(self.device_str)
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            
    def _to_numpy(self, data) -> np.ndarray:
        """
        Sanitizes input data to ensures it is a dense Numpy array.
        Handles MPS/GPU Tensors by moving to CPU first.
        """
        if data is None:
            return None
        
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
            
        if hasattr(data, "toarray"):
            return data.toarray()
            
        if hasattr(data, "values"):
            return data.values
            
        return np.array(data)
    
    def _assemble_preprocessor(self) -> ColumnTransformer:
        num_scaler = self.scaler if self.scaler else RobustScaler()
        cat_scaler = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        num_transformer = Pipeline(steps=[('scaler', num_scaler)])
        cat_transformer = Pipeline(steps=[('onehot', cat_scaler)])
        
        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numeric_features), 
                ('cat', cat_transformer, self.categorical_features)
            ], 
            remainder='drop', 
            verbose_feature_names_out=False
        )
    
    def _attempt_extract_y(self, X: pd.DataFrame, y=None):
        """Robustly separate target from X."""
        target = y if y is not None
        
        # Check string vs data
        if isinstance(target, str):
            if target in X.columns:
                y_out = X[target].values
                X = X.drop(columns=[target])
                return X, y_out
            return X, None
        
        return X, target
    
    def fit(self, X: pd.DataFrame, y=None):
        X, y = self._attempt_extract_y(X, y)
        self.numeric_features, self.categorical_features = self._get_column_types(X)
        self.preprocessor_ = self._assemble_preprocessor()
        self.preprocessor_.fit(X)
        self.feature_names_out_ = self.preprocessor_.get_feature_names_out()

        if y is not None and self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(self._to_numpy(y))
            
        return self

    # RENAMED to avoid Scikit-Learn 'transform' hook collision
    def process_to_tensors(self, X: pd.DataFrame, y=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """Converts input data into pytorch tensors safely."""
        check_is_fitted(self, ['preprocessor_'])
        
        if isinstance(X, pd.DataFrame):
            meta_cols = [col for col in self.id_cols if col in X.columns]
            X = X.drop(columns=meta_cols, errors='ignore')
        
        X_in, y_in = self._attempt_extract_y(X, y)
        X_proc = self.preprocessor_.transform(X_in)
        
        # Explicit Numpy conversion
        X_np = self._to_numpy(X_proc)
        Xt = torch.tensor(X_np, dtype=torch.float32)
        
        yt = None
        if y_in is not None:
            y_np = self._to_numpy(y_in)
            
            if self.task_type == 'classification':
                y_enc = self.label_encoder_.transform(y_np)
                yt = torch.tensor(y_enc, dtype=torch.long)
            else:
                yt = torch.tensor(y_np.astype(float), dtype=torch.float32).view(-1)
        
        # Move to Device
        Xt = Xt.to(self.device)
        if yt is not None:
            yt = yt.to(self.device)
            
        return Xt, yt, None 
    
    def _get_column_types(self, X_df):
        exclude = set(self.id_cols + [self.y_target])
        
        if not hasattr(X_df, "select_dtypes"):
            return self.numeric_features, self.categorical_features

        final_num = self.numeric_features if self.numeric_features else \
                    [c for c in X_df.select_dtypes(include=['number']).columns if c not in exclude]
        final_cat = self.categorical_features if self.categorical_features else \
                    [c for c in X_df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
        
        return final_num, final_cat

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
    
    def get_dataloader(self, X: pd.DataFrame, y=None, shuffle: bool = True) -> DataLoader:
        # Call the RENAMED function
        Xt, yt, _ = self.process_to_tensors(X, y)
        
        if yt is None:
            dataset = TensorDataset(Xt)
        else:
            dataset = TensorDataset(Xt, yt)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

#--- Helper Function ---#
def prepare_loaders(df: pd.DataFrame, target_col: str, test_pct: float = 0.2, batch_size: int = 256):
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=42)
    
    preprocessor = TorchPreprocessor(y_target=target_col, batch_size=batch_size)
    preprocessor.fit(X_train, y_train)
    
    train_loader = preprocessor.get_dataloader(X_train, y_train, shuffle=True)
    test_loader = preprocessor.get_dataloader(X_test, y_test, shuffle=False)
    
    input_dim = len(preprocessor.get_feature_names_out())
    
    return train_loader, test_loader, input_dim