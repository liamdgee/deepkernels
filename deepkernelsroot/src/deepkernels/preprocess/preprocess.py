#filename: preprocess_pipe.py

#---Dependencies (core)---#
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Literal, Optional
from pydantic import BaseModel, Field

#-Scikit-Learn Dependencies-#
import sklearn
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
    numeric_features: List[str] = Field(default_factory=lambda: ['log_amountsought', 'log_total_percap_inc'])
    categorical_features: List[str] = Field(default_factory=lambda: ['black_final_race_flag'])
    task_type: Literal['regression', 'classification'] = 'regression'
    scaler: Literal['robust', 'standard'] = 'robust'
    batch_size: int = Field(512, gt=1, le=4096)
    device: str = "cpu" #--also can be cuda mps or cpu--#
    random_state: int = 42
    test_pct: float = Field(0.2, gt=0.0, lt=1.0)
    target_variable: str = 'lmean_rejected'
    id_cols : List[str] = Field(default_factory=lambda: ['unique_borrower', 'lender_clean', 'time', 'black_final_race'])

#---Class Definition: Torch Preprocessor--#
class TorchPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compliant transformer that manages feature engineering.
    """
    def __init__(self, config: Optional[PreprocessConfig] = None,
                 num_overrides: Optional[List[str]] = ['log_amountsought', 'log_total_percap_inc'],
                 cat_overrides: Optional[List[str]] = ['black_final_race_flag'],
                 id_cols: Optional[List[str]] = None,
                 task_type: Optional[Literal['classification', 'regression']] = 'regression', 
                 use_robust_scaler: bool = True, 
                 batch_size: Optional[int] = 512, 
                 device: Optional[Literal['auto', 'cpu', 'cuda', 'mps']] = "auto", 
                 random_state: Optional[int] = 42, test_pct: Optional[float] = None, 
                 target_variable: Optional[str] = 'lmean_rejected'):
        
        self.config = config if config else PreprocessConfig()
        
        default_target = 'lmean_rejected'
        self.target_variable = target_variable if target_variable else default_target
        default_id_cols = ['unique_borrower', 'lender_clean', 'time', 'black_final_race']
        self.id_cols = id_cols if id_cols else default_id_cols
        self.drop_cols = self.id_cols.copy()
        if self.target_variable:
            self.drop_cols.append(self.target_variable)
        
        self.numeric_features = num_overrides or self.config.numeric_features or []
        self.categorical_features = cat_overrides or self.config.categorical_features or []
        
        self.device_str = device if device else "auto"
        self.device = self._get_device()
        self.random_state = random_state if random_state else self.config.random_state or 42
        self.batch_size = batch_size if batch_size else self.config.batch_size or 512
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.task_type = task_type if task_type else self.config.task_type or 'regression'
        
        
        self.feature_names_out_ = None
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.label_encoder_: Optional[LabelEncoder] = None
    
    def _get_device(self) -> torch.device:
        """Fetches device for computations"""
        if self.device_str != "auto":
            return torch.device(self.device_str)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_scaler(self):
        return RobustScaler() if self.scaler == "robust" else StandardScaler()
    
    def _assemble_preprocessor(self) -> ColumnTransformer:
        """Builds scikit-learn ColumnTransformer"""
        num_transformer = Pipeline(steps=[('scaler', self._get_scaler())])
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numeric_features),
                ('cat', cat_transformer, self.categorical_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
    
    def _fit_target(self, y: np.ndarray):
        """fots encoder if classification task"""
        if self.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
            return self.label_encoder_.classes_
        return self
    
    def _target_to_tensor(self, y:np.ndarray):
        if self.config.task_type == 'regression':
            if y.dtype == 'object':
                 y = y.astype(float)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            if self.label_encoder_ is None:
                raise ValueError(f"Label Encoder is currently set to: {self.label_encoder_} -- This param should not be null given task is set to: {self.config.task_type}")
            y_enc = self.label_encoder_.transform(y)
            y_tensor = torch.tensor(y_enc, dtype=torch.long)
        
        return y_tensor.to(self.device_)
    
    def _attempt_extract_y(self, X: pd.DataFrame, y=None):
        """Helper to separate target variable from dataframe if needed."""
        if self.target_variable in X.columns:
            y_temp = X[self.target_variable].values
            X = X.drop(columns=[self.target_variable])
            if y is None:
                y = y_temp
                
        return X, y
    
    def _engineer_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Handles the Scikit-Learn transformation logic.
        """
        check_is_fitted(self, ['preprocessor_'])

        X_proc = self.preprocessor_.transform(X)

        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
            
        return X_proc

    def fit(self, X: pd.DataFrame, y=None):
        """Fits preprocessor on X and encoder on Y"""

        #--seperate X and y --#
        X, y = self._attempt_extract_y(X, y)
        self.numeric_features, self.categorical_features = self._get_column_types(X)
        self.preprocessor_ = self._assemble_preprocessor()
        self.preprocessor_.fit(X)
        self.feature_names_out_ = self.preprocessor_.get_feature_names_out()

        if y is not None and self.config.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
            
        return self


    def transform(self, X:pd.DataFrame, y=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Converts input data into pytorch tensors from input data"""
        
        check_is_fitted(self, ['preprocessor_'])

        X_in, y_in = self._attempt_extract_y(X, y)

        X_proc = self.preprocessor_.transform(X_in)

        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
        
        Xt = torch.tensor(X_proc, dtype=torch.float32)
        yt = None

        if y_in is not None:
            if self.config.task_type == 'classification':
                y_enc = self.label_encoder_.transform(y_in)
                yt = torch.tensor(y_enc, dtype=torch.long)
            else:
                yt = torch.tensor(y_in.astype(float), dtype=torch.float32).view(-1, 1)
        
        Xt = Xt.to(self.device_)
        if yt is not None:
            yt = yt.to(self.device_)
        return Xt, yt
    
    def _get_column_types(self, X_df):
        """Auto-detects columns, respecting overrides and drop_cols"""
        final_num = self.numeric_features.copy()
        final_cat = self.categorical_features.copy()
        auto_detect = (len(final_num) == 0 and len(final_cat) == 0)
        
        if auto_detect:
            num_cols = X_df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
            final_num = [c for c in num_cols if c not in self.drop_cols]
            final_cat = [c for c in cat_cols if c not in self.drop_cols]
        return final_num, final_cat

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_