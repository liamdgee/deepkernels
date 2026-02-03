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
    num_features: List[str]
    cat_features: List[str]
    task_type: Literal['regression', 'classification'] = 'regression'
    scaler: Literal['robust', 'standard'] = 'robust'
    batch_size: int = Field(512, gt=1, le=4096)
    device: str = "auto" #--also can be cuda mps or cpu--#
    random_state: int = 42
    test_pct: float = Field(0.2, gt=0.0, lt=1.0)
    target_variable: str

#---Class Definition: Torch Preprocessor--#
class TorchPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compliant transformer that manages feature engineering.
    """
    def __init__(self, config: PreprocessConfig):
        self.config = config
        
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        
        self.feature_names_out_ = None
        self.device_ = self._get_device()
    
    def _get_device(self) -> torch.device:
        """Fetches device for computations"""
        if self.config.device != "auto":
            return torch.device(self.config.device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _get_scaler(self):
        return RobustScaler() if self.config.scaler == "robust" else StandardScaler()
    
    def _assemble_preprocessor(self) -> ColumnTransformer:
        """Builds scikit-learn ColumnTransformer"""
        num_transformer = Pipeline(steps=[('scaler', self._get_scaler())])
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.config.num_features),
                ('cat', cat_transformer, self.config.cat_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
    
    def _fit_target(self, y:np.ndarray):
        """fots encoder if classification task"""
        if self.config.task_type == 'classification':
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
    
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
    
    def _attempt_extract_y(self, X: pd.DataFrame, y=None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Helper to separate target variable from dataframe if needed."""
        if y is None and self.config.target_variable in X.columns:
            y = X[self.config.target_variable].values
            X = X.drop(columns=[self.config.target_variable])
        return X, y
    
    def _engineer_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Handles the Scikit-Learn transformation logic.
        """
        check_is_fitted(self, ['preprocessor_'])

        X_proc = self.preprocessor_.transform(X)

        #--Dense Tensor Outputs-#
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
            
        return X_proc

    def fit(self, X: pd.DataFrame, y=None):
        """Fits preprocessor on X and encoder on Y"""

        #--seperate X and y --#
        if y is None and self.config.target_variable in X.columns:
            y = X[self.config.target_variable].values
            X = X.drop(columns=[self.config.target_variable])
        
        self.preprocessor_ = self._assemble_preprocessor()
        self.preprocessor_.fit(X)
        self.feature_names_out_ = self.preprocessor_.get_feature_names_out()

        if y is not None:
            self._fit_target(y)
        
        return self

    def transform(self, X:pd.DataFrame, y=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Converts input data into pytorch tensors from input data"""
        
        check_is_fitted(self, ['preprocessor_'])

        X_in, y_in = self._attempt_extract_y(X, y)

        X_np = self._engineer_transform(X_in)
        
        Xt = torch.tensor(X_np, dtype=torch.float32).to(self.device_)

        yt = None

        if y_in is not None:
            yt = self._target_to_tensor(y_in)
        
        return Xt, yt

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_