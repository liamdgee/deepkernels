#--Dependencies--#
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import yaml
from pathlib import Path
import os
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Union, Optional, TypeAlias, Literal
import logging


#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#-custom type hints for interaction terms-#
TransformMap: TypeAlias = dict[str, list[str]]
CustomInteractionMap: TypeAlias = dict[str, list[str]]


#-helper function-#
class ConfigLoader:
    def __init__(self, filename: str = "config.yaml"):
        self.filename = filename or "config.yaml"
        config = self._load_config(self.filename)
        self._data = config.get("features", {})

    @staticmethod
    def _load_config(filename: str) -> dict:
        """
        Loads a YAML file and returns it as a dictionary.
        """
        try:
            base_dir = Path(__file__).parent.parent.parent
        except NameError:
            base_dir = Path.cwd().parent

        config_path = base_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Failed to locate {filename} at {config_path}")

        try:
            with open(config_path, 'r') as file:
                return yaml.load(file, Loader=yaml.SafeLoader) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing YAML file: {e}")
            return {}

    @property
    def transforms(self) -> TransformMap:
        return self._data.get("transforms", {})
    
    @property
    def interactions(self) -> CustomInteractionMap:
        return self._data.get("interactions", {})


class NoveltyConfig(BaseModel):
    
    default_transforms: TransformMap = {
        'black_g_pct': ['log1p', 'l1_g'],
        'black_fs_pct': ['log', 'lfs'],
        'black_sg_pct': ['log', 'lsg'],
        'black_bifsg_pct': ['log1p', 'l1_bifsg'], 
        'black_s_pct': ['1', 's']
    }

    override_transforms: Optional[TransformMap] = None
    
    default_interactions: CustomInteractionMap = {
        'lg_bifsg', ['l1_g', 'black_bifsg_pct'], 
        'sg_lsg', ['black_sg_pct', 'lsg'], 
        'lg_lsg', ['l1_g', 'lsg'], 
        's_lsg', ['s', 'lsg'], 
        'bifsg_lsg', ['black_bifsg_pct', 'lsg'], 
        's_bifsg', ['s', 'black_bifsg_pct'], 
        'sg_s', ['black_sg_pct', 's'], 
        'lg_s', ['l1_g', 's'], 
        'lg_s_bifsg', ['l1_g', 's', 'black_bifsg_pct'], 
        'sg_s_bifsg', ['black_sg_pct', 's', 'black_bifsg_pct'], 
        'sg_lg_bifsg', ['black_sg_pct', 'l1_g', 'black_bifsg_pct'], 
        'sg_lg_s', ['black_sg_pct', 'l1_g', 's'], 
        'lg_bifsg_lsg', ['l1_g', 'black_bifsg_pct', 'lsg'], 
        's_bifsg_lsg', ['s', 'black_bifsg_pct', 'lsg'], 
        'sg_bifsg_lsg', ['black_sg_pct', 'black_bifsg_pct', 'lsg'], 
        'comp_int_1', ['black_sg_pct', 'l1_g', 's', 'black_bifsg_pct', 'lsg'], 
        'comp_int_2', ['l1_g', 's', 'black_bifsg_pct', 'lsg'], 
        'comp_int_3', ['black_sg_pct', 'l1_g', 'black_bifsg_pct', 'lsg']
    }

    override_interactions: Optional[CustomInteractionMap] = None

    #-scaling params-#
    scaling_method: Literal['power', 'standard', 'robust'] = 'power'
    eps: float = 1e-8
    centering: bool = True 

    @property
    def active_interactions(self) -> CustomInteractionMap:
        return self.override_interactions if self.override_interactions is not None else self.default_interactions
    
    @property
    def active_transforms(self) -> TransformMap:
        return self.override_transforms if self.override_transforms is not None else self.default_transforms



#--Class: Feature Engineering Pipeline--#
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.scaler_ = None
        self._eps = self.config.features.scaling.eps
        self._base_term_names = [item[1] for item in self.config.features.transforms]
        self._interaction_bases = [item[0] for item in self.config.features.transforms]
        self.interaction_names_ = [item[0] for item in self.config.features.interactions]

    def _apply_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log/log1p identities defined in YAML."""
        df = X.copy()
        for src, (func, feat) in self.config.features.transforms.items():
            if func == "log1p":
                df[feat] = np.log1p(df[src])
            elif func == "log":
                df[feat] = np.log(df[src] + self._eps)
            else:
                df[feat] = df[src]
        return df

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives multi-order interaction terms"""
        for feat, components in self.config.features.interactions:
            df[feat] = df[components].prod(axis=1)
        return df

    def fit(self, X, y=None):
        """Learns scaling parameters using sklearn scalers"""
        #--Orchestration--#
        df_with_base_terms = self._apply_transforms(X)
        df_with_interaction_terms = self._create_interactions(df_with_base_terms)
        
        #--Scaler as per config---#
        method = self.config.features.scaling.method
        if method == "power":
            self.scaler_ = PowerTransformer(method='yeo-johnson')
        elif method == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = StandardScaler()
        
        #-fit on newly engineered terms under "_create_interactions_()"
        if self.interaction_names_:
            self.scaler_.fit(df_with_interaction_terms[self.interaction_names_])

        return self

    def transform(self, X):
        """Orchestration module"""
        df = self._apply_transforms(X)
        df = self._create_interactions(df)
        
        if self.scaler_ is not None and self.interaction_names_:
            df[self.interaction_names_] = self.scaler_.transform(df[self.interaction_names_])
            
        return df