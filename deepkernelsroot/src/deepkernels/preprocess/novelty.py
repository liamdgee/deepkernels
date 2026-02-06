#--Dependencies--#
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import yaml
from pathlib import Path
import os
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Union, Optional, TypeAlias, Literal, Annotated
import logging
from sklearn.utils.validation import check_is_fitted


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
    
     #--logic: {col_source_name: ['interaction_base', 'alias_for_pipeline']}-#
    default_transforms: TransformMap = {
        'black_g_pct': ['log1p', 'l1_g'],
        'black_fs_pct': ['log', 'lfs'],
        'black_sg_pct': ['log', 'lsg'],
        'black_bifsg_pct': ['log1p', 'l1_bifsg'], 
        'black_s_pct': ['1', 's']
    }

    override_transforms: Optional[TransformMap] = None
    
    #---logic: [new_feature, [i_term_1, i_term_2, ..., i_term_n]]
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
    eps: Annotated[float, Field(ge=1e-14, le=1e-2)] = 1e-8
    centering: bool = True 

    @property
    def active_interactions(self) -> CustomInteractionMap:
        return self.override_interactions if self.override_interactions is not None else self.default_interactions
    
    @property
    def active_transforms(self) -> TransformMap:
        return self.override_transforms if self.override_transforms is not None else self.default_transforms



#--Class: Feature Engineering Pipeline--#
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            config: Optional[NoveltyConfig] = None,
            transforms: Optional[TransformMap] = None,
            interactions: Optional[CustomInteractionMap] = None,
            scaling_method: Literal['power', 'standard', 'robust'] = 'power',
            eps: Annotated[float, Field(ge=1e-14, le=1e-2)] = 1e-8,
            centering: bool = True,
            **kwargs
        ):

        self.config = config if config else NoveltyConfig()
        self.transforms = transforms if transforms is not None else self.config.active_transforms
        self.interactions = interactions if interactions is not None else self.config.active_interactions
        self.eps = eps or self.config.eps
        self.scaling_method = scaling_method or self.config.scaling_method
        self.scaler_ = None
        

    def _apply_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log/log1p identities defined in YAML."""
        df = X.copy()
        for src, (func, alias) in self.transforms.items():
            logger.info(f"Creating {alias} term using math base: {func} from source term: {src}")
            if func == "log1p":
                df[alias] = np.log1p(df[src])
            elif func == "log":
                df[alias] = np.log(df[src] + self.eps)
            else:
                df[alias] = df[src] + self.eps
        return df

    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derives multi-order interaction terms"""
        df = X.copy()
        created = []

        for key, components in self.interactions.items():
            existing = [col for col in components if col in df.columns]
            if len(existing) == len(components):
                logger.info(f"Generating feature: {key} from: {len(components)} interaction components")
                df[key] = df[components].prod(axis=1)
                created.append(key)
            else:
                logger.warning(f"Skipping: {key} -- missing: {set(components) - set(existing)}")
        
        return df, created

    def fit(self, X, y=None):
        """Learns scaling parameters using sklearn scalers"""
        #--Orchestration--#
        df = self._apply_transforms(X)
        df_engineered, cols_created = self._create_interactions(df)
        self.features_out_ = cols_created
        
        #--Scaler as per config---#
        scalers = {
            "power": PowerTransformer(method='yeo-johnson'),
            "robust": RobustScaler(),
            "standard": StandardScaler()
        }
        
        self.scaler_ = scalers.get(self.scaling_method, PowerTransformer(method='yeo-johnson'))
        
        #-fit scaler on newly engineered terms under "_create_interactions()" --#
        if self.features_out_:
            verified = [col for col in self.features_out_ if col in df_engineered.columns]
            self.scaler_.fit(df_engineered[verified])

        return self

    def transform(self, X):
        """Orchestration module"""
        #-execute (transform df)-#
        check_is_fitted(self, ['scaler_', 'features_out_'])
        df = self._apply_transforms(X)
        df_engineered, _ = self._create_interactions(df)
        
        verified = [col for col in self.features_out_ if col in df_engineered.columns]
        if verified:
            logger.info(f"Adding {len(verified)} cols to final df and scaling using {self.scaler_}")
            df_engineered[verified] = self.scaler_.transform(df_engineered[verified])
             
        return df_engineered