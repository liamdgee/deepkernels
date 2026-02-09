#---Dependencies--#
import sklearn
sklearn.set_config(transform_output="pandas")

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
UpToFiveStrings: TypeAlias = Union[tuple[str], tuple[str, str], tuple[str, str, str], tuple[str, str, str, str], tuple[str, str, str, str, str]]

TwoStrings: TypeAlias = tuple[str, str]

TransformMap: TypeAlias = dict[str, TwoStrings]

CustomInteractions: TypeAlias = dict[str, UpToFiveStrings]


#-helper function-#
###class ConfigLoader:
    #def __init__(self, filename: str = "config.yaml"):
     #   self.filename = filename or "config.yaml"
      #  config = self._load_config(self.filename)
       # self._data = config.get("features", {})
#
 #   @staticmethod
  #  def _load_config(filename: str) -> dict:
   #     """
    #    Loads a YAML file and returns it as a dictionary.
     #   """
      #  try:
       #     base_dir = Path(__file__).resolve().parent.parent.parent
        #except NameError:
         #   base_dir = Path.cwd()
#
 #       config_path = base_dir / filename
#
 #       if not config_path.exists():
  #          logger.warning(f"Config file not found at {config_path}. Using internal defaults.")
   #         return {}
    #    
     #   try:
      #      with open(config_path, 'r') as file:
       #         return yaml.load(file, Loader=yaml.SafeLoader) or {}
        #except yaml.YAMLError as e:
         #   logger.warning(f"Error parsing YAML file: {e}")
          #  return {}

    ##@property
    ##def transforms(self) -> TransformMap:
        ##return self._data.get("transforms", {})
    
    ##@property
    ##def interactions(self) -> CustomInteractionMap:
        ##return self._data.get("interactions", {})


class NoveltyConfig(BaseModel):
     #--logic: {'alias_for_pipeline': ('interaction_base', 'col_source_name')}-#
    transforms: TransformMap = {
        'l1_g' : ('log1p', 'black_g_pct'),
        'lfs': ('log', 'black_fs_pct'),
        'lsg': ('log', 'black_sg_pct'),
        'l1_bifsg': ('log1p', 'black_bifsg_pct'), 
    }
    
    #---logic: [new_feature, [i_term_1, i_term_2, ..., i_term_n]]
    interactions: CustomInteractions = {
        'lg_bifsg': ('l1_g', 'black_bifsg_pct'), 
        'sg_lsg': ('black_sg_pct', 'lsg'), 
        'lg_lsg': ('l1_g', 'lsg'), 
        's_lsg': ('black_s_pct', 'lsg'), 
        'bifsg_lsg': ('black_bifsg_pct', 'lsg'), 
        's_bifsg': ('black_s_pct', 'black_bifsg_pct'), 
        'sg_s': ('black_sg_pct', 'black_s_pct'), 
        'lg_s': ('l1_g', 'black_s_pct'), 
        'lg_s_bifsg': ('l1_g', 'black_s_pct', 'black_bifsg_pct'), 
        'sg_s_bifsg': ('black_sg_pct', 'black_s_pct', 'black_bifsg_pct'), 
        'sg_lg_bifsg': ('black_sg_pct', 'l1_g', 'black_bifsg_pct'), 
        'sg_lg_s': ('black_sg_pct', 'l1_g', 'black_s_pct'), 
        'lg_bifsg_lsg': ('l1_g', 'black_bifsg_pct', 'lsg'),
        's_bifsg_lsg': ('black_s_pct', 'black_bifsg_pct', 'lsg'), 
        'sg_bifsg_lsg': ('black_sg_pct', 'black_bifsg_pct', 'lsg'),
        'comp_int_1': ('black_sg_pct', 'l1_g', 'black_s_pct', 'black_bifsg_pct', 'lsg'), 
        'comp_int_2': ('l1_g', 'black_s_pct', 'black_bifsg_pct', 'lsg'), 
        'comp_int_3': ('black_sg_pct', 'l1_g', 'black_bifsg_pct', 'lsg')
    }

    #-feature eng config-#
    seg_cols: list[str] = ["dissim_scaled", "isolation_scaled", "animus_scaled", "iat_score_f_scaled", "mdi"] #-to z score-#
    bisg_cols: list[str] = ["black_s_pct", "black_g_pct", "black_fs_pct", "black_bifsg_pct", "black_sg_pct"] #-clip -> logits -> logit_z--#--mean_bisg-#
    pop_cols: list[str] = ["share_pop_black", "share_black_pop_geba"] #-clip -> logits -> logit_z--#
    lender_cols: list[str] = ["fintech", "cdfi", "creditunion", "bank"] #-type to flag -> create lender_id column-#
    to_log_transform: Union[list[str], str] = ['total_percap_inc'] 
    drop_patterns: list[str] = ['lmean_', 'approved_', 'delta_', 'false_', 'fintech_lenders_', 'lenders_sent_', 'non_fintech_', '_sent_to', 'one_per_borrower', 'rejected', 'shr_app_', 'shr_appr_', 'shr_loan_', 'shr_rej_', 'true_', 'year', 'unique_', 'black_final_race', '_rand', 'num_', 'unique_borrower', 'lender_borrower']
    
    
    #-core config-#
    num_cols: list[str] = ['animus_scaled', 'black_bifsg_pct', 'black_fs_pct', 'black_g_pct', 'black_s_pct', 'black_sg_pct', 'dissim_scaled', 'iat_score_f_scaled', 'isolation_scaled', 'ln_tenure', 'log_amountsought', 'share_black_pop_geba', 'share_pop_black', 'total_percap_inc']
    cat_cols: list[str] = ['bank', 'cdfi', 'creditunion', 'fintech',  'mdi', 'factoringccmca', 'is_ever_ceo', 'has_masters', 'has_postgrad']
    id_cols: list[str] = ['lender_clean', 'time', 'src_idx']
    y_target: Union[list[str], str] = ['lmean_rejected']
    

    #-scaling params-#
    scaling_method: Literal['power', 'standard', 'robust'] = 'power'
    eps: float = 1e-8



#--Class: Feature Engineering Pipeline--#
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            config: Optional[NoveltyConfig] = None,
            transforms: Optional[TransformMap] = None,
            interactions: Optional[CustomInteractions] = None,
            scaling_method: Literal['power', 'standard', 'robust'] = 'power',
            eps: Optional[float] = 1e-8,
            drop_patterns: Optional[List[str]] = None,
            id_cols: Optional[List[str]] = None,
            num_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None,
            seg_cols: Optional[List[str]] = None,
            bisg_cols: Optional[List[str]] = None,
            pop_cols: Optional[List[str]] = None,
            lender_cols: Optional[List[str]] = None,
            to_log_transform: Optional[List[str]] = None,
            y_target: Optional[str] = None,
            **kwargs
        ):

        self.config = config or NoveltyConfig()
        self.transforms = transforms or self.config.transforms
        self.interactions = interactions or self.config.interactions
        self.scaling_method = scaling_method or self.config.scaling_method
        self.eps = eps or self.config.eps

        self.drop_patterns = drop_patterns if drop_patterns is not None else self.config.drop_patterns
        self.id_cols = id_cols if id_cols is not None else self.config.id_cols
        self.num_cols = num_cols if num_cols is not None else self.config.num_cols
        self.cat_cols = cat_cols if cat_cols is not None else self.config.cat_cols
        self.seg_cols = seg_cols if seg_cols is not None else self.config.seg_cols
        self.bisg_cols = bisg_cols if bisg_cols is not None else self.config.bisg_cols
        self.pop_cols = pop_cols if pop_cols is not None else self.config.pop_cols
        self.lender_cols = lender_cols if lender_cols is not None else self.config.lender_cols
        self.to_log_transform = to_log_transform if to_log_transform is not None else self.config.to_log_transform
        self.y_target = y_target if y_target is not None else self.config.y_target
        
        #-State-#
        self.stats_ = {}
        self.scaler_ = None
        self.numeric_cols_out_ = []
        self.features_out_ = []

    def _to_log(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Helper function to log transform specified columns."""
        df = df_in.copy()
        for col in self.to_log_transform:
            if col in df.columns:
                clipped = df[col].clip(lower=self.eps) #-clip to avoid log(0) and log of negatives-#
                df[f'log_{col}'] = np.log(clipped)
        return df
    
    def _drop_unwanted(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explicitly drops ID and Ignore columns."""
        if not self.drop_patterns:
            return df
        cols_to_drop = []
        if self.drop_patterns:
            for col in df.columns:
                if any (ptrn in col for ptrn in self.drop_patterns):
                    cols_to_drop.append(col)
        dropping = list(set(cols_to_drop))
        if dropping:
            drop = [c for c in dropping if c in df.columns and c not in self.id_cols]
            if drop:
                logger.info(f"Dropping {len(drop)} unwanted columns based on patterns.")
            df = df.drop(columns=drop, errors='ignore')
        return df
        
    def _to_z_score(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """
        Applies Z-score. 
        If fit_mode=True, it CALCULATES and SAVES stats.
        If fit_mode=False, it USES saved stats.
        """
        for col in self.seg_cols:
            if col in df.columns:
                if fit_mode:
                    #-LEARN-#
                    self.stats_[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std() + self.eps
                    }
                #-APPLY-#
                if col in self.stats_:
                    mu = self.stats_[col]['mean']
                    sigma = self.stats_[col]['std']
                    df[f'{col}_z'] = (df[col] - mu) / sigma
            else:
                if fit_mode:
                    logger.warning(f"Column {col} not found for z-score scaling.")
        return df

    def _to_logit_z(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """
        Applies Logit-Z.
        If fit_mode=True, it CALCULATES and SAVES stats.
        """
        cols_to_process = list(set(self.bisg_cols + self.pop_cols))
        for col in cols_to_process:
            if col in df.columns:
                clipped = df[col].clip(self.eps, 1 - self.eps)
                logit_vals = np.log(clipped / (1 - clipped))
                stats_key = f'{col}_logit'
                if fit_mode:
                    #-LEARN-#
                    self.stats_[stats_key] = {
                        'mean': logit_vals.mean(),
                        'std': logit_vals.std() + self.eps
                    }       
                #-APPLY-#
                if stats_key in self.stats_:
                    mu = self.stats_[stats_key]['mean']
                    sigma = self.stats_[stats_key]['std']
                    df[f'{stats_key}_z'] = (logit_vals - mu) / sigma
        return df
    
    def _create_binary_flags(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Helper function to create binary flags for lender types and outcome variables."""
        df = df_in.copy()
        existing_lenders = [lender for lender in self.lender_cols if lender in df.columns]
        other_flags = [col for col in self.cat_cols if col in df.columns and col not in self.lender_cols]
        if existing_lenders:
            df['lender_id'] = df[existing_lenders].idxmax(axis=1)
            if pd.api.types.is_string_dtype(df['lender_id']):
                df['lender_id'] = df['lender_id'].str.replace('is_', '', regex=False)
            for lender in self.lender_cols:
                if lender in df.columns:
                    df[f'is_{lender}'] = (df[lender] == 1).astype(int)
            df.drop(columns=existing_lenders, inplace=True)
        
        if other_flags:
            for flag in other_flags:
                if flag in df.columns:
                    df[f'{flag}_flag'] = (df[flag] == 1).astype(int)
        
        return df
    
    def _apply_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log/log1p identities defined in YAML."""
        df = X.copy()
        for alias, (func, source) in self.transforms.items():
             if source in df.columns:
                if func == "log1p":
                    df[alias] = np.log1p(df[source])
                elif func == "log":
                    clipped = df[source].clip(lower=self.eps)
                    df[alias] = np.log(clipped)
                else:
                    logger.warning(f"Unsupported transform: {func}")
             else:
                logger.warning(f"Source {source} missing for {alias}")
        return df

    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derives multi-order interaction terms"""
        df = X.copy()
        created = []

        for key, components in self.interactions.items():
            existing = [col for col in components if col in df.columns]
            if len(existing) == len(components):
                multiplicative_terms = list(components)
                df[key] = df[multiplicative_terms].prod(axis=1)
                created.append(key)
            else:
                missing = set(components) - set(existing)
                logger.warning(f"Skipping: {key} -- missing: {set(components) - set(existing)}")
        
        return df, created

    def fit(self, X: pd.DataFrame, y=None):
        """Learns scaling parameters using sklearn scalers"""
        #--Orchestration--#
        self.stats_ = {} #-reset state-#
        
        df = X.copy()

        available_ids = [id for id in self.id_cols if id in df.columns]
        df = df.drop(columns=available_ids, errors='ignore')

        #-dynamic transforms-#
        df = self._to_z_score(df, fit_mode=True)
        df = self._to_logit_z(df, fit_mode=True)

        #-new features-#
        df = self._apply_transforms(df)
        df, self.features_out_ = self._create_interactions(df)

        #-static transforms-#
        df = self._to_log(df)
        df = self._create_binary_flags(df)
        
        df = self._drop_unwanted(df)
        df = self._final_cleanup(df)

        numeric_df = df.select_dtypes(include=[np.number])
        self.numeric_cols_out_ = [c for c in numeric_df.columns if c not in self.id_cols]
        
        #--Scaler as per config---#
        scalers = {
            "power": PowerTransformer(method='yeo-johnson'),
            "robust": RobustScaler(),
            "standard": StandardScaler()
        }
        
        self.scaler_ = scalers.get(self.scaling_method, PowerTransformer(method='yeo-johnson'))
        
        #-fit scaler on newly engineered terms under "_create_interactions()" --#
        if self.numeric_cols_out_:
            logger.info(f"Fitting final {self.scaling_method} scaler on {len(self.numeric_cols_out_)} features.")
            self.scaler_.fit(df[self.numeric_cols_out_])
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Orchestration module"""
        #-execute feature engineering steps using saved stats-#
        check_is_fitted(self, ['scaler_', 'features_out_', 'stats_'])
        df = X.copy()

        passthroughs = [id for id in self.id_cols if id in df.columns]
        target_list = self.y_target if isinstance(self.y_target, list) else [self.y_target]
        passthroughs.extend([t for t in target_list if t in df.columns])
        
        df_pass = df[passthroughs].copy()
        
        df = df.drop(columns=passthroughs, errors='ignore')

        #-feature eng-#
        df = self._to_z_score(df, fit_mode=False)
        df = self._to_logit_z(df, fit_mode=False)

        df = self._apply_transforms(df)

        df, _ = self._create_interactions(df)

        df = self._to_log(df)
        df = self._create_binary_flags(df)

        df = self._drop_unwanted(df)
        df = self._final_cleanup(df)
        
        missing = set(self.numeric_cols_out_) - set(df.columns)
        if missing:
            logger.warning(f"Transform input missing {len(missing)} cols. Filling 0.")
            for c in missing:
                df[c] = 0.0
        
        if self.numeric_cols_out_:
            scaled_vals = self.scaler_.transform(df[self.numeric_cols_out_])
            df_scaled = pd.DataFrame(scaled_vals, columns=self.numeric_cols_out_, index=df.index)
        else:
            df_scaled = df
        
        df_final = pd.concat([df_pass, df_scaled], axis=1)
        
        return df_final
    
    def _final_cleanup(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Systematically removes raw features when transformed versions are present.
        Also cleans up high-order interaction noise and string leakage.
        """
        df = df_in.copy()
        drop_list = []
  
        potential_dupes = self.bisg_cols + self.pop_cols + self.seg_cols + self.to_log_transform + self.cat_cols
        for col in potential_dupes:
            if col in df.columns:
                if f"{col}_logit_z" in df.columns or f"{col}_logit" in df.columns or f"{col}_z" in df.columns or f"log_{col}" in df.columns:
                    drop_list.append(col)
        
        noise_patterns = [
            'lg_bifsg', 's_bifsg', 'sg_lg_bifsg', 'sg_s_bifsg', 'lg_s_bifsg', 'lg_bifsg_lsg',
            'lg_s_bifsg',  'comp_int_3', 's_bifsg_lsg', 'amountsought', 'sg_lg_s',
            'sg_bifsg_lsg', 'factoringccmca', 'has_masters', 'has_postgrad', 'is_ever_ceo', 'mdi_z'
        ] #-added amountsought because log_amountsought is in the original df -- this also includes ghost cols

        drop_list.extend([c for c in noise_patterns if c in df.columns])
             
        object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        drop_list.extend([c for c in object_cols if c not in self.id_cols])

        drop_list = list(set(drop_list))
        
        if drop_list:
            logger.info(f"Final Cleanup: Dropping {len(drop_list)} redundant/noisy features.")
            df = df.drop(columns=drop_list, errors='ignore')
            
        return df