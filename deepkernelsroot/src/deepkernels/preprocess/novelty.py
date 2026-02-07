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
TransformMap: TypeAlias = dict[str, tuple[str, str]]
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
            base_dir = Path(__file__).resolve().parent.parent.parent
        except NameError:
            base_dir = Path.cwd()

        config_path = base_dir / filename

        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}. Using internal defaults.")
            return {}
        
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
     #--logic: {'alias_for_pipeline': ['interaction_base', 'col_source_name']}-#
    default_transforms: TransformMap = {
        'l1_g' : ('log1p', 'black_g_pct'),
        'lfs': ('log', 'black_fs_pct'),
        'lsg': ('log', 'black_sg_pct'),
        'l1_bifsg': ('log1p', 'black_bifsg_pct'), 
        's': ('1', 'black_s_pct')
    }

    override_transforms: Optional[TransformMap] = None
    
    #---logic: [new_feature, [i_term_1, i_term_2, ..., i_term_n]]
    default_interactions: CustomInteractionMap = {
        'lg_bifsg': ['l1_g', 'black_bifsg_pct'], 
        'sg_lsg': ['black_sg_pct', 'lsg'], 
        'lg_lsg': ['l1_g', 'lsg'], 
        's_lsg': ['black_s_pct', 'lsg'], 
        'bifsg_lsg': ['black_bifsg_pct', 'lsg'], 
        's_bifsg': ['black_s_pct', 'black_bifsg_pct'], 
        'sg_s': ['black_sg_pct', 'black_s_pct'], 
        'lg_s': ['l1_g', 'black_s_pct'], 
        'lg_s_bifsg': ['l1_g', 'black_s_pct', 'black_bifsg_pct'], 
        'sg_s_bifsg': ['black_sg_pct', 'black_s_pct', 'black_bifsg_pct'], 
        'sg_lg_bifsg': ['black_sg_pct', 'l1_g', 'black_bifsg_pct'], 
        'sg_lg_s': ['black_sg_pct', 'l1_g', 'black_s_pct'], 
        'lg_bifsg_lsg': ['l1_g', 'black_bifsg_pct', 'lsg'], 
        's_bifsg_lsg': ['black_s_pct', 'black_bifsg_pct', 'lsg'], 
        'sg_bifsg_lsg': ['black_sg_pct', 'black_bifsg_pct', 'lsg'], 
        'comp_int_1': ['black_sg_pct', 'l1_g', 'black_s_pct', 'black_bifsg_pct', 'lsg'], 
        'comp_int_2': ['l1_g', 'black_s_pct', 'black_bifsg_pct', 'lsg'], 
        'comp_int_3': ['black_sg_pct', 'l1_g', 'black_bifsg_pct', 'lsg']
    }

    y_target: str = 'lmean_rejected'

    default_drop_cols: list[str] = [
            'shr_app_black_final_race', 'shr_app_white_final_race',
            'shr_appr_black_final_race', 'shr_appr_white_final_race',
            'shr_rej_black_final_race', 'shr_rej_white_final_race', 'amountfunded', 
            'approved_all', 'lmean_approved_all', 'rejected',
            'shr_appr_black_sg_cont', 'shr_appr_white_sg_cont', 'lmean_amountfunded',
            'shr_rej_black_sg_cont', 'shr_rej_white_sg_cont', 'delta_shr_appr_bc',
            'delta_shr_appr_wc', 'delta_shr_loan_bc', 'delta_shr_loan_wc', 
            'delta_shr_rej_bc', 'delta_shr_rej_wc', 'delta_shr_rej_wc_w',
            'amountsought', 'lenders_sent_to', 'fintech_lenders_sent_to', 
            'non_fintech_lenders_sent_to', 'random_race_sg'
        ]

    default_id_cols: list[str] = ['lender_clean', 'time', 'unique_borrower', 'black_final_race'] #-create lender_id from lender_clean, drop time for non-TS models, create repeat borrower from unique borrower-#
    default_seg_cols: list[str] = ["dissim_scaled", "isolation_scaled", "animus_scaled", "iat_score_f_scaled", "mdi"] #-to z score-#
    default_bisg_cols: list[str] = ["black_s_pct", "black_g_pct", "black_fs_pct", "black_bifsg_pct", "black_sg_pct"] #-clip -> logits -> logit_z--#--mean_bisg-#
    default_pop_cols: list[str] = ["share_pop_black", "share_black_pop_geba"] #-clip -> logits -> logit_z--#
    default_outcome_cols: list[str] =["false_neg_black_bisg", "false_pos_black_bisg", "true_neg_black_bisg", "true_pos_black_bisg"] #-flags-#
    default_lender_cols: list[str] = ["fintech", "cdfi", "creditunion", "bank"] #-type to flag -> create lender_id column-#
    default_to_log_transform: list[str] = ['amountsought', 'total_percap_inc'] #-also to z-#
    default_shr_cols: list[str] = [
        "shr_loan_black_final_race",
        "shr_loan_black_sg_cont",
        "shr_loan_white_final_race",
        "shr_loan_white_sg_cont",
        "shr_app_black_sg_cont", 
        "shr_app_white_sg_cont"
    ]

    override_interactions: Optional[CustomInteractionMap] = None

    #-scaling params-#
    scaling_method: Literal['power', 'standard', 'robust'] = 'power'
    eps: Annotated[float, Field(ge=1e-14, le=1e-2)] = 1e-8



#--Class: Feature Engineering Pipeline--#
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            config: Optional[NoveltyConfig] = None,
            transforms: Optional[TransformMap] = None,
            interactions: Optional[CustomInteractionMap] = None,
            scaling_method: Literal['power', 'standard', 'robust'] = 'power',
            eps: float = 1e-8,
            drop_cols: Optional[List[str]] = None,
            id_cols: Optional[List[str]] = None,
            seg_cols: Optional[List[str]] = None,
            bisg_cols: Optional[List[str]] = None,
            pop_cols: Optional[List[str]] = None,
            outcome_cols: Optional[List[str]] = None,
            lender_cols: Optional[List[str]] = None,
            to_log_transform: Optional[List[str]] = None,
            shr_cols: Optional[List[str]] = None,
            **kwargs
        ):

        self.config = config if config else NoveltyConfig()
        default_interactions = {
            'lg_bifsg': ['l1_g', 'black_bifsg_pct'], 
            'sg_lsg': ['black_sg_pct', 'lsg'], 
            'lg_lsg': ['l1_g', 'lsg'], 
            's_lsg': ['black_s_pct', 'lsg'], 
            'bifsg_lsg': ['black_bifsg_pct', 'lsg'], 
            's_bifsg': ['black_s_pct', 'black_bifsg_pct'], 
            'sg_s': ['black_sg_pct', 'black_s_pct'], 
            'lg_s': ['l1_g', 'black_s_pct'], 
            'lg_s_bifsg': ['l1_g', 'black_s_pct', 'black_bifsg_pct'], 
            'sg_s_bifsg': ['black_sg_pct', 'black_s_pct', 'black_bifsg_pct'], 
            'sg_lg_bifsg': ['black_sg_pct', 'l1_g', 'black_bifsg_pct'], 
            'sg_lg_s': ['black_sg_pct', 'l1_g', 'black_s_pct'], 
            'lg_bifsg_lsg': ['l1_g', 'black_bifsg_pct', 'lsg'], 
            's_bifsg_lsg': ['black_s_pct', 'black_bifsg_pct', 'lsg'], 
            'sg_bifsg_lsg': ['black_sg_pct', 'black_bifsg_pct', 'lsg'], 
            'comp_int_1': ['black_sg_pct', 'l1_g', 'black_s_pct', 'black_bifsg_pct', 'lsg'], 
            'comp_int_2': ['l1_g', 'black_s_pct', 'black_bifsg_pct', 'lsg'], 
            'comp_int_3': ['black_sg_pct', 'l1_g', 'black_bifsg_pct', 'lsg']
        }

        default_transforms ={
            'log1p' : ['l1_g', 'black_g_pct'],
            'log': ['lfs', 'black_fs_pct'],
            'log': ['lsg', 'black_sg_pct'],
            'log1p' : ['l1_bifsg', 'black_bifsg_pct']
        }

        y_target = 'lmean_rejected'

        default_drop_cols = [
            'shr_app_black_final_race', 'shr_app_white_final_race',
            'shr_appr_black_final_race', 'shr_appr_white_final_race',
            'shr_rej_black_final_race', 'shr_rej_white_final_race', 'amountfunded', 
            'approved_all', 'lmean_approved_all',
            'shr_appr_black_sg_cont', 'shr_appr_white_sg_cont', 'lmean_amountfunded',
            'shr_rej_black_sg_cont', 'shr_rej_white_sg_cont', 'delta_shr_appr_bc',
            'delta_shr_appr_wc', 'delta_shr_loan_bc', 'delta_shr_loan_wc', 
            'delta_shr_rej_bc', 'delta_shr_rej_wc', 'delta_shr_rej_wc_w',
            'amountsought', 'lenders_sent_to', 'fintech_lenders_sent_to', 
            'non_fintech_lenders_sent_to', 'random_race_sg'
        ]

        default_id_cols = ['lender_clean', 'time', 'unique_borrower', 'black_final_race'] #-create lender_id from lender_clean, drop time for non-TS models, create repeat borrower from unique borrower-#
        default_seg_cols = ["dissim_scaled", "isolation_scaled", "animus_scaled", "iat_score_f_scaled", "mdi"] #-to z score-#
        default_bisg_cols = ["black_s_pct", "black_g_pct", "black_fs_pct", "black_bifsg_pct", "black_sg_pct"] #-clip -> logits -> logit_z--#--mean_bisg-#
        default_pop_cols = ["share_pop_black", "share_black_pop_geba"] #-clip -> logits -> logit_z--#
        default_outcome_cols =["false_neg_black_bisg", "false_pos_black_bisg", "true_neg_black_bisg", "true_pos_black_bisg", 'black_final_race'] #-flags-#
        default_lender_cols = ["fintech", "cdfi", "creditunion", "bank"] #-type to flag -> create lender_id column-#
        default_to_log_transform = ['amountsought', 'total_percap_inc'] #-also to z-#
        default_shr_cols = [
            "shr_loan_black_final_race",
            "shr_loan_black_sg_cont",
            "shr_loan_white_final_race",
            "shr_loan_white_sg_cont",
            "shr_app_black_sg_cont", 
            "shr_app_white_sg_cont"
        ] #-clip -> logits -> logit_z--#

        self.drop_cols = drop_cols if drop_cols is not None else default_drop_cols
        self.id_cols = id_cols if id_cols is not None else default_id_cols
        self.seg_cols = seg_cols if seg_cols is not None else default_seg_cols
        self.bisg_cols = bisg_cols if bisg_cols is not None else default_bisg_cols
        self.pop_cols = pop_cols if pop_cols is not None else default_pop_cols
        self.outcome_cols = outcome_cols if outcome_cols is not None else default_outcome_cols
        self.lender_cols = lender_cols if lender_cols is not None else default_lender_cols
        self.to_log_transform = to_log_transform if to_log_transform is not None else default_to_log_transform
        self.shr_cols = shr_cols if shr_cols is not None else default_shr_cols
        self.y_target = y_target or self.config.y_target or 'lmean_rejected'
        self.transforms = transforms if transforms is not None else default_transforms
        self.interactions = interactions if interactions is not None else default_interactions
        self.eps = eps or self.config.eps or 1e-8
        self.scaling_method = scaling_method or self.config.scaling_method or 'power'
        self.scaler_ = None
        self.stats_ = {}
        self.features_out_ = None
        self.numeric_cols_out = []
        self.df_ids = None
    
    def _to_log(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Helper function to log transform specified columns."""
        df = df_in.copy()
        for col in self.to_log_transform:
            if col in df.columns:
                clipped = df[col].clip(lower=self.eps) #-clip to avoid log(0) and log of negatives-#
                df[f'log_{col}'] = np.log(clipped)
            else:
                logger.warning(f"Column {col} not found for log transformation.")
        return df
    
    def _drop_unwanted(self, df: pd.DataFrame) -> pd.DataFrame:
            """Explicitly drops ID and Ignore columns."""
            cols_to_drop = list(set(self.drop_cols + self.id_cols))
            dropping = [c for c in cols_to_drop if c in df.columns]
            
            if dropping:
                logger.info(f"Dropping {len(dropping)} columns explicitly.")
                df = df.drop(columns=dropping)
                
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
                logger.warning(f"Column {col} not found for z-score scaling.")
        return df

    def _to_logit_z(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """
        Applies Logit-Z.
        If fit_mode=True, it CALCULATES and SAVES stats.
        """
        for col in self.bisg_cols + self.pop_cols + self.shr_cols:
            if col in df.columns:
                clipped = df[col].clip(self.eps, 1 - self.eps)
                logit_vals = np.log(clipped / (1 - clipped))
                if fit_mode:
                    #-LEARN-#
                    self.stats_[f'{col}_logit'] = {
                        'mean': logit_vals.mean(),
                        'std': logit_vals.std() + self.eps
                    }       
                #-APPLY-#
                stats_key = f'{col}_logit'
                if stats_key in self.stats_:
                    mu = self.stats_[stats_key]['mean']
                    sigma = self.stats_[stats_key]['std']
                    df[f'{col}_logit_z'] = (logit_vals - mu) / sigma
            else:
                logger.warning(f"Column {col} not found for logit z transformation.")
        return df
    
    def _create_binary_flags(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Helper function to create binary flags for lender types and outcome variables."""
        df = df_in.copy()
        for lender in self.lender_cols:
            if lender in df.columns:
                logger.info(f"Creating binary flag for lender type: {lender}")
                df[f'is_{lender}'] = (df[lender] == 1).astype(int)
                df['lender_id'] = df[self.lender_cols].idxmax(axis=1).str.replace('is_', '', regex=False)
                df.drop(columns=self.lender_cols, inplace=True)
            else:
                logger.warning(f"Lender column {lender} not found for binary flag creation.")
        
        for outcome in self.outcome_cols:
            if outcome in df.columns:
                logger.info(f"Ensuring binary flag for outcome variable: {outcome}")
                df[f'{outcome}_flag'] = (df[outcome] == 1).astype(int)
                df.drop(columns=[outcome], inplace=True)
            else:
                logger.warning(f"Outcome column {outcome} not found for binary flag creation.")
        
        return df

    def _apply_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the log/log1p identities defined in YAML."""
        df = X.copy()
        for transform, (alias, source) in self.transforms.items():
             logger.info(f"Applying transform: {transform} to source column: {source} with alias: {alias}")
             if source in df.columns:
                if transform == "log1p":
                    df[alias] = np.log1p(df[source])
                elif transform == "log":
                    clipped = df[source].clip(lower=self.eps) #-clip to avoid log(0) and log of negatives-#
                    df[alias] = np.log(clipped)
                else:
                    logger.warning(f"Unsupported transform function: {transform} for alias: {alias}. Skipping this transform.")
             else:
                logger.warning(f"Source column {source} not found for transform: {transform} with alias: {alias}. Skipping this transform.")
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
        self.stats_ = {} #-reset state-#
        df = X.copy()
        self.available_ids = [c for c in self.id_cols if c in df.columns]
        self.df_ids = df[self.available_ids].copy()

        #-dynamic transforms-#
        df = self._to_z_score(df, fit_mode=True)
        df = self._to_logit_z(df, fit_mode=True)

        #-static transforms-#
        df = self._to_log(df)
        df = self._create_binary_flags(df)
        df = self._apply_transforms(df)

        #-interactions-#
        df_engineered, cols_created = self._create_interactions(df)
        self.features_out_ = cols_created
        df = self._drop_unwanted(df_engineered)

        self.numeric_cols_out_ = df.select_dtypes(include=[np.number]).columns.tolist()
        
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

    def transform(self, X):
        """Orchestration module"""
        #-execute feature engineering steps using saved stats-#
        check_is_fitted(self, ['scaler_', 'features_out_', 'stats_'])
        df = X.copy()
        self.available_ids = [c for c in self.id_cols if c in df.columns]
        self.df_ids = df[self.available_ids].copy()

        df = self._to_z_score(df, fit_mode=False)
        df = self._to_logit_z(df, fit_mode=False)
        df = self._to_log(df)
        df = self._create_binary_flags(df)
        df = self._apply_transforms(df)

        #-interaction terms-#
        df_engineered, _ = self._create_interactions(df)
        df_engineered = self._drop_unwanted(df_engineered)
        
        missing_cols = set(self.numeric_cols_out_) - set(df_engineered.columns)
        if missing_cols:
            logger.warning(f"Missing columns in transform: {missing_cols}. Filling with 0.")
            for c in missing_cols:
                df_engineered[c] = 0.0
    
        if self.numeric_cols_out_:
            df_engineered[self.numeric_cols_out_] = self.scaler_.transform(df_engineered[self.numeric_cols_out_])
            df_engineered = df_engineered[self.numeric_cols_out_]
        
        logger.info(f"Re-attaching {len(self.available_ids)} ID columns to output.")
        df_final = pd.concat([self.df_ids, df_engineered], axis=1)
        
        return df_final