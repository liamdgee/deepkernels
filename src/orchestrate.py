import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.linear_model import LogisticRegressionCV

#--Preperatory Pipeline--#
from src.train import train
from src.infer import audit, integrate, pred
from src.models import model
from src.preprocess.preprocess import cleaner, lasso, preprocess
from src.preprocess_data import harmoniser, novelty
from src.utils import dirichlet_pruner


class MasterOrchestrator:
    def __init__(self, config):
        self.config = config
    def _components(self, config):
        clean = cleaner.DataCleaner(config=cleaner.CleanerConfig(**config['clean']))
        harmonise = harmoniser.SchemaHarmoniser(config=harmoniser.HarmoniserConfig(**config['harmonise'])) #-- i dont think harmonise currently exists in yaml--#
        custom = novelty.FeatureTransformer(config=lasso.FeatureEngConfig(**config['features']))
        select = lasso.LassoFeatures(config=lasso.FeatureEngConfig(**config['lasso']))
        #-- not prod ready below this line--#
        process = preprocess #-- not ready--#

        #--Model arch--#
        transformer = model.VisionTransformerFeatureExtractor(**config['model'])
        gp = model.StatelessWoodburyRandomFourierGaussianProcess(**config['model'])
        decoder = model.ReproducingKernelHilbertSpaceDecoder(**config['model'])
        dirichlet = model.HierarchicalDirichletProcess(**config['model'])

        mod = model.InfiniteGaussianMixtureModel(**config['model'])

        trainer = train.LangevinTrainer(**config['training']) #--not ready--#

        pruner = dirichlet_pruner.DirichletPruner(**config['training']) #--not ready--#

        predict = pred.Inference(**config['inference']) #--not ready--#

        auditor = audit.StatisticalAuditor(**config['audit']) #--not ready--#

        simpson = integrate.Engine(**config['inference']) #-- not ready--#

        data_pipe = Pipeline([
            ('clean', clean),
            ('harmonise', harmonise),
            ('custom_features', custom),
            ('select_features', select),
            ('preprocess_for_torch', process)
        ])

        model_init_pipe = Pipeline([
            ('ViTb_16_transformer', transformer),
            ('sparse_gaussian_process', gp),
            ('hierarchical_dirichlet_process', dirichlet),
            ('reproducing_kernel_decoder', decoder),
            ('full_model', mod)
        ])

        training_pipe = Pipeline([
            ('training_module', trainer),
            ('prune_dirichlet_clusters', pruner),
            ('audit_model_efficacy', auditor)
        ])

        inference_pipe = Pipeline([
            ('predict', predict),
            ('integrate_posterior', simpson)
        ])

        #-note 
