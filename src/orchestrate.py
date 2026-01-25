import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.linear_model import LogisticRegressionCV

#--Preperatory Pipeline--#
from src import cleaner, harmoniser, novelty, lasso, preprocess, model, train, pred, integrate, audit, prune

def Orchestrate(config):
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

    pruner = prune.DirichletPruner(**config['training']) #--not ready--#

    predict = pred.Inference(**config['inference']) #--not ready--#

    auditor = audit.StatisticalAuditor(**config['audit']) #--not ready--#

    simpson = integrate.Engine(**config['inference']) #-- not ready--#

    #-note 
