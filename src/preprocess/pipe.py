from sklearn.pipeline import Pipeline
from src.preprocess import cleaner, lasso, preprocess, harmoniser, novelty, inducing

class DataOrchestrator:
    def __init__(self, config):
        self.config = config
    def _run_processes(self, config):
        clean = cleaner.DataCleaner(config=cleaner.CleanerConfig())
        harmonise = harmoniser.SchemaHarmoniser(config=harmoniser.HarmoniserConfig()) #-- i dont think harmonise currently exists in yaml--#
        custom = novelty.FeatureTransformer(config=lasso.FeatureEngConfig())
        select = lasso.LassoFeatures(config=lasso.FeatureEngConfig())
        process = preprocess.TorchPreprocessor(config=preprocess.PreprocessConfig())
        induce = inducing.InducingPointSelect(config=inducing.InducingConfig)

        pipe = Pipeline([
            ('individual_df_cleaning', clean),
            ('harmonise_schemas', harmonise),
            ('feature_engineering', custom),
            ('stat_feature_select', select),
            ('inducing_point_select', induce),
            ('torch_tensor_format', process)
        ])

        return pipe



