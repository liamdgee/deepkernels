from sklearn.pipeline import Pipeline
from src.preprocess import cleaner, lasso, preprocess, harmoniser, novelty, inducing
from pydantic import Field, BaseModel
from src.preprocess import cleaner, lasso, preprocess, harmoniser, novelty, inducing

class ProcessConfig(BaseModel):
    clean: cleaner.CleanerConfig = Field(default_factory=cleaner.CleanerConfig)
    harmonise: harmoniser.HarmoniserConfig = Field(default_factory=harmoniser.HarmoniserConfig)
    feature: lasso.FeatureEngConfig = Field(default_factory=lasso.FeatureEngConfig)
    induce: inducing.InducingConfig = Field(default_factory=inducing.InducingConfig)
    process: preprocess.PreprocessConfig = Field(default_factory=preprocess.PreprocessConfig)

class DataOrchestrator:
    def __init__(self, config=None):
        self.config = config if config is not None else ProcessConfig()
    
    def build_pipe(self, **overrides):

        arch = [
            ('cleaning', overrides.get('clean', cleaner.DataCleaner(self.config.clean))),
            ('harmonising', overrides.get('harmonise', harmoniser.SchemaHarmoniser(self.config.harmonise))),
            ('feat_eng', overrides.get('custom', novelty.FeatureTransformer(self.config.feature))),
            ('feat_select', overrides.get('select', lasso.LassoFeatures(self.config.feature))),
            ('inducing_points', overrides.get('induce', inducing.InducingPointSelect(self.config.induce))),
            ('torch_process', overrides.get('process', preprocess.TorchPreprocessor(self.config.process)))
        ]
        return Pipeline(arch)



