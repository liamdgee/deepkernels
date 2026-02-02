from sklearn.pipeline import Pipeline
from src.preprocess import cleaner, lasso, preprocess, harmoniser, novelty, inducing
from pydantic import Field, BaseModel

class DataOrchestrator(BaseModel):
    def __init__(self, cleanconfig=None, harmoniseconfig=None, featureconfig=None, induceconfig=None, processconfig=None):
        self.cleanconfig = cleanconfig or cleaner.CleanerConfig()
        self.harmoniseconfig = harmoniseconfig or harmoniser.HarmoniserConfig()
        self.featureconfig = featureconfig or lasso.FeatureEngConfig()
        self.induceconfig = induceconfig or inducing.InducingConfig()
        self.processconfig = processconfig or preprocess.PreprocessConfig()
    
    def _run_processes(
            self, **overrides
        ):
        arch = [
            ('cleaning', overrides.get('clean', cleaner.DataCleaner(self.cleanconfig))),
            ('harmonising', overrides.get('harmonise', harmoniser.SchemaHarmoniser(self.harmoniseconfig))),
            ('feat_eng', overrides.get('custom', novelty.FeatureTransformer(self.featureconfig))),
            ('feat_select', overrides.get('select', lasso.LassoFeatures(self.featureconfig))),
            ('inducing_points', overrides.get('induce', inducing.InducingPointSelect(self.induceconfig))),
            ('torch_process', overrides.get('process', preprocess.TorchPreprocessor(self.processconfig)))
        ]
        return Pipeline(arch)



