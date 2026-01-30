# filename: model.py

#---Dependencies---#
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P
import gpytorch as gp
from gpytorch.models import ApproximateGP as agp
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#---Deep Feature Network---#
#---A deep, spectral-normalized MLP for feature extraction---#
class DeepFeatureNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=512, depth=4, use_spectral_norm=True):
        super().__init__()
        layers = []
        layer_dim = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        for i in range(len(layer_dim) - 1):
            dim_in = layer_dim[i]
            dim_out = layer_dim[i+1]
            #--Linear Transformation---#
            linear_layer = nn.Linear(dim_in, dim_out)
            #---Spectral Normalization---#
            #---Normalise the weight matrix W by its largest singular value [sigma(W)]---#
            #---Ensures |Wx| <= |x|, ensuring stable geometry for downstream GP---#
            if use_spectral_norm:
                linear_layer = P.spectral_norm(linear_layer)
            layers.append(linear_layer)
            #---Nonlinear Activation + BatchNorm + Dropout---#
            last_layer_flag = (i == len(layer_dim) - 2)
            if not last_layer_flag:
                layers.append(nn.GELU()) #---GELU Activation prevents dead neurons---#
                layers.append(nn.BatchNorm1d(dim_out)) #---Ensures spectral norm doesn't squash signals---#
                layers.append(nn.Dropout(0.1)) #---Prevent overfitting---#
            self.network = nn.Sequential(*layers)
    #---Forward Pass---#
    def forward(self, x):
        return self.network(x)

#---Gaussian Process Layer---#
class LatentSVGP(agp):
    def __init__(self, input_dim, num_classes, num_inducing=128):
        
        #---Variational Distribution q(u) ~ N(m, S)---#
        #---Produces batch of Gaussian distributions for each class--#
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, 
            batch_shape=torch.Size([num_classes]) 
        )
        
        #---Learn Inducing Points---#
        inducing_points = torch.randn(num_classes, num_inducing, input_dim)
        
        #---Variational Strategy q(f)---#
        variational_strategy = VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        #---Mean Module: Constant Mean acts as logit bias term---#
        self.mean_module = gp.means.ConstantMean(batch_shape=torch.Size([num_classes]))
        
        #---Covariance Module: Linear Kernel + Rational Quadratic (RQ) Kernel with ARD---#

        #---KERNEL COMPONENT A: "Linear Kernel" captures global trends in latent space---#
        #---No Automatic Relevance Determination (ARD) for Linear Kernel as we want all dims to contribute---#
        linear_kernel = gp.kernels.ScaleKernel(gp.kernels.LinearKernel(batch_shape=torch.Size([num_classes])), batch_shape=torch.Size([num_classes]))
        
        #---KERNEL COMPONENT B: "Rational Quadratic (RQ) Kernel with ARD" captures local variations---#
        #---RQ Kernel equivalent to an infinite sum of RBF kernels with varying lengthscales---#
        #---ARD allows GP to learn relevance of each latent dimension and ignore noise dimensions in projection---#
        rq_kernel = gp.kernels.ScaleKernel(
            gp.kernels.RQKernel(
                ard_num_dims=input_dim,
                batch_shape=torch.Size([num_classes])
            ),
            batch_shape=torch.Size([num_classes])
        )
        rq_kernel.base_kernel.lengthscale = 1.0

        #---Combined Covariance Module: Kernel Component A (Linear) + Kernel Component B (RQ)---#
        self.covar_module = linear_kernel + rq_kernel
    
    #---Forward Pass---#
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    
class KernelClassifier(gp.Module):
    def __init__(self, deep_extractor, input_dim, num_classes, gp_latent_dim=64, num_inducing=128):
        
        super().__init__()

        #---Initialise Deep Feature Extractor---#
        self.feature_extractor = deep_extractor
        
        #---Linear MLP Projector---#
        self.projector = nn.Sequential(
            nn.Linear(input_dim, gp_latent_dim), #LAYER 1: Compress high-dim features
            nn.BatchNorm1d(gp_latent_dim, momentum=0.01) #LAYER 2: BatchNorm for Kernel stability
        )
        
        #---Gaussian Process Layers---#
        #---Sparse Variational GP (SVGP)---#
        #---Inducing Points in 64-Dim Latent Space---#
        #---LatentSVGP Class defined seperately---#
        #---Maps the projected manifold to Class Logits---#
        self.gp_layer = LatentSVGP(
            input_dim=gp_latent_dim,
            num_classes=num_classes,
            num_inducing=num_inducing
        )

    #---Forward Pass---#
    def forward(self, x):
        #---Extract Deep Features---#
        features = self.feature_extractor(x)
        
        #---Project Features to Manifold---#
        projected_features = self.projector(features)
        
        #---Gaussian Process Inference---#
        #---Output is a multivariate normal distribution over class logits---#
        logits = self.gp_layer(projected_features)
        
        return logits