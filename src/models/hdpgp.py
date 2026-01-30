import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import IndependentMultitaskVariationalStrategy, VariationalStrategy
import torch.nn as nn

from src.models.dirichlet_variational import VariationalDirichlet as hdp
from src.kernels.weighted_rff_kernel import CrossCovarRFFKernel as rffkrnl
from src.models.model_config import RootConfig as root
from src.models.transformer import VisionTransformerFeatureExtractor as vit

class HDPGP(ApproximateGP):
    def __init__(self, config: root, deep_network: vit, dp_module: hdp, kernel: rffkrnl, num_inducing=1024):
        self.config = config
        self.backbone = deep_network
        self.dirichlet = dp_module
        self.num_inducing = num_inducing

        self.output_dim = dp_module.D #-D tasks-#
        self.inducing_points_init = torch.randn(self.num_inducing, self.output_dim)
        self.inducing_points = nn.Parameter(self.inducing_points_init)

        variational_dist = CholeskyVariationalDistribution(num_inducing_points=self.num_inducing, batch_shape=torch.Size([self.output_dim]))

        inner_strategy = VariationalStrategy(self, self.inducing_points, variational_dist, learn_inducing_locations=True)

        self.outer_strategy = IndependentMultitaskVariationalStrategy(inner_strategy, num_tasks=self.output_dim, num_classes=None)

        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ZeroMean(), num_tasks=self.output_dim)

        self.covar_module = rffkrnl(self.dirichlet)
    
    def forward(self, x):
        z = self.backbone(x)

        pi, beta, omega, bias = self.dirichlet(z=z)
        
        mean_x = self.mean_module(z)

        dir_params = {'variational_dirichlet_output': (pi, beta, omega, bias)}

        covar_x = self.covar_module(z, **dir_params)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)