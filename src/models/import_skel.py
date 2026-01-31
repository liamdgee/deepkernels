#-dependencies--#
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from torch.distributions import Normal, RelaxedOneHotCategorical, Transform
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import math
from gpytorch.models import ApproximateGP

from src.models import model_config

#-init config-#
def init_inducing_points(self, train_x, n_inducing=model_config.GPConfig.num_inducing):
    self.eval()
    with torch.no_grad():
        
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_inducing, n_init=3).fit(z.cpu().numpy())
    centroids= torch.tensor(km.cluster_centers_).float()


from src.models import dirichlet_variational, resnet, hypernetwork, lmc
from src.kernels import dynamic_spectral, NTK, weighted_rff_kernel

custom_kernels = {
    "dyn_sm": dynamic_spectral.DynamicEigenKernel,
    "ntk": NTK.SparseNTK,
    "kron_rff": weighted_rff_kernel.CrossCovarRFFKernel
}

class DeepMultiTaskGaussianProcess(ApproximateGP):
    def __init__(self, 
            inducing_points, feature_extractor=resnet.ResNet, hypernet=hypernetwork.SpectralEncoder,
            dirichlet_module=dirichlet_variational.VariationalDirichlet, lmc_module=lmc.LinearCoregionalisation, 
            kernel_dict=custom_kernels, num_latents=4, latent_dim=128
    ):
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=
        )
