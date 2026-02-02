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
def init_inducing_points(self, z, n_inducing=model_config.GPConfig.num_inducing):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_inducing, n_init=3).fit(z.cpu().numpy())
    centroids= torch.tensor(km.cluster_centers_).float()
    return centroids #-dummy class with basic logic-#


from src.models import dirichlet_variational, resnet, hypernetwork, lmc, snvgp_layer
from src.kernels import dynamic_spectral, NTK, weighted_rff_kernel

custom_kernels = {
    "dyn_sm": dynamic_spectral.DynamicEigenKernel,
    "ntk": NTK.SparseNTK,
    "kron_rff": weighted_rff_kernel.CrossCovarRFFKernel
}

custom_modules = {
    "feature_extractor": resnet.ResNet, 
    "hypernet": hypernetwork.SpectralEncoder,
    "dirichlet": dirichlet_variational.VariationalDirichlet, 
    "lmc": lmc.LinearCoregionalisation,
    "specnormgp": snvgp_layer.SpectralVariationalGaussianProcess
}

class DeepMultiTaskGaussianProcess(ApproximateGP):
    def __init__(self, 
            inducing_points, module_dict=custom_modules,
            kernel_dict=custom_kernels, num_latents=4, latent_dim=128
    ):
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=
        )
