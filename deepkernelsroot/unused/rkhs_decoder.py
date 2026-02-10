import torch
import torch.nn as nn

from src.models.model_config import RootConfig

class ReproducingKernelHilbertSpaceDecoder(nn.Module):
    def __init__(self, config: RootConfig):
        super().__init__()
        self.config = config

        #-dimensions-#
        self.input_dim = self.config.model.gp.bottleneck_dim #-from gp-#
        self.dim_out = self.config.model.rkhs.transformer_out_dim #-matches ViT output-#
        self.n_anchor_points = self.config.model.rkhs.n_anchors
        self.register_buffer('eps', torch.tensor(getattr(config.model, 'eps', 1e-7)))

        #---Learnable Anchor Points on Latent Manifold Z---#
        self.anchors = nn.Parameter(torch.randn(self.n_anchor_points, self.input_dim))

        #---Kernel Hyperparameters (Gaussian RBF Kernel)---#
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))

        #--Projection weights--#
        self.weights_out = nn.utils.spectral_norm(nn.Linear(self.n_anchor_points, self.dim_out, bias=False))
    
    def compute_kernel(self, z):
        """Computes RBF Kernel between two sets of points. z: [batch, n_anchor_points]"""
        lengthscale = torch.exp(self.log_lengthscale) + self.eps
        dist_sq  = torch.cdist(z, self.anchors, p=2)**2 #--power inside bracket calcs euclidian distance
        kernel_scores = torch.exp(-0.5 * dist_sq / (lengthscale**2))
        return kernel_scores
    
    def forward(self, z, scores_out=False):
        """
        Reconstructs original vision transformer feature vectors from latent representations
        h_hat = sum_{i=1}^{n_anchor_points} alpha_i * k(z, anchor_point_i)

        z: [batch, bottleneck_dim] ; anchors: [n_anchor_points, bottleneck_dim] ; dist: [batch, n_anchor_points]
        """

        #---Compute manifold coordinates (kernel score)--
        Kz = self.compute_kernel(z)  #---[B, 256]---#

        #---Nyström Interpolation---#
        h_recon = self.weights_out(Kz) #--[B, 768]

        if scores_out:
            return h_recon, Kz

        return h_recon