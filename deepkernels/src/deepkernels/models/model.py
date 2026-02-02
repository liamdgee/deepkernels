#---Dependencies--#
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights
import math
import logging

#---Init Logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


from src.models.model_config import TransformerConfig, DirichletConfig, GPConfig, RKHSConfig, RFFConfig, SpectralConfig, ModelConfig, RootConfig

#--- Class Definition: Feature Extractor---#
class VisionTransformerFeatureExtractor(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.latent_dim = self.config.latent_dim
        pretrained = self.config.pretrained
        freeze_vit = self.config.freeze_vit
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit_model = vit_b_16(weights=weights)
        self.vit_model.head = nn.Identity()
        if freeze_vit:
            for p in self.vit_model.parameters():
                p.requires_grad = False
        
        #---Projection Layer for RKHS---#
        self.proj_stack = nn.Sequential(
            nn.Linear(768, self.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(self.latent_dim)
        )

        #---Orthogonal init for RFF Convergence---#
        nn.init.orthogonal_(self.proj_stack[0].weight)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.proj_stack[0].weight.requires_grad):
            token = self.vit_model(x) #---[B, 768]---#
            z = self.proj_stack(token) #---[B, latent_dim]---#
            
            return z


#---Class Definition: Random Fourier Features in Woodbury GP O(nlogn) ---#
class StatelessWoodburyRandomFourierGaussianProcess(nn.Module):
    def __init__(self, config: GPConfig):
        super().__init__()
        self.config = config
        self.num_inducing = self.config.num_inducing #---n inducing points--#
        self.latent_dim = self.config.latent_dim #--Latent Dim (model endogenous fixed param)---#
        self.Z = nn.Parameter(torch.randn(self.num_inducing, self.latent_dim)) #--Local Inducing Points---#

    def _get_phi(self, x, mu, log_sigma):
        """Maps inputs to the spectral hilbert space using external params"""
        #---Reparameterisation Trick---#
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        omega = mu + sigma * eps
        #--Projection--#
        proj = x @ omega.t()
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return phi * math.sqrt(2.0 / self.M)

    def forward(self, x, mu_atom, log_sigma_atom, sigma_noise, y=None):
        #---Forward Pass through RFF---#
        #---Compute Random Fourier Features for inputs and inducing points---#
        phi_x = self._get_phi(x, mu_atom, log_sigma_atom)
        phi_z = self._get_phi(self.Z, mu_atom, log_sigma_atom)
        rff_dim = phi_x.size(-1)

        #---Update Woodbury Statistics---#
        prec = torch.eye(rff_dim, device=x.device) + (phi_z.t() @ phi_z) / sigma_noise

        #---Evidence Gathering---#
        if y is not None:
            prec = prec + (phi_x.t() @ phi_x) / sigma_noise
            target = (phi_x.t() @ y) / sigma_noise
        else:
            target = torch.zeros(rff_dim, 1, device=x.device)
        
        #---Solve for Posterior Weights---#
        L = torch.linalg.cholesky(prec + 1e-7 * torch.eye(rff_dim, device=x.device))
        w_post = torch.cholesky_solve(target, L)
        #---Predictive Mean---#
        mu_pred = phi_x @ w_post
        #---Predictive Variance---#
        v = torch.linalg.solve_triangular(L, phi_x.t(), upper=False)
        var_pred = torch.sum(v**2, dim=0).unsqueeze(-1) + sigma_noise
        
        return mu_pred, var_pred

class HierarchicalDirichletProcess(nn.Module):
    def __init__(self, config: DirichletConfig):
        super().__init__()
        self.config = config
        self.k_atoms = self.config.n_global
        self.j_tables = self.config.n_local
        self.latent_dim = self.config.latent_dim

        #---Spectral Atoms---#
        self.M = self.config.fourier_dim
        self.mu_atom = nn.Parameter(torch.randn(self.k_atoms, self.M, self.latent_dim) * 0.1)
        self.log_sigma_atom = nn.Parameter(torch.full((self.k_atoms, self.M, self.latent_dim), -2.0)) #---exp(-2.0) approx equal to 0.13

         #---Learnable Weights for non-parametric clustering---#
        #---'v' params determine final cluster assignments---#
        self.v_k = nn.Parameter(torch.randn(self.k_atoms - 1)) #--Global--#
        self.v_j = nn.Parameter(torch.randn(self.j_tables, self.k_atoms - 1)) #--local--#

        alpha0 = self.config.alpha
        gamma0 = self.config.gamma
        
        #--Learnable Params with gradient flow to tune cluster sparsity---#
        if self.config.learnable_params:
            self.alpha = nn.Parameter(torch.tensor(alpha0))
            self.gamma = nn.Parameter(torch.tensor(gamma0)) 
        #---Fixed Priors (fixed with buffer state)---#
        #---No gradient flow to the below params---#
        else:
            self.register_buffer('alpha', torch.tensor([float(alpha0)]))
            self.register_buffer('gamma', torch.tensor([float(gamma0)]))
        
        #--Observation noise Prior--#
        self.register_buffer('sigma_noise', torch.tensor(self.config.sigma_noise))

    def _break_stick(self, v_raw, concentration):
        """GEM Construction"""
        v = torch.sigmoid(v_raw - torch.log(concentration))
        cum_prod = torch.cumprod(1 - v, dim=-1)
        pi = torch.zeros(len(v) + 1, device=v.device)
        pi[0] = v[0]
        pi[1:-1] = v[1:] * cum_prod[:-1]
        pi[-1] = cum_prod[-1]
        return pi
    
    def get_weights(self, local_idx):
        """Compute hierarchical weights using fixed alpha/gamma state"""
        beta = self._break_stick(self.v_k, self.gamma)
        pi_raw = self._break_stick(self.v_j[local_idx], self.alpha)
        pi_j = pi_raw * beta
        weights = pi_j / (pi_j.sum() + 1e-8)
        return weights
    
    def forward(self, local_idx):
        return self.get_weights(local_idx)


#---Class Definition: Reproducing Kernel Hilbert Space Decoder----#
class ReproducingKernelHilbertSpaceDecoder(nn.Module):
    def __init__(self, config: RKHSConfig):
        super().__init__()
        self.config = config
        self.latent_dim = self.config.latent_dim
        self.feature_dim = self.config.transformer_out_dim
        self.n_anchor_points = self.config.n_anchors

        #---Learnable Anchor Points on Latent Manifold Z---#
        self.anchor_points = nn.Parameter(torch.randn(self.n_anchor_points, self.latent_dim))

        #---RKHS Coefficients (Alpha) for Kernel Reconstruction---#
        self.alpha = nn.Parameter(torch.randn(self.n_anchor_points, self.feature_dim))

        #---Kernel Hyperparameters (Gaussian RBF Kernel)---#
        self.log_lengthscale = nn.Parameter(torch.zeros(1))
    
    def compute_kernel(self, z_1, z_2):
        """Computes RBF Kernel between two sets of points"""
        lengthscale = torch.exp(self.log_lengthscale)
        dist_sq  = torch.cdist(z_1, z_2)**2
        kernel = torch.exp(-0.5 * dist_sq / (lengthscale**2 + 1e-7))
        return kernel
    
    def forward(self, z):
        """
        Reconstructs original vision transformer feature vectors from latent representations
        h_hat = sum_{i=1}^{n_anchor_points} alpha_i * k(z, anchor_point_i)
        """

        #---Compute Kernel between input z and anchor points on latent manifold---#
        K_za = self.compute_kernel(z, self.anchor_points)  #---[B, n_anchor_points]---#

        #---Linear Combination in Hilbert Space for RKHS reconstruction---#
        h_recon = torch.matmul(K_za, self.alpha)

        return h_recon
    
#Class definition: Final Model architecture & flow with Bayesian nonparametric clustering---#
class InfiniteGaussianMixtureModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        const_params = self.config.latent_dim

        #--Share fixed params--#
        for key in ['transformer', 'dirichlet', 'gp', 'rkhs']:
            config[key]['latent_dim'] = const_params


        #--Feature Extractor---#
        self.transformer = VisionTransformerFeatureExtractor(self.config.transformer)

        #---Hierarchical Dirichlet Process---#
        self.dirichlet = HierarchicalDirichletProcess(self.config.dirichlet)

        #---Stateless Woodbury GP Operator---#
        self.gp = StatelessWoodburyRandomFourierGaussianProcess(self.config.gp)

        #---RKHS Decoder--#
        self.decoder = ReproducingKernelHilbertSpaceDecoder(self.config.rkhs)

        self.K = self.config.dirichlet.n_global
        self.sigma_noise = self.config.dirichlet.sigma_noise
    
    def forward(self, x, local_idx, y=None):
        #---Forward Pass for entire model--#
        z = self.transformer(x) #---Proj to latent space---#
        pi_j = self.dirichlet.get_weights(local_idx) #--Weights for each known local cluster---#
        #--Inifinite GP---#
        mu_proj = 0.0
        var_pred = 0.0
        for k in range(self.K):
            mu_k, sig_k = self.gp(
                z,
                mu_atom=self.dirichlet.mu_atom[k],
                log_sigma_atom = self.dirichlet.log_sigma_atom[k],
                sigma_noise=self.sigma_noise,
                y=y
            )

            #--Aggregate Variance---#
            mu_proj += pi_j[k] * mu_k
            var_pred += pi_j[k] * (sig_k + mu_k**2)
        var_pred = var_pred - mu_proj**2
        var_pred = torch.clamp(var_pred, min=1e-9)

        #---RKHS Manifold Projection--#
        h_recon = self.decoder(z)

        return{
            "mu": mu_proj,
            "var": var_pred,
            "hilbert_space_kernel_recon": h_recon,
            "z_latent": z,
            "weights": pi_j
        }


