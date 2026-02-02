#-dependencies-#
from src.models.hypernetwork import SpectralEncoder as hnet
import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import Kernel

#-kernel class-#
class DynamicEigenKernel(Kernel):
    """
    Non-Stationary Spectral Mixture Kernel (positive definite gibbs sampler) built for HMC.
    """
    def __init__(self, input_dim, n_mixtures=4, hidden_dim=32, hypernet=hnet, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_mixtures = n_mixtures
        self.hidden_dim = hidden_dim
        self.jitter = 1e-6

        #--init hypernetwork-#
        self.hypernet = hypernet(self.input_dim, self.n_mixtures, self.hidden_dim)

        #-Static global frequencies (we are performing bayesian optimisation in parameter space)-#
        self.register_parameter(
            name="raw_mixture_means", 
            parameter=torch.nn.Parameter(torch.zeros(self.n_mixtures, self.input_dim))
        )
        
        #-- for hamiltonian monte carlo: prior on neural net weights to define an energy function--#
        self.register_prior(
            "hypernet_weight_prior",
            gpytorch.priors.NormalPrior(0.0, 1.0),
            lambda m: [p for p in m.hypernet.parameters()], 
            lambda m, v: None
        )
        
        #--HMC prior on spectral frequencies-#
        self.register_prior(
            "mixture_mean_prior",
            gpytorch.priors.NormalPrior(0.0, 1.0),
            "raw_mixture_means"
        )

    @property
    def mixture_means(self):
        return self.raw_mixture_means

    def forward(self, x1, x2, diag=False, **params):
        """
        Args:
         - x1: [..., N, D]
         - x2: [..., M, D]
        """
        #-generate dynamic params-#
        w1, l1 = self.hypernet(x1) #-w_star: [N, Q]
        w2, l2 = self.hypernet(x2) #-l_star: [N, Q, D]
        
        #-total variance terms-#
        S1 = w1.sum(dim=-1, keepdim=True) #- [..., N, 1]
        S2 = w2.sum(dim=-1, keepdim=True) #- [..., M, 1]

        #-normalise weights -> probability space-#
        pi1 = torch.clamp(w1 / (S1 + self.jitter), min=self.jitter) #-[..., N, Q]
        pi2 = torch.clamp(w2 / (S2 + self.jitter), min=self.jitter) #-[..., N, Q]
        
        # ---------------------------------------------------------
        # Computationally light (matrix diag only) ~ O(N)
        # ---------------------------------------------------------

        if diag:
            scale = S1.squeeze(-1) # [..., N]
            diff = x1 - x2
            sq_dist = diff.pow(2)
            kernel_diag = torch.zeros_like(scale)
            
            for q in range(self.n_mixtures):
                l1_q = l1[:, q, :]
                l2_q = l2[:, q, :]
                
                #-Gibbs terms-#
                l1_sq = l1_q.pow(2)
                l2_sq = l2_q.pow(2)
                l_sum = l1_sq + l2_sq
                prefactor_num = 2 * l1_q * l2_q
                gibbs_prefactor = torch.sqrt(prefactor_num / l_sum).prod(dim=-1)
                gibbs_exp = torch.exp(-(sq_dist / l_sum).sum(dim=-1))
                
                #-cos oscillation term-#
                mu = self.mixture_means[q, :]
                cos_arg = 2 * torch.pi * (diff * mu).sum(dim=-1)
                oscil = torch.cos(cos_arg)
                
                #-Matusita term-#
                matusita_q = torch.sqrt(pi1[:, q] * pi2[:, q])
                
                kernel_diag += matusita_q * gibbs_prefactor * gibbs_exp * oscil
                
            return scale * kernel_diag
        
        # ---------------------------------------------------------
        # FULL MATRIX ~ O(N^2)
        # ---------------------------------------------------------

        #-Global Volume (scalar) -- geometric mean of total variance-#
        scale = torch.sqrt(S1 * S2.transpose(-1, -2))

        #-Shape Matching - Matusita Affinity-#
        matusita_all = torch.sqrt(pi1.unsqueeze(-2) * pi2.unsqueeze(-3))

        #-distances (euclidean)-#
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3) #-[..., N, M, D]
        sq_dist = diff.pow(2)
        
        #- Accumulate Kernel Values -- Iter over Q mixture compopnents to compute the kernel K(x1, x2)-#
        unscaled_kernel = torch.zeros(x1.shape[:-1] + (x2.shape[-2],), device=x1.device)


        for q in range(self.n_mixtures):
            l1_q = l1[:, q, :] #- [..., N, D]
            l2_q = l2[:, q, :] #- [..., M, D]
    
            # --- Gibbs Term (Spatial Decay) ---#
            l1_sq = l1_q.pow(2) # [..., N, D]
            l2_sq = l2_q.pow(2) # [..., M, D]
            l_sum = l1_sq.unsqueeze(-2) + l2_sq.unsqueeze(-3) #-broadcast for dimension generality--#
            
            #- Prefactor for non-stationarity in lengthscales-#
            prefactor_num = 2 * l1_q.unsqueeze(-2) * l2_q.unsqueeze(-3)
            gibbs_prefactor = torch.sqrt(prefactor_num / l_sum).prod(dim=-1) # Product over dimensions
            
            #--Exponential term for smoothness-#
            gibbs_exp = torch.exp(-torch.sum(sq_dist / l_sum, dim=-1))
            
            # --- Cosine Term (Spectral Oscillation) --#
            mu = self.mixture_means[q, :] #- [D]
            cos_arg = 2 * torch.pi * (diff * mu).sum(dim=-1)
            oscil = torch.cos(cos_arg)

            #--Matusita Term (Texture Gate)-#
            #--ellipsis slicing to allow dimension agnostic kernel-#
            matusita_affinity = matusita_all[..., q]
            
            #-accumulate kernel-#
            unscaled_kernel += matusita_affinity * gibbs_prefactor * gibbs_exp * oscil

        kernel = scale * unscaled_kernel
        
        return kernel