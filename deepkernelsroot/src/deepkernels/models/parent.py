import torch
import gpytorch
from gpytorch.mlls import AddedLossTerm
from gpytorch.priors import NormalPrior, GammaPrior, HorseshoePrior
import torch.distributions as dist
import torch.nn.functional as F
from deepkernels.losses.simple import SimpleLoss
import torch.distributions as dist
import math
import logging
from typing import Union, Optional, Dict, Tuple, TypeAlias, Union
import pykeops

import torch
from torch.distributions import LowRankMultivariateNormal, Independent, Normal, kl_divergence

import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseGenerativeModel(gpytorch.Module):
    def __init__(self):
        super().__init__()
    
    
    def register_constrained_parameter(self, name, parameter, constraint):
        self.register_parameter(name, parameter)
        self.register_constraint(name, constraint)
        return self
    
    def reparameterise(self, mu, logvar, eps_min=-3.3, eps_max=3.3):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps = torch.clamp(eps,  min=eps_min, max=eps_max)
        return mu + eps * std
    
    def forward(self, x, vae_out, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        raise NotImplementedError("Subclass must implement forward()")
        
    def get_variational_strategy(self):
        raise NotImplementedError("Get strategy from subclass: model or orchestrate")
    
    def multivariate_projection(self, mu, factor, diag, jitter=1e-6):
        """for alpha params: input projections from three alpha heads"""
        mvn = dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=factor,
            cov_diag=diag
        )

        logits = mvn.rsample()
        alpha = torch.nn.functional.softplus(logits) + jitter
        
        return alpha
    def inverse_softplus(self, target_value, min=3e-7):
        """
        Numerically stable mapping from concentration space (0, inf) 
        to logit space (-inf, inf).
        """
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.tensor(target_value)
            
        safe_target = torch.clamp(target_value, min=min)
        
        return torch.where(
            safe_target > 20.0,
            safe_target,
            torch.log(torch.expm1(safe_target))
        )
    
    def dirichlet_sample(self, alpha):
        alpha = F.softplus(alpha)
        alpha = torch.clamp(alpha, min=4e-2)
        q_alpha= torch.distributions.Dirichlet(alpha)
        pi_sample = q_alpha.rsample()
        return pi_sample

    def get_device(self, device_request: Union[str, torch.device, None] = None) -> torch.device:

        """
        Resolves the optimal available device for PyTorch operations.
        
        Priority:
        1. explicit device_request (if provided and valid)
        2. cuda:0 (NVIDIA GPU)
        3. mps (Apple Silicon Metal Performance Shaders)
        4. cpu
        
        Args:
            device_request: Optional string ('cuda', 'mps', 'cpu') or torch.device 
                            to force a specific device.
        
        Returns:
            torch.device: The resolved device.
        """
        if device_request is not None:
            device = torch.device(device_request)
            if device.type == 'cuda' and not torch.cuda.is_available():
                logging.warning(f"CUDA requested but unavailable. Falling back to CPU.")
                return torch.device('cpu')
            if device.type == 'mps' and not torch.backends.mps.is_available():
                logging.warning(f"MPS (Apple Silicon) requested but unavailable. Falling back to CPU.")
                return torch.device('cpu')
            return device
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        return torch.device('cpu')
    
    def lowrankmultivariatenorm(self, mu, factor, diag):
        mvn = torch.distributions.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        logits = mvn.rsample()
        return logits
    
    def apply_softplus(self, x, jitter=1e-6):
        return F.softplus(x) + jitter
    
    def stack_features(self, latent_kernels):
        return torch.stack(latent_kernels)
    
    def get_resource(self, name_string, **params):
        return getattr(self, name_string, None)