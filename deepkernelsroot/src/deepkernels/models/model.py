import torch
import gpytorch
import torch.nn as nn

from deepkernels.models.parent import BaseGenerativeModel

import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

from tqdm import tqdm

from deepkernels.models.variationalautoencoder import SpectralVAE, StateSpaceOutput
from deepkernels.models.gaussianprocess import AcceleratedKernelGP
from typing import NamedTuple, Optional
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelOutput(NamedTuple):
    state: StateSpaceOutput
    gp_out: Optional[gpytorch.distributions.MultivariateNormal]
    gp_in: torch.Tensor

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, likelihood=None, gp=None, run_gp:bool=False, k_atoms=30, min_noise=1e-3):
        super().__init__()
        self.vae = SpectralVAE()
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=k_atoms, noise_constraint=gpytorch.constraints.GreaterThan(min_noise))
        
        self.run_gp = run_gp
        if gp is not None:
            self.gp = gp(likelihood=self.likelihood)
        else:
            self.gp = AcceleratedKernelGP(
                likelihood=self.likelihood
            )
    
    def pack_features(self, gates, linear, periodic, rational, polynomial, matern, pi):
        """Safely pool any 4D tensors to 3D and concatenate the 198D payload."""
        def process_param(p):
            return p.mean(dim=2) if p.dim() == 4 else p

        packed = torch.cat([
            process_param(gates), 
            process_param(linear), 
            process_param(periodic), 
            process_param(rational), 
            process_param(polynomial), 
            process_param(matern),
            pi
        ], dim=-1)
        
        return packed.contiguous()

    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        batch_size = x.size(0)
        steps = steps if steps is not None else 3
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x, x.device, batch_size=batch_size)
        
        

        state = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            indices=indices
        )
        
        if features_only:
            return state
        
        
        gp_features = self.pack_features(gates=state.gates, 
                                         linear=state.linear, 
                                         periodic=state.periodic, 
                                         rational=state.rational, 
                                         polynomial=state.polynomial, 
                                         matern=state.matern, 
                                         pi=state.pi
                                         )
        
        lmc_learned = state.lmc_matrices
        
        mvn = None

        if self.gp is not None and self.run_gp:
            mvn = self.gp(gp_features, x=gp_features, lmc_learned=lmc_learned, indices=indices)
        
        return ModelOutput(
            state=state,
            gp_out=mvn, 
            gp_in=gp_features
        )
    
    