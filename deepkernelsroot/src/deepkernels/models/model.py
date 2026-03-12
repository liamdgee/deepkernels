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
from deepkernels.models.NKN import GPParams
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
    def __init__(self, vae=None, likelihood=None, gp=None, run_gp:bool=False, **kwargs):
        super().__init__()
        self.vae = vae if vae is not None else SpectralVAE(**kwargs)
        k_atoms = kwargs.get("k_atoms", 30)
        min_noise = kwargs.get("min_noise", 1e-3)
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=k_atoms, noise_constraint=gpytorch.constraints.GreaterThan(min_noise))
        
        self.run_gp = run_gp
        if gp is not None:
            self.gp = gp
        else:
            self.gp = AcceleratedKernelGP(
                likelihood=self.likelihood,
                **kwargs
            )

        self.kwargs = kwargs
    
    def pack_features(self, gp_params, pi):
        """Safely pool any 4D tensors to 3D and concatenate the 198D payload."""
        def process_param(p):
            return p.mean(dim=2) if p.dim() == 4 else p

        packed = torch.cat([
            process_param(gp_params.gates), 
            process_param(gp_params.linear), 
            process_param(gp_params.periodic), 
            process_param(gp_params.rational), 
            process_param(gp_params.polynomial), 
            process_param(gp_params.matern),
            pi
        ], dim=-1)
        
        return packed.contiguous()

    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        batch_size = x.size(0)
        steps = steps if steps is not None else 3

        updated_params = {**params, "indices": indices}
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x, x.device, batch_size=batch_size)
        
        

        state = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            indices=indices,
            **params
        )
        
        if features_only:
            return state
        
        
        gp_features = self.pack_features(state.gp_params, state.pi)
        lmc_learned = state.lmc_matrices
        
        mvn = None

        if self.gp is not None and self.run_gp:
            mvn = self.gp(gp_features, x=gp_features, lmc_learned=lmc_learned, indices=indices, **self.kwargs)
        
        return ModelOutput(
            state=state,
            gp_out=mvn, 
            gp_in=gp_features,
            **updated_params
        )
    
    