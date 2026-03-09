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
from deepkernels.models.exactgp import Simple
from deepkernels.models.NKN import GPParams
from typing import NamedTuple, Optional

class ModelOutput(NamedTuple):
    state: DecoderStateOutput
    gp_out: Optional[gpytorch.distributions.MultivariateNormal]
    gp_in: torch.Tensor

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, vae=None, gp=None, run_gp:bool=False, **kwargs):
        super().__init__()
        self.vae = vae if vae is not None else SpectralVAE(**kwargs)
        self.gp = gp if gp else None
        self.kwargs = kwargs
        self.run_gp = run_gp if run_gp is not None else False
        

    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        batch_size = x.size(0)
        steps = steps if steps is not None else 3
        
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x.device, batch_size=batch_size)
        
        state = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            **params
        )
        
        if features_only:
            return state
        
        gp_in = state.gp_params
        
        gp_features = torch.cat([gp_in.gates, gp_in.linear, gp_in.periodic, gp_in.rational, gp_in.polynomial, gp_in.matern], dim=-1)

        if isinstance(gp_in, tuple):
            gp_in = gp_in[0]
        
        gp_in = gp_in.contiguous()
        
        mvn = None
        if self.gp is not None and self.run_gp:
            gp_kwargs = {
                "gp_params": state.gp_params,
                "pi": state.pi
            }

            mvn = self.gp(gp_in, **gp_kwargs)
        
        return ModelOutput(
            state=state,
            gp_out=mvn, 
            gp_in=gp_in
        )
    
    def pack_gp_features(self, hyperparams):
        """
        Takes the structured GPParams and flattens them into a single 
        168-dimensional tensor for the Gaussian Process.
        """
        param_keys = ['gates', 'linear', 'periodic', 'rational', 'polynomial', 'matern']
        pooled_params = []

        for k in param_keys:
            val = getattr(hyperparams, k)
            if val.dim() == 4:
                val = val.mean(dim=2)
                
            pooled_params.append(val)
        
        return torch.cat(pooled_params, dim=-1)