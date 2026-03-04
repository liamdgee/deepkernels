import torch
import gpytorch
import torch.nn as nn

from deepkernels.models.parent import BaseGenerativeModel

from tqdm import tqdm

from deepkernels.models.variationalautoencoder import SpectralVAE, StateSpaceOutput, DecoderStateOutput, HistoryOutput
from deepkernels.models.exactgp import Simple
from deepkernels.models.NKN import GPParams
from typing import NamedTuple, Optional

class ModelOutput(NamedTuple):
    state: DecoderStateOutput
    history: HistoryOutput
    gp_out: Optional[gpytorch.distributions.MultivariateNormal]
    gp_in: torch.Tensor

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, vae=None, gp=None, run_gp:bool=False):
        super().__init__()
        self.vae = vae if vae is not None else SpectralVAE()
        self.gp = gp if gp else None
        
        self.run_gp = run_gp

    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        batch_size = x.size(0)
        steps = steps if steps is not None else 3
        
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x.device, batch_size=batch_size)
        
        ss_output = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            features_only=features_only,
            **params
        )
        
        if features_only:
            return ss_output

        gp_in = ss_output.state.gp_features

        if isinstance(gp_in, tuple):
            gp_in = gp_in[0]
        
        gp_in = gp_in.contiguous()
        
        mvn = None
        if self.gp is not None and self.run_gp:
            gp_kwargs = {
                "gp_params": ss_output.history.gp_params,
            }

            mvn = self.gp(gp_in, **gp_kwargs)
        
        return ModelOutput(
            state=ss_output.state, 
            history=ss_output.history, 
            gp_out=mvn, 
            gp_in=gp_in
        )