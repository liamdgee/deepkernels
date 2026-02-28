import torch
import gpytorch
import torch.nn as nn

from deepkernels.losses import kl
from deepkernels.models.parent import BaseGenerativeModel

from tqdm import tqdm

from deepkernels.models.variationalautoencoder import SpectralVAE, StateSpaceOutput, DecoderStateOutput, HistoryOutput
from deepkernels.models.gaussianprocess import AcceleratedKernelGP
from deepkernels.models.NKN import GPParams
from typing import NamedTuple

class ModelOutput(NamedTuple):
    state: DecoderStateOutput
    history: HistoryOutput
    gp_out: gpytorch.distributions.MultivariateNormal
    gp_in: torch.Tensor

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, vae=None, gp=None):
        super().__init__()
        self.vae = vae if vae is not None else SpectralVAE()
        self.gp = gp if gp is not None else AcceleratedKernelGP()
    
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
        if gp_in.dim() == 4:
            if gp_in.size(1) == self.gp.num_latents and gp_in.size(0) != self.gp.num_latents:
                gp_in = gp_in.permute(1, 0, 2, 3)
        elif gp_in.dim() == 3:
            if gp_in.size(1) == self.gp.num_latents:
                gp_in = gp_in.transpose(0, 1)
        
        gp_in = gp_in.contiguous()

        gp_kwargs = {
            "gp_params": ss_output.history.gp_params,
            "pi": ss_output.history.pis
        }

        mvn = self.gp(gp_in, **gp_kwargs)
        
        return ModelOutput(state=ss_output.state, history=ss_output.history, gp_out=mvn, gp_in=gp_in)