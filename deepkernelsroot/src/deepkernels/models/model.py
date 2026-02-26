import torch
import gpytorch
import torch.nn as nn

from deepkernels.losses import kl
from deepkernels.models.parent import BaseGenerativeModel

from tqdm import tqdm

from deepkernels.models.variationalautoencoder import SpectralVAE, StateSpaceOutput, DecoderStateOutput, HistoryOutput
from deepkernels.models.gaussianprocess import AcceleratedKernelGP, GPOutput
from deepkernels.models.NKN import GPParams
from typing import NamedTuple

class ModelOutput(NamedTuple):
    state: DecoderStateOutput
    history: HistoryOutput
    gp_out: GPOutput
    gp_in: torch.Tensor

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, vae=None, gp=None):
        super().__init__()
        self.vae = vae if vae is not None else SpectralVAE()
        self.gp = gp if gp is not None else AcceleratedKernelGP()
    
    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        
        
        state, history = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            features_only=features_only,
            **params
        )

        gp_in = history.gp_features

        if isinstance(gp_in, tuple):
            gp_in = gp_in[0]
        
        gp_in = gp_in.contiguous()
        
        gp_kwargs = {
            "gp_params": history.gp_params, 
        }

        gp_out = self.gp(gp_in, **gp_kwargs)
        
        return ModelOutput(state=state, history=history, gp_out=gp_out, gp_in=gp_in)