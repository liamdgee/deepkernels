import torch
import gpytorch
import torch.nn as nn

from deepkernels.losses import kl
from deepkernels.models.parent import BaseGenerativeModel

from tqdm import tqdm

from deepkernelsroot.src.deepkernels.models.variationalautoencoder import *
from deepkernelsroot.src.deepkernels.models.gaussianprocess import *

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, vae=None, gp=None):
        self.vae = vae if vae is not None else SpectralVAE()
        self.gp = gp if gp is not None else AcceleratedKernelGP()
    
    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        
        
        state_space_tuple = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            features_only=features_only,
            **params
        )
        history = state_space_tuple.history
        gp_input = history.features 
        
        if gp_input.dim() == 3:
            gp_input = gp_input.unsqueeze(-2)
            
        gp_kwargs = {
            "gp_params": history.expert_params,
            "mixture_means_per_expert": history.expert_mixtures,
        }
        
        gp_output = self.gp(gp_input, **gp_kwargs)
        
        return state_space_tuple.state, gp_output, gp_input, history