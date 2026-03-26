import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F
from deepkernels.models.parent import BaseGenerativeModel

from tqdm import tqdm

from deepkernels.models.variationalautoencoder import SpectralVAE, StateSpaceOutput
from deepkernels.models.gaussianprocess import AcceleratedKernelGP
from typing import NamedTuple, Optional
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, likelihood=None, gp=None, k_atoms=30, num_latents=8, min_noise=1e-3, device='cuda', **kwargs):
        super().__init__()
        self.device = self.get_device(device)
        self.vae = SpectralVAE()
        self.gp = AcceleratedKernelGP(likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(min_noise)))
        self.input_dim = kwargs.get("input_dim", 30)
        self.n_data = kwargs.get('n_data', 87636.0)

    def forward(self, x, vae_out, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, generative_mode:bool=False, **params):
        steps = steps if steps is not None else 3
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x, x.device, batch_size=x.size(0))
        

        state = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            indices=indices,
            generative_mode=generative_mode
        )
        
        if features_only:
            return state, None, None
        
        zz = self.pack_features(state.gates, state.linear, state.periodic, state.rational, state.polynomial, state.matern, state.pi)
        
        mvn = None
        
        lmc_raw = state.lmc_consensus.mean(dim=0)
        top_val, _ = torch.topk(lmc_raw, k=4, dim=0)
        min_val = top_val.min(dim=0, keepdim=True)[0]
        lmc_sparse = torch.where(lmc_raw >= min_val, lmc_raw, torch.zeros_like(lmc_raw))
        squashed_weights = lmc_sparse.sum(dim=0)
        
        final_weights = squashed_weights / (squashed_weights.sum() + 1e-7)
        
        self.gp.variational_strategy.lmc_coefficients = final_weights.contiguous()
        mvn = self.gp(zz)

        return state, mvn, zz