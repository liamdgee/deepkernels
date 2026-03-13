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
from deepkernels.models.gaussianprocess import AcceleratedKernelGP, DynamicStrategy
from typing import NamedTuple, Optional
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(self, likelihood=None, gp=None, k_atoms=30, num_latents=8, min_noise=3e-3):
        super().__init__()
        self.vae = SpectralVAE()
        self.gp = AcceleratedKernelGP(
                likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=k_atoms,
                    noise_constraint=gpytorch.constraints.GreaterThan(min_noise))
            )
    def pack_features(self, gates, linear, periodic, rational, polynomial, matern):
        def to_3d(p):
            if p.dim() == 4:
                return p.squeeze(1) if p.size(1) == 1 else p.squeeze(2)
            if p.dim() == 2:
                return p.unsqueeze(0).expand(8, -1, -1)
            return p

        packed = torch.cat([
            to_3d(gates), to_3d(linear), to_3d(periodic),
            to_3d(rational), to_3d(polynomial), to_3d(matern)
        ], dim=-1)
        
        return packed.contiguous()

    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        steps = steps if steps is not None else 3
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x, x.device, batch_size=x.size(0))
        
        

        state = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            indices=indices
        )
        
        if features_only:
            return state, None, None
        
        gp_features = torch.cat([state.gates, state.linear, state.periodic, state.rational,state.polynomial, state.matern], dim=1)
        gp_input = gp_features.unsqueeze(0).expand(8, -1, -1)
        # --- CATCH THE MUTATION ---
        print(f"DEBUG: periodic shape right before KeOps: {state.periodic.shape}, gates: {state.gates.shape}, linear: {state.linear.shape}, rational: {state.rational.shape}, poly: {state.polynomial.shape}, matern: {state.matern.shape}, lmc: {state.lmc_matrices.shape}")
        # --------------------------
        
        #lmc = state.lmc_matrices
        #lmc_param = lmc.view(-1).contiguous()
        
        mvn = None

        mvn = self.gp(gp_features, indices=indices)

        return state, mvn, gp_features
    
    