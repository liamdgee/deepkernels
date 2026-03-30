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

    def zero_state(self, x, device, batch_size):
        state = self.vae.get_zero_state(x, device, batch_size)
        return state

    def generate_trajectory(self, xt, state=None, horizon=64, device='cuda'):
        """
        Natively generates a multi-step trajectory by feeding 
        hallucinations back into the latent state.
        """
        self.eval()
        mu_history = []
        var_history = []
        if state is None:
            state = self.zero_state(xt, device, batch_size=xt.size(0))
        with torch.no_grad():
            for t in range(1, horizon + 1):
                state, _, _ = self.forward(
                    xt, 
                    vae_out=state, 
                    steps=2,
                    features_only=True,
                    generative_mode=True
                )
                zz = self.pack_features(state.gates, state.linear, state.periodic, state.rational, state.polynomial, state.matern, state.pi)
                lmc_raw = state.lmc_consensus.mean(dim=0)
                top_val, _ = torch.topk(lmc_raw, k=4, dim=0)
                min_val = top_val.min(dim=0, keepdim=True)[0]
                lmc_sparse = torch.where(lmc_raw >= min_val, lmc_raw, torch.zeros_like(lmc_raw))
                squashed_weights = lmc_sparse.sum(dim=0)
                
                final_weights = squashed_weights / (squashed_weights.sum() + 1e-7)
                with torch.no_grad():
                    self.gp.variational_strategy.lmc_coefficients = final_weights.detach().contiguous()
                mvn = self.gp(zz)

                mu_history.append(mvn.mean.unsqueeze(1))
                var_history.append(mvn.variance.unsqueeze(1))
                
                xt = state.recon
        
        full_mu = torch.cat(mu_history, dim=1)
        full_var = torch.cat(var_history, dim=1)
        
        return full_mu, full_var

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