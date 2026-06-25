"""
Note: These are lite display versions of real, larger and more complex model components. 
Real StateSpaceOutput and architecture build are much more comprehensive.
"""

import logging
from typing import NamedTuple, Optional

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

#-local imports-#
from .parent import BaseGenerativeModel
from .vae import SpectralVAE
from .state import StateSpaceOutput, ModelOutput
from .gp import AcceleratedKernelGP

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class ShallowKernels(BaseGenerativeModel):
    def __init__(
        self,
        likelihood=None,
        gp=None,
        min_noise=1e-3,
        device="cuda",
        **kwargs
    ):
        super().__init__()
        self.device = self.get_device(device)
        self.input_dim = kwargs.get("input_dim", 30)
        self.n_data = kwargs.get("n_data", 10000.0)
        self.vae = SpectralVAE(input_dim=self.input_dim)
        
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(min_noise)
            )
        
        self.gp = AcceleratedKernelGP(likelihood=likelihood, input_dim=self.input_dim * 7)

    def zero_state(self, x, device, batch_size):
        return self.vae.get_zero_state(x, device, batch_size)

    def generate_trajectory(self, xt, state=None, steps=1, horizon=64, device="cuda"):
        """
        Autoregressive trajectory generation using the native forward method.
        Optimized with pre-allocation and VRAM flushing for 6GB limits.
        """
        self.eval()
        batch_size = xt.size(0)

        if state is None:
            state = self.zero_state(xt, device, batch_size=batch_size)
            
        full_mu = torch.empty((batch_size, horizon), dtype=torch.float32, device=device)
        full_var = torch.empty((batch_size, horizon), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            for t in range(horizon):
                state, mvn, zz = self.forward(
                    xt,
                    vae_out=state,
                    steps=steps,
                    features_only=False,
                    generative_mode=True,
                )

                full_mu[:, t] = mvn.mean.view(-1)
                full_var[:, t] = mvn.variance.view(-1)

                xt = state.recon.detach()

                del mvn, zz

        return full_mu, full_var

    def forward(
        self,
        x,
        state: Optional[StateSpaceOutput]=None,
        indices=None,
        batch_shape=torch.Size([]),
        features_only: bool = False,
        generative_mode: bool = False,
        **params,
    ) -> ModelOutput:
        if state is None:
            state = self.vae.get_zero_state(x, x.device, batch_size=x.size(0))

        state = self.vae(
            x,
            state=state,
            batch_shape=batch_shape,
            indices=indices,
            generative_mode=generative_mode,
        )

        if features_only:
            return ModelOutput.features_only(state)

        
        zz = self.pack_features(state)
        
        mvn = self.gp(zz)

        return ModelOutput(state=state, gp_out=mvn, zz=zz)

# --- Runnable Test Block --- #
if __name__ == "__main__":
    logger.info("Initializing public StateSpaceKernelProcess model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Instantiate
    model = ShallowKernels(input_dim=30, device=device).to(device)
    
    # 2. Dummy Data
    dummy_input = torch.randn(4, 30).to(device)
    
    # 3. Test Forward
    state, mvn, zz = model(dummy_input, vae_out=None)
    logger.info(f"Forward Pass GP Output Mean Shape: {mvn.mean.shape}")
    
    # 4. Test Trajectory Generation
    logger.info("Testing trajectory generation...")
    mu, var = model.generate_trajectory(dummy_input, horizon=10, device=device)
    logger.info(f"Trajectory Gen Success. Mean Matrix Shape: {mu.shape}")