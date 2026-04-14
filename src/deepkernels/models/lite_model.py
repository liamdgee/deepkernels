import logging
from typing import NamedTuple, Optional

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

try:
    import deepsecrets
    from deepsecrets.src.deepsecrets.models.gaussianprocess import AcceleratedKernelGP
    from deepsecrets.src.deepsecrets.models.parent import BaseGenerativeModel
    from deepsecrets.src.deepsecrets.models.variationalautoencoder import (SpectralVAE,
                                                        StateSpaceOutput)
    AUTH_USER = True
    logger.info("Authentication Status approved. Loading core model architecture.")
except ImportError:
    AUTH_USER = False
    from deepkernels.models.lite_gp import SpectralVariationalGaussianProcess as DummyGP
    from deepkernels.models.lite_dirichlet import HierarchicalDirichletProcess as DummyVAE
    from deepkernels.models.lite_model_config import RootConfig as DummyConfig
    from deepkernelsroot.src.deepkernels.models.lite_parent import BaseGenerativeModel
    logger.info("Authentication Status declined. Loading dummy model architecture.")



class StateSpaceKernelProcess(BaseGenerativeModel):
    def __init__(
        self,
        likelihood=None,
        gp=None,
        k_atoms=30,
        num_latents=8,
        min_noise=1e-3,
        device="cuda",
        **kwargs
    ):
        super().__init__()
        self.device = self.get_device(device)
        if AUTH_USER:
            self.vae = SpectralVAE()
            self.gp = AcceleratedKernelGP(
                likelihood=gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(min_noise)
                )
            )
        else:
            self.vae = DummyVAE(config=DummyConfig())
            self.gp = DummyGP(config=DummyConfig())
        
        self.input_dim = kwargs.get("input_dim", 30)
        self.n_data = kwargs.get("n_data", 87636.0)

    def zero_state(self, x, device, batch_size):
        state = self.vae.get_zero_state(x, device, batch_size)
        return state

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
        full_var = torch.empty(
            (batch_size, horizon), dtype=torch.float32, device=device
        )
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
        vae_out,
        indices=None,
        steps=2,
        batch_shape=torch.Size([]),
        features_only: bool = False,
        generative_mode: bool = False,
        **params
    ):
        if vae_out is None:
            vae_out = self.vae.get_zero_state(x, x.device, batch_size=x.size(0))

        state = self.vae(
            x,
            vae_out=vae_out,
            steps=steps,
            batch_shape=batch_shape,
            indices=indices,
            generative_mode=generative_mode,
        )

        if features_only:
            return state, None, None

        zz = self.pack_features(
            state.gates,
            state.linear,
            state.periodic,
            state.rational,
            state.polynomial,
            state.matern,
            state.pi,
        )

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
