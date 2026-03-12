import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from deepkernels.models.encoder import ConvolutionalLoopEncoder, ConvolutionalNetwork1D, EncoderOutput
from deepkernels.models.decoder import SpectralDecoder, DecoderOutput
from deepkernels.models.dirichlet import AmortisedDirichlet, DirichletOutput
from typing import Tuple, Optional, TypeAlias, Tuple, Union, NamedTuple
import torch.nn.functional as F
import numpy as np
import logging
from gpytorch.mlls import AddedLossTerm
from torch.distributions import kl_divergence
import gpytorch

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from deepkernels.models.parent import BaseGenerativeModel
from deepkernels.models.NKN import GPParams

class StateSpaceOutput(NamedTuple):
    gp_params: GPParams
    bottleneck: torch.Tensor
    alpha: torch.Tensor
    alpha_mu: torch.Tensor
    alpha_factor: torch.Tensor
    alpha_diag: torch.Tensor
    gp_features: torch.Tensor
    parameters_per_expert: torch.Tensor
    recon: torch.Tensor
    bandwidth_mod: torch.Tensor
    pi: torch.Tensor
    amp: torch.Tensor
    trend: torch.Tensor
    res: torch.Tensor
    ls: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    lmc_matrices: torch.Tensor
    real_x: torch.Tensor

class SpectralVAE(BaseGenerativeModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.dirichlet = AmortisedDirichlet(**kwargs)
        self.decoder = SpectralDecoder(**kwargs)
        self.encoder = ConvolutionalLoopEncoder(**kwargs)
        self.eps = kwargs.get('eps_dirichlet', 1e-4)
        self.kwargs=kwargs
    
    def get_zero_state(self, x, device, batch_size):
        k = self.kwargs.get("k_atoms", 30)
        e = self.kwargs.get("num_latents", 8)
        f = self.kwargs.get("latent_dim", 16)
        r = self.kwargs.get("alpha_factor_rank", 3) # Use the actual rank
        x_in = self.kwargs.get("input_dim", 30)
        bottleneck = self.kwargs.get("bottleneck_dim", 64)
        
        uniform_logit = self.inverse_softplus(1.0).item()

        alpha_factor = torch.zeros(batch_size, k, r, device=device)
        for j in range(r):
            start, end = j * (k // r), (j + 1) * (k // r)
            alpha_factor[:, start:end, j] = 1.0
        alpha_factor += torch.randn_like(alpha_factor) * 0.001
        #-pi init-#
        init_pi = self.init_pi_value(batch_size=batch_size, device=device)
        initial_lmc = torch.zeros(k, e, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        symmetry_breaker = torch.randn_like(initial_lmc) * 1e-4
        initial_lmc = initial_lmc + symmetry_breaker

        return StateSpaceOutput(
            recon=torch.randn(batch_size, x_in, device=device) * self.eps,
            gp_features=torch.randn(batch_size, e, f, device=device) * self.eps,
            pi=init_pi,
            alpha=torch.ones(batch_size, k, device=device),
            alpha_mu=torch.full((batch_size, k), uniform_logit, device=device),
            alpha_factor=alpha_factor,
            alpha_diag = torch.full((batch_size, k), -0.45, device=device), #-jeffreys prior-#
            bottleneck=torch.randn(batch_size, bottleneck, device=device) * self.eps,
            parameters_per_expert=torch.randn(batch_size, e, f, device=device) * self.eps,
            bandwidth_mod=torch.ones(batch_size, e, device=device),
            amp=torch.ones(batch_size, x_in, device=device),
            trend=torch.zeros(batch_size, x_in, device=device),
            res=torch.zeros(batch_size, x_in, device=device),
            ls=torch.ones(batch_size, k, device=device),
            mu_z = torch.randn(batch_size, f, device=device) * self.eps,
            logvar_z = torch.ones(batch_size, f, device=device) * 0.05,
            lmc_matrices=initial_lmc,
            real_x=x,
            
            gp_params=GPParams(
                gates=torch.ones(batch_size, 8, device=device) * 0.125,
                periodic=torch.randn(batch_size, 32, device=device) * 0.01,
                linear=torch.randn(batch_size, 32, device=device) * 0.01,
                matern=torch.randn(batch_size, 32, device=device) * 0.01,
                rational=torch.randn(batch_size, 32, device=device) * 0.01,
                polynomial=torch.randn(batch_size, 32, device=device) * 0.01
            )
        )

    def init_pi_value(self, batch_size, device, k_atoms=30):
        pi = torch.full((batch_size, k_atoms), 1.0/k_atoms, device=device)
        pi = pi + (torch.randn_like(pi) * 0.01)
        pi = F.softmax(pi, dim=-1)
        return pi

    def refinement_loop(self, x, steps, indices=None, current_state=None, generative_mode:bool=False):
        if current_state is None:
            current_state = self.get_zero_state(x, x.device, batch_size=x.size(0))

        #hist_recons, hist_pis = [], []
        #hist_gp_features, hist_bottlenecks, hist_expert_params = [], [], []
        #hist_frequencies, hist_trends, hist_bw_mods = [], [], []
        #hist_ls, hist_gate_weights, hist_params_nkn = [], [], []
        #hist_mu_z, hist_logvar_z, hist_lmcs = [], [], []
        
        batch_size = x.size(0)
        
        # --- SEQUENCE LOOP --- #
        for t in range(steps): 
            if generative_mode and t > 0:
                x_t = current_state.recon 
            else:
                x_t = x[:, t, :] if x.dim() == 3 else x
            encoder_out = self.encoder(x_t, vae_out=current_state, indices=indices)
            
            
            dirichlet_out = self.dirichlet(
                encoder_out.z, 
                encoder_out,
                ls=encoder_out.ls,
                alpha_mu=encoder_out.alpha_mu,
                alpha_diag=encoder_out.alpha_diag,
                alpha_factor=encoder_out.alpha_factor,
                indices=indices
            )
            
            decoder_out = self.decoder(
                dirichlet_out.features,
                dirichlet_out,
                indices=indices
            )
            
            current_state = StateSpaceOutput(
                recon=decoder_out.recon,
                gp_features=decoder_out.gp_features,
                pi=decoder_out.pi,
                alpha=decoder_out.alpha,
                alpha_mu=decoder_out.alpha_mu,
                alpha_factor=decoder_out.alpha_factor,
                alpha_diag=decoder_out.alpha_diag,
                bottleneck=decoder_out.bottleneck,
                parameters_per_expert=decoder_out.parameters_per_expert,
                bandwidth_mod=decoder_out.bandwidth_mod,
                amp=decoder_out.amp,
                trend=decoder_out.trend,
                res=decoder_out.res,
                ls=decoder_out.ls,
                mu_z = decoder_out.mu_z,
                logvar_z = decoder_out.logvar_z,
                lmc_matrices=decoder_out.lmc_matrices,
                real_x=x,
            
                gp_params=GPParams(
                    gates=decoder_out.gp_params.gates,
                    periodic=decoder_out.gp_params.periodic,
                    linear=decoder_out.gp_params.linear,
                    matern=decoder_out.gp_params.matern,
                    rational=decoder_out.gp_params.rational,
                    polynomial=decoder_out.gp_params.polynomial
                )
            )
            
        return current_state
    
    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        steps = steps if steps is not None else 3
        current_state = self.refinement_loop(x=x, steps=steps, indices=indices, current_state=vae_out, generative_mode=False)
        return current_state