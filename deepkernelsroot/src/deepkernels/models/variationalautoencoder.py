import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from deepkernels.models.encoder import ConvolutionalLoopEncoder, EncoderConfig
from deepkernels.models.decoder import SpectralDecoder, DecoderConfig
from deepkernels.models.dirichlet import AmortisedDirichlet, DirichletConfig
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

class StateSpaceOutput(NamedTuple):
    gates: torch.Tensor
    linear: torch.Tensor
    periodic: torch.Tensor
    polynomial: torch.Tensor
    matern: torch.Tensor
    rational: torch.Tensor
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
    def __init__(self, config_encoder=None, dirichlet_config=None, decoder_config=None, seq_len=32):
        super().__init__()
        self.config_encoder = config_encoder if config_encoder is not None else EncoderConfig()
        self.decoder_config = decoder_config if decoder_config is not None else DecoderConfig()
        self.dirichlet_config = dirichlet_config if dirichlet_config is not None else DirichletConfig()
        self.encoder = ConvolutionalLoopEncoder(config=self.config_encoder)
        self.dirichlet = AmortisedDirichlet(config=self.dirichlet_config)
        self.decoder = SpectralDecoder(config=self.decoder_config)
        self.eps = self.dirichlet_config.eps
        self.k_atoms = self.dirichlet_config.k_atoms
        self.num_latents = self.dirichlet_config.num_latents
        self.latent_dim = self.dirichlet_config.latent_dim
        self.rank = self.config_encoder.rank
        self.input_dim = self.config_encoder.input_dim
        self.bottleneck_dim = self.dirichlet_config.bottleneck_dim
        self.seq_len = seq_len

    
    def get_zero_state(self, x, device, batch_size):
        k = self.k_atoms or 30
        e = self.num_latents or 8
        f = self.latent_dim or 16
        r = self.rank or 3
        x_in = self.input_dim or 30
        bottleneck = self.bottleneck_dim or 64
        seq_len = self.seq_len or 32
        
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
            recon=torch.randn(batch_size, 1, x_in, device=device) * self.eps,
            amp=torch.ones(batch_size, 1, x_in, device=device),
            trend=torch.zeros(batch_size, 1, x_in, device=device),
            gp_features=torch.randn(batch_size, e, f, device=device) * self.eps,
            pi=init_pi,
            alpha=torch.ones(batch_size, k, device=device),
            alpha_mu=torch.full((batch_size, k), uniform_logit, device=device),
            alpha_factor=alpha_factor,
            alpha_diag = torch.full((batch_size, k), -0.45, device=device), #-jeffreys prior-#
            bottleneck=torch.randn(batch_size, bottleneck, device=device) * self.eps,
            parameters_per_expert=torch.randn(batch_size, e, f, device=device) * self.eps,
            bandwidth_mod=torch.ones(batch_size, e, device=device),
            res=torch.zeros(batch_size, x_in, device=device),
            ls=torch.ones(batch_size, k, device=device),
            mu_z = torch.randn(batch_size, f, device=device) * self.eps,
            logvar_z = torch.ones(batch_size, f, device=device) * 0.05,
            lmc_matrices=initial_lmc,
            real_x=x,
            gates=torch.ones(batch_size, 8, device=device) * 0.125,
            periodic=torch.randn(batch_size, 32, device=device) * 0.01,
            linear=torch.randn(batch_size, 32, device=device) * 0.01,
            matern=torch.randn(batch_size, 32, device=device) * 0.01,
            rational=torch.randn(batch_size, 32, device=device) * 0.01,
            polynomial=torch.randn(batch_size, 32, device=device) * 0.01
        )

    def init_pi_value(self, batch_size, device):
        pi = torch.full((batch_size, self.k_atoms), 1.0/self.k_atoms, device=device)
        pi = pi + (torch.randn_like(pi) * 0.01)
        pi = F.softmax(pi, dim=-1)
        return pi

    def refinement_loop(self, x, steps, indices=None, current_state=None, generative_mode:bool=False):
        if current_state is None:
            current_state = self.get_zero_state(x, x.device, batch_size=x.size(0))

        
        batch_size = x.size(0)
        for t in range(steps): 
            if generative_mode and t > 0:
                x_t = current_state.recon 
            else:
                x_t = x[:, t:t+1, :] if x.dim() == 3 else x.unsqueeze(1)
            
            encoder_out = self.encoder(x_t, vae_out=current_state, indices=indices)
            
            dirichlet_out = self.dirichlet(
                encoder_out.z, 
                encoder_out,
                ls=encoder_out.ls,
                alpha_mu=encoder_out.alpha_mu,
                alpha_diag=encoder_out.alpha_diag,
                alpha_factor=encoder_out.alpha_factor,
                t=t,
                indices=indices
            )
            
            decoder_out = self.decoder(
                dirichlet_out.features,
                dirichlet_out,
                t=t,
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
                real_x=encoder_out.real_x,
                gates=decoder_out.gp_params.gates,
                periodic=decoder_out.gp_params.periodic,
                linear=decoder_out.gp_params.linear,
                matern=decoder_out.gp_params.matern,
                rational=decoder_out.gp_params.rational,
                polynomial=decoder_out.gp_params.polynomial
            )
            
        return current_state
    
    def forward(self, x, vae_out=None, indices=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        steps = steps if steps is not None else 3
        current_state = self.refinement_loop(x=x, steps=steps, indices=indices, current_state=vae_out, generative_mode=False)
        return current_state