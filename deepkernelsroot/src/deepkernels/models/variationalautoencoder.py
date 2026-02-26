import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from deepkernels.models.encoder import ConvolutionalLoopEncoder, ConvolutionalNetwork1D, EncoderOutput
from deepkernels.models.decoder import SpectralDecoder, DecoderOutput
from deepkernels.models.dirichlet import AmortisedDirichlet, DirichletOutput
from typing import Tuple, Optional, TypeAlias, Tuple, Union, NamedTuple

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

class HistoryOutput(NamedTuple):
    gp_params: GPParams
    recons: torch.Tensor
    latents: torch.Tensor
    pis: torch.Tensor
    gp_features: torch.Tensor
    bottlenecks: torch.Tensor
    expert_params: torch.Tensor
    frequencies: torch.Tensor
    trends: torch.Tensor
    bw_mods: torch.Tensor
    ls: torch.Tensor
    gate_weights: torch.Tensor

class DecoderStateOutput(NamedTuple):
    """Structured output for the SpectralDecoder."""
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

class StateSpaceOutput(NamedTuple):
    state: DecoderStateOutput
    history: HistoryOutput

class SpectralVAE(BaseGenerativeModel):
    def __init__(self):
        super().__init__()
        self.dirichlet = AmortisedDirichlet()
        self.decoder = SpectralDecoder()
        self.encoder = ConvolutionalLoopEncoder()
        self.eps = 1e-4
    
    def refinement_loop(self, x, steps, initial_state=None, generative_mode:bool=False):
        current_state = initial_state #-this is vae_out-#

        hist_recons, hist_latents, hist_pis = [], [], []
        hist_gp_features, hist_bottlenecks, hist_expert_params = [], [], []
        hist_frequencies, hist_trends, hist_bw_mods = [], [], []
        hist_ls, hist_gate_weights, hist_params_nkn = [], [], []
        
        batch_size, seq_len, features = x.shape
        
        # --- SEQUENCE LOOP --- #
        for t in range(seq_len):
            if generative_mode and t > 0:
                x_t = current_state.recon 
            else:
                x_t = x[:, t, :]
                
            # --- REFINEMENT LOOP --- #
            for _ in range(steps):
                encoder_out = self.encoder(x_t, vae_out=current_state)
                
                alpha = encoder_out.alpha
                if alpha.dim() > 2:
                    alpha = alpha.squeeze(-1)
                
                pi_current = self.dirichlet_sample(alpha)
                
                current_ls = encoder_out.ls
                if current_ls is not None and current_ls.numel() == 0:
                    current_ls = None
                
                dirichlet_out = self.dirichlet(
                    encoder_out.z, 
                    encoder_out, 
                    pi=pi_current, 
                    ls=current_ls
                )
                
                decoder_out = self.decoder(
                    dirichlet_out.features,
                    dirichlet_out
                )
                
                current_state = decoder_out
            
            hist_params_nkn.append(decoder_out.gp_params)
            hist_recons.append(decoder_out.recon)
            hist_pis.append(dirichlet_out.pi)
            hist_latents.append(encoder_out.z)
            hist_bottlenecks.append(decoder_out.bottleneck)
            hist_frequencies.append(dirichlet_out.frequencies) # FIX 3: frequencies, not omega
            hist_expert_params.append(decoder_out.parameters_per_expert)
            hist_gp_features.append(decoder_out.gp_features)
            hist_trends.append(decoder_out.trend)
            hist_ls.append(decoder_out.ls)
            hist_bw_mods.append(decoder_out.bandwidth_mod)
            hist_gate_weights.append(dirichlet_out.gated_weights)
        
        stacked_gp_params = GPParams(
            gates=torch.stack([p.gates for p in hist_params_nkn], dim=1).unsqueeze(1),
            ls_rbf=torch.stack([p.ls_rbf for p in hist_params_nkn], dim=1).unsqueeze(1),
            ls_per=torch.stack([p.ls_per for p in hist_params_nkn], dim=1).unsqueeze(1),
            p_per=torch.stack([p.p_per for p in hist_params_nkn], dim=1).unsqueeze(1),
            ls_mat=torch.stack([p.ls_mat for p in hist_params_nkn], dim=1).unsqueeze(1),
            w_sm=torch.stack([p.w_sm for p in hist_params_nkn], dim=1).unsqueeze(1),
            mu_sm=torch.stack([p.mu_sm for p in hist_params_nkn], dim=1).unsqueeze(1),
            v_sm=torch.stack([p.v_sm for p in hist_params_nkn], dim=1).unsqueeze(1)
        )
        
        history = HistoryOutput(
            gp_params=stacked_gp_params,
            recons=torch.stack(hist_recons, dim=1),
            latents=torch.stack(hist_latents, dim=1),
            pis=torch.stack(hist_pis, dim=1),
            gp_features=torch.stack(hist_gp_features, dim=1).transpose(1, 2),
            bottlenecks=torch.stack(hist_bottlenecks, dim=1),
            expert_params=torch.stack(hist_expert_params, dim=1).transpose(1, 2),
            frequencies=torch.stack(hist_frequencies, dim=1),
            trends=torch.stack(hist_trends, dim=1),
            bw_mods=torch.stack(hist_bw_mods, dim=1),
            ls=torch.stack(hist_ls, dim=1),
            gate_weights=torch.stack(hist_gate_weights, dim=1)
        )
            
        return current_state, history
    
    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        steps = steps if steps is not None else 3
        current_state, history = self.refinement_loop(x=x, steps=steps, initial_state=vae_out, generative_mode=False)
        return StateSpaceOutput(state=current_state, history=history)