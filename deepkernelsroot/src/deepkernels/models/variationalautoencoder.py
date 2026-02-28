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
    mu_z: torch.Tensor
    logvar_z: torch.Tensor


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
    
    def get_zero_state(self, device, batch_size=64):
        # Standard dimensions from your architecture
        k = 30 # k_atoms
        e = 8  # num_experts / num_latents
        f = 16 # latent_dim / features
        x_in = 30
        fact = k * 30
        
        init_pi = self.init_pi_value(batch_size=batch_size, device=device)
        spread = torch.linspace(0.05, 0.15, 4, device=device)
        init_sms = spread.unsqueeze(0).expand(batch_size, -1).to(device)

        return DecoderStateOutput(
            recon=torch.zeros(batch_size, x_in, device=device),
            gp_features=torch.zeros(batch_size, e, f, device=device),
            pi=init_pi,
            alpha=torch.zeros(batch_size, k, device=device),
            alpha_mu=torch.zeros(batch_size, k, device=device),
            alpha_factor=torch.ones(batch_size, fact, device=device),
            alpha_diag=torch.eye(batch_size, k, device=device),
            bottleneck=torch.zeros(batch_size, batch_size, device=device),
            parameters_per_expert=torch.zeros(batch_size, e, f, device=device),
            bandwidth_mod=torch.ones(batch_size, e, device=device),
            amp=torch.ones(batch_size, x_in, device=device),
            trend=torch.zeros(batch_size, x_in, device=device),
            res=torch.zeros(batch_size, x_in, device=device),
            ls=torch.ones(batch_size, k, device=device),
            mu_z = torch.zeros(batch_size, f, device=device),
            logvar_z = torch.ones(batch_size, f, device=device) * 0.1,
            
            gp_params=GPParams(
                gates=torch.zeros(batch_size, 16, device=device),
                ls_rbf=torch.ones(batch_size, 1, device=device),
                ls_per=torch.ones(batch_size, 1, device=device),
                p_per=torch.ones(batch_size, 1, device=device),
                ls_mat=torch.ones(batch_size, 1, device=device),
                w_sm=init_sms.clone(),
                mu_sm=init_sms.clone(),
                v_sm=torch.ones(batch_size, 4, device=device) * 0.1
            )
        )

    def init_pi_value(self, batch_size, device, k_atoms=30):
        device = self.get_device()
        pi = torch.full((batch_size, k_atoms), 1.0/k_atoms, device=device)
        pi = pi + (torch.randn_like(pi) * 0.01)
        pi = F.softmax(pi, dim=-1)
        return pi

    def refinement_loop(self, x, steps, initial_state=None, generative_mode:bool=False):
        current_state = initial_state #-this is vae_out-#

        hist_recons, hist_latents, hist_pis = [], [], []
        hist_gp_features, hist_bottlenecks, hist_expert_params = [], [], []
        hist_frequencies, hist_trends, hist_bw_mods = [], [], []
        hist_ls, hist_gate_weights, hist_params_nkn = [], [], []
        hist_mu_z, hist_logvar_z = [], []
        
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
            hist_frequencies.append(dirichlet_out.frequencies)
            hist_expert_params.append(decoder_out.parameters_per_expert)
            hist_gp_features.append(decoder_out.gp_features)
            hist_trends.append(decoder_out.trend)
            hist_ls.append(decoder_out.ls)
            hist_bw_mods.append(decoder_out.bandwidth_mod)
            hist_gate_weights.append(dirichlet_out.gated_weights)
            hist_mu_z.append(encoder_out.mu_z)
            hist_logvar_z.append(encoder_out.logvar_z)
        
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
            mu_z = torch.stack(hist_mu_z, dim=1),
            logvar_z = torch.stack(hist_logvar_z, dim=1)
        )
            
        return current_state, history
    
    def forward(self, x, vae_out=None, steps=None, batch_shape=torch.Size([]), features_only:bool=False, **params):
        steps = steps if steps is not None else 3
        current_state, history = self.refinement_loop(x=x, steps=steps, initial_state=vae_out, generative_mode=False)
        return StateSpaceOutput(state=current_state, history=history)