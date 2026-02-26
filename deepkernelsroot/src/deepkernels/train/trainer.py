#---Dependencies---#
import os
import logging
import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import math
import torch.nn as nn
import functools
import mlflow

from deepkernels.train.stochastic_annealer import StochasticAnnealer
from deepkernels.models.model import StateSpaceKernelProcess
from deepkernels.train.objective import EvidenceLowerBound
from deepkernels.train.langevin_trainer import LangevinTrainer
from typing import Union, Optional

from tqdm import tqdm

import torch
from tqdm import tqdm

class ParameterIsolate:
    def __init__(self, model, objective=None, device='cuda', **kwargs):
        self.model = model
        self.device = self.get_device(device)
        self.kwargs = kwargs
        self.objective = objective if objective is not None else EvidenceLowerBound(self.model)

    def seperate_params_and_build_optimisers(self):
        """Executes the massive parameter routing and returns the two optimizers."""
        model = self.model.to(self.device)
        
        model.train()
        self.objective.likelihood.train()
        
        #-GROUP MODEL PARAMS-#
        
        #-encoder params include:
        all_encoder_params = []
        conv_params = [] #-standard e.g. 1e-3
        fusion_params = [] #higher weight decay
        latent_params = [] #-penalised by beta term-# --lower lr
        
        
        #-decoder params include:
        all_decoder_params = []
        deterministic_recon_params = [] #-  e.g. 1e-3
        probabilistic_nn_params = [] #-e.g. 1e-4

        
        #-dirichlet params include:
        dirichlet_all_params = []

        dirichlet_all_nn_params = []
        dirichlet_atom_params = []
        dirichlet_global_dist_params = []
        dirichlet_variational_params = []
        dirichlet_gamma_params = []
        dirichlet_ls_params = []
        
        #-kernel hypernetwork params include:
        all_hypernetwork_params = []
        primitive_params = [] #- e.g 1e-3
        combinatorics_params = [] #-e.g. 5e-4
        sensitive_ls_params = []
        ultrasensitive_spectral_params = []
        
        #-gp params include:
        all_gp_params = []
        gp_variational_params = [] #-fast: e.g. 0.04
        gp_lmc_params = [] #-e.g. 0.015
        gp_mean_params = [] #-e.g. 0.01
        gp_kernel_hyperparams = [] #-limit outputscale learning-# -- set kernel lr to approx 0.005

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'vae.dirichlet.kernel_network' in name or 'kernel_network' in name:
                if any(k in name for k in ['param_heads.p_per', 'param_heads.mu_sm', 'param_heads.v_sm']):
                    ultrasensitive_spectral_params.append(param)
                    all_hypernetwork_params.append(param)
                elif 'param_heads.ls_' in name:
                    sensitive_ls_params.append(param)
                    all_hypernetwork_params.append(param)
                elif any(k in name for k in ['gate_head', 'param_heads.w_sm', 'selection_weights', 'complex_interactions', 'spectral_feedback_loop']):
                    combinatorics_params.append(param)
                    all_hypernetwork_params.append(param)
                elif 'linear' in name or 'rbf' in name or 'matern' in name or 'periodic' in name:
                    primitive_params.append(param)
                    all_hypernetwork_params.append(param)
                else:
                    primitive_params.append(param)
                    all_hypernetwork_params.append(param)
            elif 'vae.encoder' in name:
                if 'latent' in name or 'latent_mu' in name or 'latent_logvar' in name:
                    latent_params.append(param)
                    all_encoder_params.append(param)
                elif any(keyword in name for keyword in [
                    'stem', 'stage', 'stage1', 'stage2', 'stage3', 'pool', 'conv', 'norm', 'act',
                    'conv1', 'conv2', 'norm1', 'norm2', 'act1', 'act2', 'shortcut'
                ]):
                    conv_params.append(param)
                    all_encoder_params.append(param)
                elif 'fc' in name or 'fusion' in name:
                    fusion_params.append(param)
                    all_encoder_params.append(param)
                else:
                    fusion_params.append(param)
                    all_encoder_params.append(param)
            elif 'vae.decoder' in name:
                if any(keyword in name for keyword in ['alpha', 'lengthscale', 'expert', 'variational', 'logit', 'mu', 'logvar', 'factor', 'diag']):
                    probabilistic_nn_params.append(param)
                    all_decoder_params.append(param)
                elif 'recon' in name or 'compression' in name or 'network' in name:
                    deterministic_recon_params.append(param)
                    all_decoder_params.append(param)
                else:
                    deterministic_recon_params.append(param)
                    all_decoder_params.append(param)
            elif 'gp' in name:
                if 'variational_distribution' in name:
                    gp_variational_params.append(param)
                    all_gp_params.append(param)
                elif 'lmc_coefficients' in name:
                    gp_lmc_params.append(param)
                    all_gp_params.append(param)
                elif 'mean_module' in name:
                    gp_mean_params.append(param)
                    all_gp_params.append(param)
                elif 'covar_module' in name:
                    gp_kernel_hyperparams.append(param)
                    all_gp_params.append(param)
                else:
                    gp_kernel_hyperparams.append(param)
                    all_gp_params.append(param)
            elif 'likelihood' in name:
                gp_kernel_hyperparams.append(param)
                all_gp_params.append(param)
            elif 'vae.dirichlet' in name:
                if 'mu_atom' in name or 'log_sigma_atom' in name:
                    dirichlet_atom_params.append(param)
                    dirichlet_all_params.append(param)
                elif 'h_mu' in name or 'h_log_sigma' in name:
                    dirichlet_global_dist_params.append(param)
                    dirichlet_all_params.append(param)
                elif 'raw_gamma' in name or 'gamma' in name:
                    dirichlet_gamma_params.append(param)
                    dirichlet_all_params.append(param)
                elif 'lengthscale_log_uncertainty' in name or 'lengthscale' in name:
                    dirichlet_ls_params.append(param)
                    dirichlet_all_params.append(param)
                elif 'q_mu' in name or 'q_log_sigma' in name:
                    dirichlet_variational_params.append(param)
                    dirichlet_all_params.append(param)
                elif 'compress' in name or 'mixer' in name or 'head' in name:
                    dirichlet_all_nn_params.append(param)
                    dirichlet_all_params.append(param)
                else:
                    dirichlet_all_params.append(param)
                    dirichlet_all_nn_params.append(param)
        
        #-build adamW optimiser-#
        #-fdr adam optimiser-#
        base_lr_adamw = self.kwargs.get('base_lr_adamw', 1.175e-3)
        
        base_decay_adamw = base_lr_adamw / 10
        slow_lr = (base_lr_adamw / 10) * 4.77    #~-5e-4
        very_slow_lr = (base_lr_adamw / 250) * 2.77 #~1.3e-5

        adamw_optimiser = torch.optim.AdamW([
            {'params': conv_params, 'lr': base_lr_adamw, 'weight_decay': base_decay_adamw},
            {'params': fusion_params, 'lr': base_lr_adamw, 'weight_decay': base_lr_adamw},
            {'params': latent_params, 'lr': slow_lr, 'weight_decay': very_slow_lr},
            
            {'params': deterministic_recon_params, 'lr': base_lr_adamw, 'weight_decay': base_decay_adamw},
            {'params': probabilistic_nn_params, 'lr': slow_lr, 'weight_decay': very_slow_lr},
            
            {'params': dirichlet_all_nn_params, 'lr': base_lr_adamw},
            {'params': primitive_params, 'lr': base_lr_adamw},
            {'params': combinatorics_params, 'lr': slow_lr},
        ])

        #-for SGLD optimiser-#
        fast_dir = self.kwargs.get('fast_dir', 1e-2)
        med_dir = self.kwargs.get('med_dir', 2e-3)
        slow_dir = self.kwargs.get('slow_dir', 3.5e-4)
        gamma_lr = self.kwargs.get('gamma_lr', 1e-4)

        langevin_temp = self.kwargs.get('langevin_temp', 7.5e-6)
        
        ultrasensitive_lr = self.kwargs.get('ultrasensitive_lr', 1e-6)
        sensitive_lr = self.kwargs.get('sensitive_lr', 5e-5)

        gp_variational_lr = self.kwargs.get('gp_variational_lr', 0.045)
        gp_lmc_lr = self.kwargs.get('gp_lmc_lr', 0.01277)
        gp_mean_lr = self.kwargs.get('gp_mean_lr', 7.77e-3)
        gp_likelihood_lr = self.kwargs.get('gp_likelihood_lr', 1.77e-2)
        gp_hyper_lr = self.kwargs.get("gp_hyper_lr", 3.5e-3)

        langevin_optimiser = torch.optim.Adagrad([
            {'params': gp_variational_params, 'lr': gp_variational_lr},
            {'params': gp_lmc_params, 'lr': gp_lmc_lr},
            {'params': gp_mean_params, 'lr': gp_mean_lr},
            {'params': gp_kernel_hyperparams, 'lr': gp_hyper_lr},
            
            {'params': self.objective.likelihood.parameters(), 'lr': gp_likelihood_lr},
            
            {'params': sensitive_ls_params, 'lr': sensitive_lr},
            {'params': ultrasensitive_spectral_params, 'lr': ultrasensitive_lr}, 
            
            {'params': dirichlet_atom_params, 'lr': med_dir},
            {'params': dirichlet_global_dist_params, 'lr': slow_dir},
            {'params': dirichlet_gamma_params, 'lr': gamma_lr},
            {'params': dirichlet_ls_params, 'lr': sensitive_lr},
            {'params': dirichlet_variational_params, 'lr': fast_dir},
        ])

        return adamw_optimiser, langevin_optimiser
    
    def get_device(self, device_request: Union[str, torch.device, None] = None) -> torch.device:

            """
            Resolves the optimal available device for PyTorch operations.
            
            Priority:
            1. explicit device_request (if provided and valid)
            2. cuda:0 (NVIDIA GPU)
            3. mps (Apple Silicon Metal Performance Shaders)
            4. cpu
            
            Args:
                device_request: Optional string ('cuda', 'mps', 'cpu') or torch.device 
                                to force a specific device.
            
            Returns:
                torch.device: The resolved device.
            """
            if device_request is not None:
                device = torch.device(device_request)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    logging.warning(f"CUDA requested but unavailable. Falling back to CPU.")
                    return torch.device('cpu')
                if device.type == 'mps' and not torch.backends.mps.is_available():
                    logging.warning(f"MPS (Apple Silicon) requested but unavailable. Falling back to CPU.")
                    return torch.device('cpu')
                return device
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device('mps')
            return torch.device('cpu')