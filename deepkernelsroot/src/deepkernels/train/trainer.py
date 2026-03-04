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

from torch.optim import Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import LinearLR

from deepkernels.train.stochastic_annealer import StochasticAnnealer
from deepkernels.models.model import StateSpaceKernelProcess
from deepkernels.train.exact_objective import ExactObjective
from typing import Union, Optional, Iterable
import gpytorch
from tqdm import tqdm

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ParameterIsolate:
    def __init__(self, model, objective=None, device='cuda', **kwargs):
        self.model = model
        self.device = self.get_device(device)
        self.kwargs = kwargs
        self.objective = objective if objective is not None else ExactObjective(self.model)
        
        self.adam_optimiser = None
        self.adamw_optimiser = None
        self.langevin_optimiser = None

        self.encoder_module = self.model.vae.encoder
        self.decoder_module = self.model.vae.decoder
        self.dirichlet_module = self.model.vae.dirichlet

    def seperate_params_and_build_optimisers(self):
        """Executes the massive parameter routing and returns the two optimizers."""
        model = self.model.to(self.device)
        
        model.train()
        
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
        gp_mean_params = [] #-e.g. 0.01
        gp_kernel_hyperparams = [] #-limit outputscale learning-# -- set kernel lr to approx 0.005
        likelihood_params = []
        
        
        total_trainable_params = 0
        routed_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            total_trainable_params += 1
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
                if 'mean_module' in name:
                    gp_mean_params.append(param)
                    all_gp_params.append(param)
                elif 'covar_module' in name:
                    gp_kernel_hyperparams.append(param)
                    all_gp_params.append(param)
                elif 'likelihood' in name:
                    likelihood_params.append(param)
                    all_gp_params.append(param)
                else:
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
            else:
                logger.warning(f"[!] Parameter unrouted: '{name}'. Defaulting to AdamW fusion_params.")
                fusion_params.append(param)
                
            routed_params += 1

        if total_trainable_params != routed_params:
            logger.error(f"Parameter mismatch! Trainable: {total_trainable_params}, Routed: {routed_params}")
        #-build adamW optimiser-#
        #-fdr adam optimiser-#
        base_lr_adamw = self.kwargs.get('base_lr_adamw', 1.175e-3)
        
        base_decay_adamw = base_lr_adamw / 10
        slow_lr = (base_lr_adamw / 10) * 4.77    #~-5e-4
        very_slow_lr = (base_lr_adamw / 250) * 2.77 #~1.3e-5

        #-for SGLD optimiser-#
        fast_dir = self.kwargs.get('fast_dir', 1e-3)
        med_dir = self.kwargs.get('med_dir', 5e-4)
        slow_dir = self.kwargs.get('slow_dir', 1e-4)
        gamma_lr = self.kwargs.get('gamma_lr', 1e-5)
        
        ultrasensitive_lr = self.kwargs.get('ultrasensitive_lr', 5e-5)
        sensitive_lr = self.kwargs.get('sensitive_lr', 1e-4)

        gp_mean_lr = self.kwargs.get('gp_mean_lr', 7.77e-3)
        gp_likelihood_lr = self.kwargs.get('gp_likelihood_lr', 1.77e-2)
        gp_hyper_lr = self.kwargs.get("gp_hyper_lr", 3.5e-3)

        self.adamw_optimiser = torch.optim.AdamW([
            {'params': conv_params, 'lr': base_lr_adamw, 'weight_decay': base_decay_adamw},
            {'params': fusion_params, 'lr': base_lr_adamw, 'weight_decay': base_lr_adamw},
            {'params': latent_params, 'lr': slow_lr, 'weight_decay': very_slow_lr},
            {'params': deterministic_recon_params, 'lr': base_lr_adamw, 'weight_decay': base_decay_adamw},
        ])

        self.langevin_optimiser = torch.optim.RMSprop([
            {'params': dirichlet_atom_params, 'lr': med_dir},
            {'params': dirichlet_global_dist_params, 'lr': slow_dir},
            {'params': dirichlet_gamma_params, 'lr': gamma_lr},
            {'params': dirichlet_ls_params, 'lr': ultrasensitive_lr},
            {'params': dirichlet_all_nn_params, 'lr': med_dir},
            {'params': dirichlet_variational_params, 'lr': fast_dir},
            {'params': probabilistic_nn_params, 'lr': med_dir}
        ], alpha=0.975, eps=1e-7)

        self.adam_optimiser = torch.optim.Adam([
            {'params': likelihood_params, 'lr': gp_likelihood_lr},
            {'params': gp_mean_params, 'lr': gp_mean_lr},
            {'params': gp_kernel_hyperparams, 'lr': gp_hyper_lr},
            {'params': sensitive_ls_params, 'lr': sensitive_lr},
            {'params': ultrasensitive_spectral_params, 'lr': ultrasensitive_lr},
            {'params': primitive_params, 'lr': base_lr_adamw},
            {'params': combinatorics_params, 'lr': slow_lr},
        ], weight_decay=0.0)
        
        total_opt_params = sum(len(group['params']) for opt in [self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser] for group in opt.param_groups)
        logger.info(f"Parameter Isolation Complete. {total_opt_params}/{total_trainable_params} tensors strictly routed into 3 Optimizers.")
        
        self.module_groups = {
            "encoder_total": all_encoder_params,
            "decoder_total": all_decoder_params,
            "dirichlet_total": dirichlet_all_params,
            "hypernetwork_total": all_hypernetwork_params,
            "gp_total": all_gp_params
        }

        return self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser, self.module_groups
    
    def _set_group_grad(self, group_name: str, requires_grad: bool):
        """Helper to cleanly flip gradients for a specific parameter list."""
        for param in self.module_groups[group_name]:
            param.requires_grad = requires_grad

    def train_vae_only(self):
        """
        Stage 1: Standard Mini-Batch Training.
        Unfreezes the VAE. Freezes the Dirichlet mixing, NKN Hypernetwork, and GP.
        """
        self._set_group_grad("encoder_total", True)
        self._set_group_grad("decoder_total", True)
        self._set_group_grad("dirichlet_total", False)
        self._set_group_grad("hypernetwork_total", False)
        self._set_group_grad("gp_total", False)
        logger.info("Mode: VAE Only. (Use standard mini-batches)")

    def train_vae_and_dirichlet(self):
        """
        Stage 1.5: Joint representation and cluster mixing.
        """
        self._set_group_grad("encoder_total", True)
        self._set_group_grad("decoder_total", True)
        self._set_group_grad("dirichlet_total", True)
        self._set_group_grad("hypernetwork_total", False)
        self._set_group_grad("gp_total", False)
        logger.info("Mode: VAE + Dirichlet. (Use standard mini-batches)")

    def train_gp_warmup(self):
        """
        Stage 2: GP Warmup.
        Freezes the VAE and the complex NKN Hypernetwork. 
        Only trains the GP's core parameters (Mean and Likelihood noise) 
        to stabilize the exact MLL before full covariance learning.
        """
        self._set_group_grad("encoder_total", False)
        self._set_group_grad("decoder_total", False)
        self._set_group_grad("dirichlet_total", False)    
        self._set_group_grad("hypernetwork_total", False) # <-- FROZEN
        self._set_group_grad("gp_total", True)
        logger.info("Mode: GP Warmup. (Calibrating Mean and Likelihood Noise)")
    
    def train_gp_only(self):
        """
        Stage 2: Full-Batch KeOps ExactGP Training.
        Freezes the entire VAE to save VRAM. Unfreezes the Neural Kernel Network and GP.
        """
        self._set_group_grad("encoder_total", False)
        self._set_group_grad("decoder_total", False)
        self._set_group_grad("dirichlet_total", False)    
        self._set_group_grad("hypernetwork_total", True)
        self._set_group_grad("gp_total", True)
        logger.info("Mode: GP Only. (Requires passing the FULL dataset at once)")
    
    def train_cyclically(self):
        """
        Unfreezes everything. Warning: Only use if the dataset is small enough 
        that full-batch ExactGP + full VAE backprop fits in VRAM.
        """
        self._set_group_grad("encoder_total", True)
        self._set_group_grad("decoder_total", True)
        self._set_group_grad("dirichlet_total", True)
        self._set_group_grad("hypernetwork_total", True)
        self._set_group_grad("gp_total", True)
        logger.info("Mode: Fully Unfrozen / Cyclical.")

    def get_warm_start_optimiser(self, active_modules, lr=1e-3, warmup_epochs=5):
        """
        Creates a fresh optimizer to clear stale momentum states, 
        paired with a linear warmup scheduler for a smooth start.
        """
        params = []
        for module in active_modules:
            params += list(module.parameters())
        
        optimizer = Adam(params, lr=lr)
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        return optimizer, scheduler

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
    
    def check_dead_gradients(self, debug_groups):
        for group_name, params in debug_groups.items():
            zero_grads = sum(1 for p in params if p.grad is None or torch.all(p.grad == 0))
            if zero_grads > 0:
                logger.warning(f"{group_name} has {zero_grads} tensors with zero/None gradients!")
    
    def log_macro_gradients(self, debug_groups, step):
        for group_name, params in debug_groups.items():
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            
            total_norm = total_norm ** 0.5
            mlflow.log_metric(f"grad_norm_{group_name}", total_norm, step=step)