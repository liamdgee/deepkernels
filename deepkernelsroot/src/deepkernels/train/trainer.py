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

from typing import Union, Optional, Iterable
import gpytorch
from tqdm import tqdm

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from dataclasses import dataclass

@dataclass
class TrainerConfig:
    # --- Dataset & Epochs ---
    n_data: float = 87636.0
    warmup_vae_epochs: int = 0
    vae_epochs: int = 0
    warmup_gp_epochs: int = 0
    gp_epochs: int = 0
    #-epochs-#
    em_macro_cycles: int = 20
    e_epochs_per_cycle: int = 3
    m_epochs_per_cycle: int = 5
    joint_epochs: int = 0

    # --- AdamW Optimiser (VAE / Encoder / Decoder) ---
    base_lr_adamw: float = 0.0003
    base_decay_adamw: float = 0.000001
    slow_decay_adamw: float = 0.0000001

    # --- Adam Optimiser (GPyTorch / Kernel Network) ---
    gp_global_hyper_lr: float = 5e-5
    gp_mean_lr: float = 5e-4
    gp_likelihood_lr: float = 2e-4
    gp_lengthscale_lr: float = 8e-4
    gp_kernel_nkn_lr: float = 1e-4
    gp_variational_lr: float = 5e-4
    gp_inducing_lr: float = 1e-5

    # --- SGLD Optimiser (Dirichlet / Latent Variables) ---
    fast_dir: float = 1e-3
    med_dir: float = 5e-4
    slow_dir: float = 1e-4
    gamma_lr: float = 8e-5
    lmc_lr: float = 1e-4
    ultrasensitive_lr: float = 5e-5
    sensitive_lr: float = 1e-4
    langevin_temp: float = 5e-5
    k_atoms: int = 30
    
    # --- Gradient Clipping ---
    max_grad_norm: float = 2.0
    langevin_clip_norm: float = 7.0
    rmspropalpha: float = 0.93
    rmspropeps: float = 1e-6

class ParameterIsolate:
    def __init__(self, model, config=None, objective=None, device='cuda'):
        self.model = model
        self.config = config if config is not None else TrainerConfig()
        self.device = self.get_device(device)
        # --- VAE LR ---
        self.base_lr_adamw = self.config.base_lr_adamw
        self.slow_lr = self.base_lr_adamw / 10  # Calculated safely here
        self.base_decay_adamw = self.config.base_decay_adamw
        self.slow_decay_adamw = self.config.slow_decay_adamw
        
        # --- GP LR ---
        self.gp_global_hyper_lr = self.config.gp_global_hyper_lr
        self.gp_mean_lr = self.config.gp_mean_lr
        self.gp_likelihood_lr = self.config.gp_likelihood_lr
        self.gp_lengthscale_lr = self.config.gp_lengthscale_lr
        self.gp_kernel_nkn_lr = self.config.gp_kernel_nkn_lr
        self.gp_variational_lr = self.config.gp_variational_lr
        self.gp_inducing_lr = self.config.gp_inducing_lr

        # --- SGLD LR ---
        self.fast_dir = self.config.fast_dir
        self.med_dir = self.config.med_dir
        self.slow_dir = self.config.slow_dir
        self.gamma_lr = self.config.gamma_lr
        self.lmc_lr = self.config.lmc_lr
        self.ultrasensitive_lr = self.config.ultrasensitive_lr
        self.sensitive_lr = self.config.sensitive_lr
        self.rmspropalpha = self.config.rmspropalpha
        self.rmspropeps = self.config.rmspropeps

        self.encoder_module = self.model.vae.encoder
        self.decoder_module = self.model.vae.decoder
        self.dirichlet_module = self.model.vae.dirichlet

        self.langevin_optimiser = None
        self.adamw_optimiser = None
        self.adam_optimiser = None

    def seperate_params_and_build_optimisers(self):
        """Executes the massive parameter routing and returns the two optimizers."""
        model = self.model.to(self.device)
        
        model.train()
        
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
        dirichlet_lmc_params = []
        
        #-kernel hypernetwork params include:
        all_hypernetwork_params = []
        primitive_params = [] #- e.g 1e-3
        combinatorics_params = [] #-e.g. 5e-4
        sensitive_ls_params = []
        ultrasensitive_spectral_params = []
        
        #-gp params include:
        all_gp_params = []
        gp_mean_params = [] #-e.g. 0.01
        all_gp_kernel_hyperparams = [] #-limit outputscale learning-# -- set kernel lr to approx 0.005
        gp_kernel_global_params = []
        gp_kernel_ls_params = []
        gp_kernel_nkn_params = []
        likelihood_params = []
        gp_inducing_params = []
        gp_variational_params = []
        gp_other_params = []
        
        
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
            elif 'gp' in name or 'likelihood' in name:
                if 'likelihood' in name:
                    likelihood_params.append(param)
                    all_gp_params.append(param) 
                elif 'mean_module' in name:
                    gp_mean_params.append(param)
                    all_gp_params.append(param)
                elif 'covar_module' in name:
                    if "outputscale" in name or "amplitude" in name:
                        gp_kernel_global_params.append(param)
                        all_gp_kernel_hyperparams.append(param)
                        all_gp_params.append(param)
                    elif any(keyword in name for keyword in ["_12", "_32", "_52", "alpha", "poly", "linear", "rq", "offset", "bandwidth"]):
                        gp_kernel_ls_params.append(param)
                        all_gp_kernel_hyperparams.append(param)
                        all_gp_params.append(param)
                    elif "nkn_weights" in name:
                        gp_kernel_nkn_params.append(param)
                        all_gp_kernel_hyperparams.append(param)
                        all_gp_params.append(param)
                    else:
                        gp_kernel_ls_params.append(param)
                        all_gp_kernel_hyperparams.append(param)
                        all_gp_params.append(param)
                elif 'variational_strategy' in name or 'lmc' in name:
                    if 'inducing_points' in name:
                        gp_inducing_params.append(param)
                        all_gp_params.append(param)
                    elif 'lmc' in name:
                        gp_kernel_global_params.append(param)
                        all_gp_params.append(param)
                    else:
                        gp_variational_params.append(param)
                        all_gp_params.append(param)  
                else:
                    gp_other_params.append(param)
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
                elif 'lmc' in name:
                    dirichlet_lmc_params.append(param)
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
                all_encoder_params.append(param)
                
            routed_params += 1
        

        self.adamw_optimiser = torch.optim.AdamW([
            {'params': conv_params, 'lr': self.base_lr_adamw, 'weight_decay': self.base_decay_adamw},
            {'params': fusion_params, 'lr': self.base_lr_adamw, 'weight_decay': self.base_decay_adamw},
            {'params': latent_params, 'lr': self.slow_lr, 'weight_decay': self.slow_decay_adamw},
            {'params': deterministic_recon_params, 'lr': self.base_lr_adamw, 'weight_decay': self.base_decay_adamw},
            {'params': probabilistic_nn_params, 'lr': self.slow_lr, 'weight_decay': self.slow_decay_adamw},
        ])

        self.langevin_optimiser = torch.optim.RMSprop([
            {'params': dirichlet_atom_params, 'lr': self.med_dir},
            {'params': dirichlet_global_dist_params, 'lr': self.slow_dir},
            {'params': dirichlet_gamma_params, 'lr': self.gamma_lr},
            {'params': dirichlet_ls_params, 'lr': self.ultrasensitive_lr},
            {'params': dirichlet_all_nn_params, 'lr': self.med_dir},
            {'params': dirichlet_variational_params, 'lr': self.fast_dir},
            {'params': dirichlet_lmc_params, 'lr': self.lmc_lr},
        ], alpha=self.rmspropalpha, eps=self.rmspropeps)

        self.adam_optimiser = torch.optim.Adam([
            {'params': likelihood_params, 'lr': self.gp_likelihood_lr},
            {'params': gp_mean_params, 'lr': self.gp_mean_lr},
            {'params': gp_kernel_global_params, 'lr': self.gp_global_hyper_lr},
            {'params': gp_variational_params, 'lr': self.gp_variational_lr},
            {'params': gp_kernel_ls_params, 'lr': self.gp_lengthscale_lr},
            {'params': gp_kernel_nkn_params, 'lr': self.gp_kernel_nkn_lr},
            {'params': sensitive_ls_params, 'lr': self.sensitive_lr},
            {'params': ultrasensitive_spectral_params, 'lr': self.ultrasensitive_lr},
            {'params': primitive_params, 'lr': self.gp_kernel_nkn_lr},
            {'params': combinatorics_params, 'lr': self.gp_kernel_nkn_lr},
            {'params': gp_inducing_params, 'lr': self.gp_inducing_lr},
            {'params': gp_other_params, 'lr': self.gp_variational_lr}
        ], weight_decay=0.0)

        ##self.ngd_optimizer = gpytorch.optim.NGD(self.model.gp.variational_parameters(), lr=0.1)

        total_opt_params = sum(len(group['params']) for opt in [self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser] for group in opt.param_groups)
        logger.info(f"Parameter Isolation Complete. {total_opt_params}/{total_trainable_params} tensors strictly routed into 3 Optimizers.")
        
        if total_trainable_params != total_opt_params:
            logger.error(f"Parameter Optimizer Mismatch! Model has {total_trainable_params} trainable tensors, but optimizers only received {total_opt_params}.")

        self.module_groups = {
            "encoder_total": all_encoder_params,
            "decoder_total": all_decoder_params,
            "dirichlet_total": dirichlet_all_params,
            "hypernetwork_total": all_hypernetwork_params,
            "gp_kernels_only": all_gp_kernel_hyperparams,
            "gp_total": all_gp_params,
            "gp_warmup_safe": likelihood_params + gp_mean_params + gp_variational_params
        }

        return self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser, self.module_groups
    
    def _set_group_grad(self, group_name: str, requires_grad: bool):
        """Helper to cleanly flip gradients for a specific parameter list."""
        for param in self.module_groups[group_name]:
            param.requires_grad = requires_grad
            if not requires_grad:
                param.grad = None

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
        TRUE WARMUP: Freezes the VAE, NKN, Inducing Points, and Kernel Lengthscales.
        ONLY trains the Variational Distribution, Mean, and Likelihood Noise.
        """
        self._set_group_grad("encoder_total", False)
        self._set_group_grad("decoder_total", False)
        self._set_group_grad("dirichlet_total", False)
        self._set_group_grad("hypernetwork_total", False)
        self._set_group_grad("gp_total", False)
        self._set_group_grad("gp_warmup_safe", True)
        
        logger.info("Mode: GP Warmup. (Calibrating Variational Dist, Mean, and Noise Only)")
    
    def train_gp_only(self):
        """
        Freezes the entire VAE to save VRAM. Unfreezes the Neural Kernel Network and GP.
        """
        self._set_group_grad("encoder_total", False)
        self._set_group_grad("decoder_total", False)
        self._set_group_grad("dirichlet_total", False)    
        self._set_group_grad("hypernetwork_total", True)
        self._set_group_grad("gp_total", True)
    
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
        
        optimiser = Adam(params, lr=lr)
        scheduler = LinearLR(optimiser, start_factor=0.1, total_iters=warmup_epochs)
        return optimiser, scheduler
    
    def reset_variational_params(self):
        """
        Resets the Variational Distribution to the Identity Prior.
        Uses named_parameters to safely bypass LMC/Multitask wrappers.
        """
        logger.info("♻️ Stage 3 Transition: Clearing Warmup noise...")
        
        with torch.no_grad():
            # 1. Hunt down the exact parameters, no matter how deep they are wrapped
            for name, param in self.model.gp.named_parameters():
                if 'variational_mean' in name:
                    param.data.zero_()
                elif 'chol_variational_covar' in name:
                    param.data.zero_()
                    # Fill the diagonal with 1.0 (The Identity Prior)
                    diag_idx = torch.arange(param.size(-1), device=param.device)
                    param.data[..., diag_idx, diag_idx] = 1.0
            
            # 2. Clear GPyTorch Cache (forces recalculation on the new Stage 3 geometry)
            if hasattr(self.model.gp, 'variational_strategy'):
                self.model.gp.variational_strategy._memoize_cache.clear()
                
        logger.info("✅ Variational Reset Complete. Inducing points are now 'clean'.")
    
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