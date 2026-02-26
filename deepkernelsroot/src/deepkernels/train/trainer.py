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
from typing import Union, Optional
from tqdm import tdqm



#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--Tracking Function Decorator using mlflow--#
def tracker(kernel_experiment):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(kernel_experiment)
            with mlflow.start_run() as run:
                mlflow.log_params(kwargs)
                result = fn(*args, **kwargs)
                mlflow.set_tag("train_dict", fn.__name__)
                return result
        return wrapper
    return decorator

@tracker(kernel_experiment="Dirichlet_Mixture_Proj")
class Trainer:
    def __init__(self, model, objective, device='cuda', **kwargs):
        super().__init__()
        self.model = model
        self.gp = self.model.gp
        self.vae = self.model.vae
        self.objective = objective
        self.device = self.get_device(device)

    def train(
            self,
            dataloader,
            epochs=200,
            **kwargs
        ):
        
        model = self.model.to(self.device)
        vae = self.model.vae.to(self.device)
        gp = self.model.gp.to(self.device)
        objective = self.objective.to(self.device)
        
        model.train()
        objective.likelihood.train()
        
        #-param grouping-#
        
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
                
        self.fast_dir = kwargs.get('fast_dir', 1e-2)
        self.med_dir = kwargs.get('med_dir', 2e-3)
        self.slow_dir = kwargs.get('slow_dir', 4e-4)
        self.gamma_lr = kwargs.get('gamma_lr', 5e-5)
        self.temp = kwargs.get('langevin_temp', 7.5e-6)
        
        dirichlet_params = optim.Adagrad([
            {'params': [dirichlet.mu_atom], 'lr': self.med_dir},
            {'params': [dirichlet.log_sigma_atom], 'lr': self.med_dir},
            {'params': [dirichlet.h_mu], 'lr': self.slow_dir},
            {'params': [dirichlet.h_log_sigma], 'lr': self.slow_dir},
            {'params': [dirichlet.gamma], 'lr': self.gamma_lr},
            {'params': [dirichlet.q_mu], 'lr': self.fast_dir},
            {'params': [dirichlet.q_log_sigma], 'lr': self.fast_dir},
        ])

        optimiser = optim.AdamW([
            {'params': vae.parameters(), 'lr': 2e-3, 'weight_decay': 1.2e-4},
            {'params': gp.parameters(), 'lr': 0.015},
            {'params': objective.likelihood.parameters(), 'lr': 0.015}
        ])

        total_steps = epochs * len(dataloader)

        annealers = {
            "dirichlet_global_kl": StochasticAnnealer(total_steps, n_cycles=4, stop_beta=0.1, noise_scale=0.01),
            "dirichlet_local_kl": StochasticAnnealer(total_steps, n_cycles=4, stop_beta=0.1, noise_scale=0.01),
            "lengthscale_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0),
            "alpha_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0)
        }

        global_step = 0

        # --- THE EPOCH LOOP ---
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, x_batch in enumerate(loop):
                x_batch = x_batch.to(device) # Shape: [Batch, SeqLen, Features]
                optimiser.zero_grad()
                
                # CRITICAL: Clear the VAE's internal loss dictionary from the previous batch!
                # (Assuming your BaseGenerativeModel has a way to clear these. If it's a list/dict, clear it here)
                if hasattr(model, 'added_loss_terms'):
                    model.added_loss_terms.clear()

                # Step the annealers and update the criterion weights
                current_kl_weights = {
                    name: annealer(global_step) for name, annealer in annealers.items()
                }
                criterion.kl_weights = current_kl_weights

                # --- 1. VAE FORWARD PASS ---
                ss_out = model(x_batch)
                history = ss_out.history

                # --- 2. GP FORWARD PASS ---
                # Extract the bottleneck features to feed the GP
                # Assuming features_per_expert is [Batch, 8, N, 16] or similar
                gp_input = history.bottlenecks  
                
                # Address the shape trap (if N is missing, e.g., [Batch, 8, 16] -> [Batch, 8, 1, 16])
                if gp_input.dim() == 3:
                    gp_input = gp_input.unsqueeze(-2)

                # Package the dynamic hyperparameters from the VAE to the GP
                gp_kwargs = {
                    "gp_params": {
                        "ls_rbf": history.expert_params[..., 0], # Map these to your actual dictionary keys!
                        "w_sm": history.expert_params[..., 1],
                        # ... etc ...
                        "gates": history.gate_weights
                    },
                    "mixture_means_per_expert": history.expert_mixtures,
                    "pi": history.pis
                }

                # Run the KeOps LMC GP!
                gp_output = gp_model(gp_input, **gp_kwargs)

                # --- 3. LOSS COMPUTATION ---
                # Define what the GP is trying to predict (Target). 
                # If it's predicting the expert features themselves, pass them here.
                gp_target = history.expert_params # <-- Update to your specific target tensor
                
                loss, metrics = criterion(
                    model=model,
                    x_target=x_batch,
                    ss_history=history,
                    gp_output=gp_output,
                    gp_target=gp_target
                )

                # --- 4. BACKWARD PASS & OPTIMIZE ---
                loss.backward()
                
                # Optional but highly recommended for RNNs/VAEs: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                global_step += 1

                # --- 5. LOGGING ---
                # Update the progress bar with the most critical metrics
                loop.set_postfix(
                    Loss=f"{metrics['loss_total']:.2f}", 
                    Recon=f"{metrics['loss_recon']:.2f}",
                    GP=f"{metrics['loss_gp']:.2f}",
                    DirKL=f"{metrics.get('loss_dirichlet_global_kl', 0.0):.2f}"
                )
                
                # (Optional) Log to Weights & Biases here:
                # wandb.log(metrics, step=global_step)

        return model, gp

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