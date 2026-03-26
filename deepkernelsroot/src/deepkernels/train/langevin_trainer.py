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
from collections import defaultdict
import torch.nn.functional as F
import gpytorch
import gc

from deepkernels.train.variational_objective import EvidenceLowerBound
from deepkernels.train.stochastic_annealer import StochasticAnnealer
from typing import Union
from deepkernels.train.trainer import ParameterIsolate, TrainerConfig

import pykeops
import shutil
import os
from tqdm import tqdm

##from deepkernels.train.keras import LossMonitor

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wipe the local project cache
#rm -rf ./keops_cache/*
#--Tracking Function Decorator using mlflow--#
def tracker(experiment):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            mlflow.set_tracking_uri("sqlite:////home/liam/deepkernels/deepkernelsroot/mlruns.db")
            mlflow.set_experiment(experiment)
            with mlflow.start_run(nested=True):
                safe_params = {k: v for k, v in kwargs.items() 
                               if isinstance(v, (int, float, str, bool, list))}
                mlflow.log_params(safe_params)
                return fn(self, *args, **kwargs)
        return wrapper
    
    return decorator
#---Class Definition: Stochastic Gradient Optimiser with Adaptive Langevin Dynamics--#
class LangevinTrainer:
    def __init__(self, model, config=None, device='cuda', **kwargs):
        self.device = self.get_device(device)
        self.model = model
        self.model = self.model.to(device=self.device, dtype=torch.float64)
        self.n_data = kwargs.get('n_data', 87636.0)
        
        self.config = config if config is not None else TrainerConfig()
        
        self.temp = self.config.langevin_temp
        
        self.objective = EvidenceLowerBound(self.model, **kwargs)
        self.orchestrator = ParameterIsolate(model, device=self.device)
        self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser, self.debug = self.orchestrator.seperate_params_and_build_optimisers()
        self.max_grad_norm = self.config.max_grad_norm
        self.warmup_vae_epochs = self.config.warmup_vae_epochs
        self.vae_epochs = self.config.vae_epochs
        self.gp_epochs = self.config.gp_epochs
        self.k_atoms = self.config.k_atoms
        self.warmup_gp_epochs = self.config.warmup_gp_epochs
        self.combined_vae_epochs = self.vae_epochs + self.warmup_vae_epochs
        self.vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.adamw_optimiser, 
            T_0=max(1, self.combined_vae_epochs // 4),
            T_mult=1,
            eta_min=1e-5
        )
        self.em_macro_cycles = self.config.em_macro_cycles
        self.e_epochs_per_cycle = self.config.e_epochs_per_cycle
        self.m_epochs_per_cycle = self.config.m_epochs_per_cycle
        self.joint_epochs = self.config.joint_epochs
        
        #-global chol is here-#
        self.global_cholesky_jitter = 3e-3

        self.combined_gp_epochs = self.gp_epochs + self.warmup_gp_epochs
        
        self.gp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.adam_optimiser,
            T_max=self.combined_gp_epochs, # Decays over the exact number of GP epochs
            eta_min=1e-5
        )

        self.langevin_clip_norm = self.config.langevin_clip_norm

        self.adamw_params = [p for g in self.adamw_optimiser.param_groups for p in g['params']]
        self.adam_params = [p for g in self.adam_optimiser.param_groups for p in g['params']]
        self.langevin_params = [p for g in self.langevin_optimiser.param_groups for p in g['params']]
        
    def _inject_langevin_noise(self, temp):
        with torch.no_grad():
            for g in self.langevin_optimiser.param_groups:
                lr = g['lr']
                noise_scale = math.sqrt((2 * lr * temp) / self.n_data)
                for p in g['params']:
                    if p.grad is not None:
                        state = self.langevin_optimiser.state[p]
                        if 'square_avg' in state:
                            G = state['square_avg']
                            precond = 1.0 / (torch.sqrt(G) + 1e-7)
                        else:
                            precond = torch.ones_like(p.data)
                        step_noise = torch.randn(p.shape, device=p.device, dtype=p.dtype) * (noise_scale * precond)
                        p.add_(step_noise)
    
    def save_checkpoint(self, filepath="langevin_generative_kernel_v1.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.model.gp.likelihood.state_dict(), # Fixed path
            'sgld_optimiser_state_dict': self.langevin_optimiser.state_dict(),
            'adam_optimiser_state_dict': self.adam_optimiser.state_dict(),
            'adamw_optimiser_state_dict': self.adamw_optimiser.state_dict(),
        }, filepath)
        if mlflow.active_run():
            mlflow.log_artifact(filepath, artifact_path="model_checkpoints")
        logger.info(f"\n[+] Checkpoint safely saved to {filepath}")
        return self
    
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
    
    def step_vae(self, x, y, ind, global_step, annealers, total_steps):
        """Standard mini-batch step for Stage 1."""
        self.adamw_optimiser.zero_grad()
        self.langevin_optimiser.zero_grad()
        state, _, _ = self.model.forward(x, indices=ind, steps=3, features_only=True)
        loss, _, metrics = self.objective(
            model=self.model, 
            state_out=state, 
            global_step=global_step, 
            annealers=annealers,
            total_steps=total_steps,
            gp_output=None,
            gp_target=y
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adamw_params, max_norm=2.5)
        torch.nn.utils.clip_grad_norm_(self.langevin_params, max_norm=7.0)

        self.adamw_optimiser.step()
        self.langevin_optimiser.step()
        self._inject_langevin_noise(temp=self.temp)
        
        return metrics
    
    def export_keops_cache(self, output_filename="keops_cache"):
        """
        Finds the active PyKeOps build folder and exports it as a tar.gz
        """
        cache_dir = pykeops.get_build_folder()
        
        print(f"Locating KeOps cache at: {cache_dir}")
        
        # 2. Tar it up! (This will create 'keops_cache.tar.gz' in your working dir)
        # 'gztar' tells shutil to make a .tar.gz file
        archive_path = shutil.make_archive(output_filename, 'gztar', cache_dir)
        
        print(f"Successfully exported KeOps cache to: {archive_path}")
        return archive_path
    
    def step_gp(self, x, y, ind, annealers, global_step, total_steps):
        """Full-batch exact step for Stage 2."""
        decay_factor = global_step / max(1, total_steps)
        
        current_jitter = 3e-3 * (0.005 ** decay_factor)
        current_jitter = max(1e-4, current_jitter * 0.95)

        with gpytorch.settings.max_cholesky_size(2048), \
             gpytorch.settings.max_preconditioner_size(65), \
             gpytorch.settings.num_trace_samples(13), \
             gpytorch.settings.max_cg_iterations(888), \
             gpytorch.settings.cg_tolerance(0.025), \
             gpytorch.settings.cholesky_max_tries(7), \
             gpytorch.settings.fast_computations(solves=True, log_prob=True, covar_root_decomposition=False), \
            gpytorch.settings.cholesky_jitter(current_jitter):
        
            self.adam_optimiser.zero_grad()
            target_aligned = y.view(-1).contiguous()
            state, mvn, gp_features = self.model.forward(x, indices=ind, steps=1, features_only=False)
            loss, gp_loss, metrics = self.objective(
                model=self.model,
                gp_output=mvn,
                state_out=state,
                gp_target=target_aligned,
                global_step=global_step,
                annealers=annealers,
                total_steps=total_steps
            )
            
            
            gp_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.adam_params, max_norm=0.75
            )
            self.adam_optimiser.step()
        
        del loss, gp_loss, mvn, state, gp_features
        torch.cuda.empty_cache()
        
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if math.isnan(v) or math.isinf(v) or v > 1e10:
                metrics[k] = 1e6
            else:
                metrics[k] = v
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, prefix="train"):
        """Dynamically logs any metric returned by EvidenceLowerBound to MLflow."""
        for key, value in metrics.items():
            clean_val = value.item() if hasattr(value, 'item') else value

            mlflow.log_metric(f"{prefix}/{key}", clean_val, step=epoch)
    
    def evaluate(self, test_loader, epoch=0):
        self.model.eval()
        self.objective.eval()
        
        t_preds = []
        t_targets = []
        t_vars = []
        test_stats = defaultdict(float)
        n_batches = len(test_loader)
        logger.info(f"---Running test eval for epoch: {epoch}---")
        
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(2e-3):
             
            for x, y, ind in test_loader:
                x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                ind = ind.to(self.device, non_blocking=True)
                target_aligned = y.view(-1).contiguous()
                
                state, mvn, _ = self.model.forward(x, indices=ind, features_only=False)
                
                _, _, metrics = self.objective(
                    model=self.model,
                    gp_output=mvn,
                    state_out=state,
                    gp_target=target_aligned,
                    total_steps=None,
                    annealers=None,
                    global_step=None
                )
                
                for k, v in metrics.items():
                    test_stats[k] += v if isinstance(v, float) else v.item()
                
                if mvn is not None:
                    p_mean = mvn.mean
                    p_var = mvn.variance
                    if p_mean.dim() > 1 and p_mean.size(0) == 8:
                        p_mean = p_mean.mean(dim=0)
                        p_var = p_var.mean(dim=0) 

                    t_preds.append(p_mean.detach().cpu().flatten())
                    t_vars.append(p_var.detach().cpu().flatten())
                    t_targets.append(target_aligned.detach().cpu().flatten())
        
        mean_test_stats = {k: v / n_batches for k, v in test_stats.items()}

        if len(t_preds) > 0:
            preds = torch.cat(t_preds, dim=0)    # Should be [N_total]
            targets = torch.cat(t_targets, dim=0) # Should be [N_total]
            vars = torch.cat(t_vars, dim=0)      # Should be [N_total]
            valid_mask = torch.isfinite(preds) & torch.isfinite(targets) & torch.isfinite(vars)
            if valid_mask.any():
                clean_preds = preds[valid_mask]
                clean_targets = targets[valid_mask]
                
                mean_test_stats['val_mse'] = F.mse_loss(clean_preds, clean_targets).item()
                mean_test_stats['val_rmse'] = torch.sqrt(F.mse_loss(clean_preds, clean_targets)).item()
                mean_test_stats['val_uncertainty'] = vars[valid_mask].mean().item()
                
                logger.info(f"EPOCH {epoch} | RMSE: {mean_test_stats['val_rmse']:.4f}")

        return mean_test_stats
    
    def evaluate_accuracy(self, x, y_val, model, steps=1, generative_mode=False):
        model.eval()
        model.gp.likelihood.eval()
        state = model.vae.get_zero_state(x, device=x.device, batch_size=x.size(0))
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(1e-3):
            state, predictive_dist, _ = model.forward(x, state, steps=steps, features_only=False, generative_mode=generative_mode)
            
            nlpd = -predictive_dist.log_prob(y_val).mean().item()
            
            mean = predictive_dist.mean
            std = torch.sqrt(predictive_dist.variance)
            
            in_1_sigma = ((y_val >= mean - std) & (y_val <= mean + std)).float().mean().item()
            in_2_sigma = ((y_val >= mean - 1.96*std) & (y_val <= mean + 1.96*std)).float().mean().item()
            
            ece_1_sigma = abs(in_1_sigma - 0.6827)
            ece_2_sigma = abs(in_2_sigma - 0.9545)
            
            ece = (ece_1_sigma + ece_2_sigma) / 2.0

        return nlpd, ece
    
    @tracker(experiment="princess")
    def fit(self, train_loader, test_loader=None, joint_training: bool=True, maniac_override: bool=False):
        logger.info(f"Starting Two-Stage Training on {self.device}")
        logger.info("Calculating global target statistics for GP scaling...")
        
        batches_per_epoch = len(train_loader)
        total_steps_vae_warmup = self.warmup_vae_epochs * batches_per_epoch
        total_steps_vae_full   = self.vae_epochs * batches_per_epoch
        total_steps_gp_warmup  = self.warmup_gp_epochs * batches_per_epoch
        total_steps_gp_full    = self.gp_epochs * batches_per_epoch
        total_joint_steps      = self.joint_epochs * batches_per_epoch
        best_val_mse = float('inf')
        
        vae_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_vae_full, n_cycles=4, ratio=0.7, stop_beta=0.035, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(total_steps_vae_full, n_cycles=4, ratio=0.25, stop_beta=0.02, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(total_steps_vae_full, n_cycles=1, ratio=0.2, stop_beta=0.02, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps_vae_full, n_cycles=1, ratio=0.2, stop_beta=0.03, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps_vae_full, n_cycles=1, ratio=0.2, stop_beta=0.005, noise_scale=0.0001),
            "recon_kl": StochasticAnnealer(total_steps_vae_full, n_cycles=1, ratio=0.4, stop_beta=0.01, noise_scale=0.0)
        }

        gp_warmup_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_gp_warmup, n_cycles=1, stop_beta=0.035, ratio=0.0, noise_scale=0.0),
            "local_divergence":  StochasticAnnealer(total_steps_gp_warmup, n_cycles=1, stop_beta=0.02, ratio=0.0, noise_scale=0.0),
            "alpha_kl":          StochasticAnnealer(total_steps_gp_warmup, n_cycles=1, stop_beta=0.02, ratio=0.0, noise_scale=0.0),
            "lengthscale_kl":    StochasticAnnealer(total_steps_gp_warmup, n_cycles=1, stop_beta=0.03, ratio=0.0, noise_scale=0.0),
            "inverse_wishart":   StochasticAnnealer(total_steps_gp_warmup, n_cycles=1, stop_beta=0.005, ratio=0.0, noise_scale=0.0),
            "recon_kl":          StochasticAnnealer(total_steps_gp_warmup, n_cycles=1, stop_beta=1.0, ratio=0.0, noise_scale=0.0) 
        }

        gp_full_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_gp_full, n_cycles=4, ratio=0.45, stop_beta=0.5, noise_scale=0.0001),
            "local_divergence":  StochasticAnnealer(total_steps_gp_full, n_cycles=4, ratio=0.55, stop_beta=0.5, noise_scale=0.0001),
            "alpha_kl":          StochasticAnnealer(total_steps_gp_full, n_cycles=2, ratio=0.40, stop_beta=0.8, noise_scale=0.0),
            "lengthscale_kl":    StochasticAnnealer(total_steps_gp_full, n_cycles=2, ratio=0.40, stop_beta=0.8, noise_scale=0.0),
            "inverse_wishart":   StochasticAnnealer(total_steps_gp_full, n_cycles=1, ratio=0.25, stop_beta=0.015, noise_scale=0.0001),         
            "recon_kl":          StochasticAnnealer(total_steps_gp_full, n_cycles=4, ratio=0.50, stop_beta=1.0, noise_scale=0.0)
        }

        warmup_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_vae_warmup, n_cycles=1, ratio=0.7, stop_beta=0.0, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(total_steps_vae_warmup, n_cycles=1, ratio=0.25,stop_beta=0.0, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(total_steps_vae_warmup, n_cycles=1, stop_beta=0.0, ratio=0.50, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps_vae_warmup, n_cycles=1,  stop_beta=0.0, ratio=0.50, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps_vae_warmup, n_cycles=1,  stop_beta=0.0, ratio=0.10, noise_scale=0.0),
            "recon_kl": StochasticAnnealer(total_steps_vae_warmup, n_cycles=1, stop_beta=0.0, ratio=0.50, noise_scale=0.0)
        }

        #==========
        # STAGE 1 (warmup): Train the VAE alone
        # ==========================================
        #-WARMUP EPOCHS-#
        absolute_global_step = 0
        stage_step = 0
        global_epochs = 0
        if total_steps_vae_warmup is not None and total_steps_vae_warmup != 0:
            logger.info(f"--- Entering Stage 0: VAE Warmup ({self.warmup_vae_epochs} Epochs) ---")
            self.orchestrator.train_vae_only()
            stage_step = 0
            for epoch in range(1, self.warmup_vae_epochs + 1):
                self.model.train()
                self.objective.train()
                train_stats = defaultdict(float)
                global_epochs += 1
                
                for x, y, ind in train_loader:
                    absolute_global_step += 1
                    stage_step += 1
                    x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                    y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                    ind = ind.to(self.device, non_blocking=True)
                    metrics = self.step_vae(x, y, ind, annealers=warmup_annealers, global_step=stage_step, total_steps=total_steps_vae_warmup)
                    for k, v in metrics.items(): 
                        train_stats[k] += v if isinstance(v, float) else v.item()
                
                self.vae_scheduler.step()
                mlflow.log_metric("vae_lr", self.adamw_optimiser.param_groups[0]['lr'], step=absolute_global_step)

                mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                self._log_metrics(global_epochs, mean_train_stats, prefix="warmup")
                logger.info(f"Warmup Epoch {epoch}/{self.warmup_vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
            
            self.clear_gpytorch_caches()
            local_save_path = "model_stage1_vae_warmup.pt"
            torch.save(self.model.state_dict(), local_save_path)
            mlflow.log_artifact(local_save_path, artifact_path="checkpoints")
        
        
        # ==========================================
        # STAGE 2: Train the VAE (Mini-Batches) with dirichlet module
        # ==========================================
        if total_steps_vae_full is not None and total_steps_vae_full != 0:
            self.orchestrator.train_vae_and_dirichlet()
            stage_step = 0
            logger.info("--- Entering Stage 1 (Full): VAE Training ---")
            for epoch in range(1, self.vae_epochs + 1):
                self.model.train()
                self.objective.train()
                train_stats = defaultdict(float)
                global_epochs += 1
                for x, y, ind in train_loader:
                    absolute_global_step += 1
                    stage_step += 1
                    x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                    y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                    ind = ind.to(self.device, non_blocking=True)
                    metrics = self.step_vae(x, y, ind, global_step=stage_step, total_steps=total_steps_vae_full, annealers=vae_annealers)
                    for k, v in metrics.items(): 
                        train_stats[k] += v if isinstance(v, float) else v.item()
                
                self.vae_scheduler.step()
                mlflow.log_metric("vae_lr", self.adamw_optimiser.param_groups[0]['lr'], step=absolute_global_step)

                mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                self._log_metrics(global_epochs, mean_train_stats, prefix="vae_train")
                logger.info(f"VAE Epoch {epoch}/{self.vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
            
            self.clear_gpytorch_caches()
            local_save_path = "model_stage2_vae.pt"
            torch.save(self.model.state_dict(), local_save_path)
            mlflow.log_artifact(local_save_path, artifact_path="checkpoints")

        # ==========================================
        # STAGE 3: GP Warmup (Mean & Noise Calibration)
        # ==========================================
        if total_steps_gp_warmup is not None and total_steps_gp_warmup != 0:
            self.orchestrator.train_gp_warmup()
            logger.info(f"--- Entering Stage 2: GP Warmup ({self.warmup_gp_epochs} Epochs) ---")
            stage_step = 0
            for epoch in range(1, self.warmup_gp_epochs + 1):
                self.model.train()
                self.objective.train()
                train_stats = defaultdict(float)
                global_epochs += 1
                pbar = tqdm(train_loader, desc=f"WARMUP GP Epoch {epoch}/{self.warmup_gp_epochs}", dynamic_ncols=True, leave=True)
                
                for x, y, ind in pbar:
                    absolute_global_step += 1
                    stage_step += 1
                    x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                    y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                    ind = ind.to(self.device, non_blocking=True)
                    metrics = self.step_gp(x, y, ind, annealers=gp_warmup_annealers, global_step=stage_step, total_steps=total_steps_gp_warmup)
                    for k, v in metrics.items(): 
                        train_stats[k] += v if isinstance(v, float) else v.item()
                    
                    if stage_step % 10 == 0: # Updated to 10 so the bar feels more "alive"
                        current_loss = metrics.get('loss_gp', 0.0)
                        val = current_loss if isinstance(current_loss, float) else current_loss.item()
                        
                        pbar.set_postfix({"MLL": f"{val:.4f}"})
                
                mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                self._log_metrics(global_epochs, mean_train_stats, prefix="gp_warmup")
                self.gp_scheduler.step()
                mlflow.log_metric("gp_lr", self.adam_optimiser.param_groups[0]['lr'], step=global_epochs)

                
                logger.info(f"GP Warmup Epoch {epoch}/{self.warmup_gp_epochs} | VariationalELBO: {mean_train_stats.get('loss_gp', 0.0):.4f}")
                if test_loader:
                    val_stats = self.evaluate(test_loader, global_epochs)
                    self._log_metrics(global_epochs, val_stats, prefix="val")
                    if epoch % 10 == 0:
                        current_val_mse = val_stats.get('val_mse', float('inf'))
                        if torch.isfinite(torch.tensor(current_val_mse)) and current_val_mse < best_val_mse:
                            best_val_mse = current_val_mse
                            self.save_checkpoint("best_val_model.pth")
                            self.export_keops_cache()
                            mlflow.log_artifact("best_val_model.pth")
                            logger.info(f"New best model saved with Val MSE: {best_val_mse:.4f}")
            self.clear_gpytorch_caches()
            checkpoint_path = "model_stage3_warmup.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        # ==========================================
        # STAGE 4
        # ==========================================
        if total_steps_gp_full is not None and total_steps_gp_full != 0:
            self.orchestrator.train_gp_only() #-nkn is now unfrozen-#
            stage_step = 0
            logger.info(f"--- Entering Stage 3: Full GP ({self.gp_epochs} Epochs) ---")
            for epoch in range(1, self.gp_epochs + 1):
                self.model.train()
                self.objective.train()
                train_stats = defaultdict(float)
                global_epochs += 1
                pbar = tqdm(train_loader, desc=f"FULL GP Epoch {epoch}/{self.gp_epochs}", dynamic_ncols=True, leave=True)
                for x, y, ind in pbar:
                    absolute_global_step += 1
                    stage_step += 1
                    x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                    y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                    ind = ind.to(self.device, non_blocking=True)
                    metrics = self.step_gp(
                        x, y, ind, 
                        annealers=gp_full_annealers, 
                        global_step=stage_step, 
                        total_steps=total_steps_gp_full
                    )
                    
                    for k, v in metrics.items(): 
                        train_stats[k] += v if isinstance(v, float) else v.item()

                    if stage_step % 10 == 0:
                        current_loss = metrics.get('loss_gp', 0.0)
                        val = current_loss if isinstance(current_loss, float) else current_loss.item()
                        
                        pbar.set_postfix({"MLL": f"{val:.4f}"})
                    if absolute_global_step % 50 == 0:
                        current_loss = metrics.get('loss_gp', 0.0)
                        current_loss_val = current_loss if isinstance(current_loss, float) else current_loss.item()
                        logger.info(f"Epoch {epoch} | Step {absolute_global_step} | Batch MLL: {current_loss_val:.4f}")
                        for k, v in metrics.items():
                            val = v if isinstance(v, float) else v.item()
                            mlflow.log_metric(f"step_gp_train_{k}", val, step=absolute_global_step)
                
                mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                self._log_metrics(global_epochs, mean_train_stats, prefix="gp_train")
                self.gp_scheduler.step()
                mlflow.log_metric("gp_lr", self.adam_optimiser.param_groups[0]['lr'], step=global_epochs)
                
                logger.info(f"GP Epoch {epoch}/{self.gp_epochs} | MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
            
                if test_loader and epoch % 10 == 0:
                    try:
                        val_stats = self.evaluate(test_loader, global_epochs)
                        
                        if val_stats:
                            self._log_metrics(global_epochs, val_stats, prefix="val")
                            
                            current_val_mse = val_stats.get('val_mse', float('inf'))
                        if epoch % 10 == 0:
                            current_val_mse = val_stats.get('val_mse', float('inf'))
                            if torch.isfinite(torch.tensor(current_val_mse)) and current_val_mse < best_val_mse:
                                best_val_mse = current_val_mse
                                self.save_checkpoint("best_val_model_fullgp.pth")
                                mlflow.log_artifact("best_val_model_fullgp.pth")
                                logger.info(f"🌟 New best model saved with Val MSE: {best_val_mse:.4f}")
                                self.export_keops_cache()
                    except Exception as e:
                        logger.error(f"❌ CRITICAL: Evaluation crashed during Epoch {global_epochs}!")
                        logger.error(f"Error details: {str(e)}")
                        logger.warning("🛡️ Shield activated: Bypassing crash to keep the training loop alive.")
                        self.clear_gpytorch_caches() 
                        torch.cuda.empty_cache()

            self.clear_gpytorch_caches()
            checkpoint_path = "model_stage3_full.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        # ==========================================
        # PREP FOR STAGES 4 & 5: Linear LR Decay
        # ==========================================
        # Calculate total remaining batches for Stages 4 and 5
        if joint_training:
            total_em_batches = (self.em_macro_cycles * (self.e_epochs_per_cycle + self.m_epochs_per_cycle)) * len(train_loader)
            if maniac_override:
                total_joint_batches = self.joint_epochs * len(train_loader)
                total_finetune_steps = total_em_batches + total_joint_batches
            else:
                total_finetune_steps = total_em_batches

            logger.info(f"Initializing LinearLR for final {total_finetune_steps} fine-tuning steps.")
            linear_vae_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.adamw_optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_finetune_steps
            )
            linear_gp_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.adam_optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_finetune_steps
            )

            if self.em_macro_cycles > 0:
                logger.info(f"--- Entering Stage 4: E-M Cyclical Refinement ({self.em_macro_cycles} Macro-Cycles) ---")
                e_steps = 0
                m_steps = 0
                total_e_steps = self.em_macro_cycles * self.e_epochs_per_cycle * len(train_loader)
                total_m_steps = self.em_macro_cycles * self.m_epochs_per_cycle * len(train_loader)
                for cycle in range(1, self.em_macro_cycles + 1):
                    logger.info(f"=== E-M Macro-Cycle {cycle}/{self.em_macro_cycles} ===")
                    train_stats = defaultdict(float)
                    # --------------------------------------------------
                    # E-Step: Refine Latent Representations (Mini-Batches)
                    # --------------------------------------------------
                    self.orchestrator.train_vae_only()
                    e_epoch_pbar = tqdm(range(1, self.e_epochs_per_cycle + 1), desc=f"Cycle {cycle} [E-Step]", unit="epoch")
                    for e_epoch in e_epoch_pbar:
                        self.model.train()
                        self.objective.train()
                        global_epochs += 1
                        train_stats = defaultdict(float)
                        for x, y, ind in tqdm(train_loader, desc=f"E-Epoch {e_epoch}", leave=False, unit="batch"):
                            x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                            y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                            ind = ind.to(self.device, non_blocking=True)
                            e_steps += 1
                            metrics = self.step_vae(x, y, ind, annealers=vae_annealers, global_step=e_steps, total_steps=total_e_steps) 
                            
                            for k, v in metrics.items(): 
                                train_stats[k] += v if isinstance(v, float) else v.item()
                        linear_vae_scheduler.step()
                        mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()} 
                        self._log_metrics(global_epochs, mean_train_stats, prefix="em_e_step")
                        e_epoch_pbar.set_postfix({'Recon': f"{mean_train_stats['loss_recon']:.4f}"})
                        logger.info(f"  [E-Step] VAE Epoch {e_epoch} | Recon: {mean_train_stats['loss_recon']:.4f}")
                        if test_loader and e_epoch % 5 == 0:  # Changed to e_epoch!
                            try:
                                val_stats = self.evaluate(test_loader, global_epochs)
                                if val_stats:
                                    self._log_metrics(global_epochs, val_stats, prefix="val")
                                    current_val_mse = val_stats.get('val_mse', float('inf'))
                                    if torch.isfinite(torch.tensor(current_val_mse)) and current_val_mse < best_val_mse:
                                        best_val_mse = current_val_mse
                                        self.save_checkpoint("best_val_model_estep.pth")
                                        mlflow.log_artifact("best_val_model_estep.pth")
                                        logger.info(f"🌟 New best model saved (E-Step) with Val MSE: {best_val_mse:.4f}")
                                        self.export_keops_cache()
                            except Exception as e:
                                logger.error(f"❌ CRITICAL: Evaluation crashed during E-step Epoch {e_epoch}!")
                                logger.error(f"Error details: {str(e)}")
                                logger.warning("🛡️ Shield activated: Bypassing crash to keep the training loop alive.")
                                self.clear_gpytorch_caches() 
                                torch.cuda.empty_cache()
                    # M-Step: Maximize Marginal Likelihood (Full-Batch)
                    # --------------------------------------------------
                    self.orchestrator.train_gp_warmup() # VAE Frozen
                    m_epoch_pbar = tqdm(range(1, self.m_epochs_per_cycle + 1), desc=f"Cycle {cycle} [M-Step]", unit="epoch")
                    for m_epoch in m_epoch_pbar:
                        self.model.train()
                        self.objective.train()
                        train_stats = defaultdict(float)
                        global_epochs += 1 # Step forward in MLflow time
                        for x, y, ind in tqdm(train_loader, desc=f"M-Epoch {m_epoch}", leave=False, unit="batch"):
                            x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                            y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                            ind = ind.to(self.device, non_blocking=True)
                            m_steps += 1
                            metrics = self.step_gp(x, y, ind, annealers=gp_full_annealers, global_step=m_steps, total_steps=total_m_steps)
                            for k, v in metrics.items(): 
                                train_stats[k] += v if isinstance(v, float) else v.item()
                        linear_gp_scheduler.step()
                        mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()} 
                        self._log_metrics(global_epochs, mean_train_stats, prefix="em_m_step")
                        m_epoch_pbar.set_postfix({'MLL': f"{mean_train_stats.get('loss_gp', 0.0):.4f}"})
                        logger.info(f"  [M-Step] GP Epoch {m_epoch} | MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
                        
                        if test_loader and m_epoch % 5 == 0:
                            try:
                                val_stats = self.evaluate(test_loader, global_epochs)
                                if val_stats:
                                    self._log_metrics(global_epochs, val_stats, prefix="val")
                                    current_val_mse = val_stats.get('val_mse', float('inf'))
                                    if torch.isfinite(torch.tensor(current_val_mse)) and current_val_mse < best_val_mse:
                                        best_val_mse = current_val_mse
                                        self.save_checkpoint("best_val_model_mstep.pth")
                                        mlflow.log_artifact("best_val_model_mstep.pth")
                                        logger.info(f"🌟 New best model saved (M-Step) with Val MSE: {best_val_mse:.4f}")
                                        self.export_keops_cache()
                            except Exception as e:
                                logger.error(f"❌ CRITICAL: Evaluation crashed during M-step Epoch {m_epoch}!")
                                logger.error(f"Error details: {str(e)}")
                                logger.warning("🛡️ Shield activated: Bypassing crash to keep the training loop alive.")
                                self.clear_gpytorch_caches() 
                                torch.cuda.empty_cache()
            self.clear_gpytorch_caches()
            checkpoint_path = "model_stage_em_full.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        if maniac_override:
            total_very_risky_steps = (self.joint_epochs * len(train_loader))
            linear_langevin_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.langevin_optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_very_risky_steps
            )
            linear_vae_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.adamw_optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_very_risky_steps
            )
            linear_gp_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.adam_optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_very_risky_steps
            )
            if self.joint_epochs > 0:
                self.orchestrator.train_cyclically()
                cycles = 0
                total_joint_steps = self.joint_epochs * len(train_loader)
                logger.info(f"--- Entering Stage 5: Fully Unfrozen Joint Training ({self.joint_epochs} Epochs) ---")
                
                for epoch in range(1, self.joint_epochs + 1):
                    self.model.train()      
                    self.objective.train()
                    train_stats = defaultdict(float)
                    global_epochs += 1
                    
                    with gpytorch.settings.cholesky_jitter(self.global_cholesky_jitter), \
                        gpytorch.settings.max_preconditioner_size(15), \
                        gpytorch.settings.max_cg_iterations(30), \
                        gpytorch.settings.cg_tolerance(0.07), \
                        gpytorch.settings.num_trace_samples(1), \
                        gpytorch.settings.max_root_decomposition_size(25), \
                        gpytorch.settings.eval_cg_tolerance(0.01), \
                        gpytorch.settings.fast_computations(False, True, False):
                            
                            for x, y, ind in train_loader:
                                x = x.to(self.device, dtype=torch.float64, non_blocking=True)
                                y = y.to(self.device, dtype=torch.float64, non_blocking=True)
                                ind = ind.to(self.device, non_blocking=True)
                                self.adam_optimiser.zero_grad()
                                self.adamw_optimiser.zero_grad()
                                self.langevin_optimiser.zero_grad()
                                cycles += 1
                                state, mvn, _ = self.model.forward(x, indices=ind, features_only=False)
                                loss, gp_loss, metrics = self.objective(self.model, gp_output=mvn, state_out=state, gp_target=y_scaled, annealers=gp_full_annealers, global_step=cycles, total_steps=total_joint_steps)
                                if not torch.isfinite(loss):
                                    logger.warning("NaN/Inf loss detected. Skipping batch.")
                                    logger.info(f"0a. Total Loss value         : {loss.item()}")
                                    logger.info(f"0b. GP Loss value            : {gp_loss.item()}")
                                    continue
                                
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.adamw_params, max_norm=self.max_grad_norm)
                                torch.nn.utils.clip_grad_norm_(self.adam_params, max_norm=self.max_grad_norm)
                                torch.nn.utils.clip_grad_norm_(self.langevin_params, max_norm=self.langevin_clip_norm)
                                
                                self.adam_optimiser.step()
                                self.adamw_optimiser.step()
                                self.langevin_optimiser.step()
                                
                                for k, v in metrics.items(): 
                                    train_stats[k] += v if isinstance(v, float) else v.item()
                    mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                    linear_vae_scheduler.step()
                    linear_gp_scheduler.step()
                    linear_langevin_scheduler.step()
                    self._log_metrics(global_epochs, mean_train_stats, prefix="joint")
                    logger.info(f"Joint Epoch {epoch}/{self.joint_epochs} | Total Loss: {mean_train_stats.get('loss_total', float('nan')):.4f}")
        return best_val_mse

    def clear_gpytorch_caches(self):
        """Safely purges GPyTorch memoization caches to prevent serialization deadlocks."""
        for module in self.model.modules():
            if hasattr(module, '_memoize_cache'):
                module._memoize_cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

