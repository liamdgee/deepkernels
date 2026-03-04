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

from deepkernels.train.exact_objective import ExactObjective
from deepkernels.train.stochastic_annealer import StochasticAnnealer
from typing import Union
from deepkernels.train.trainer import ParameterIsolate

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--Tracking Function Decorator using mlflow--#
def tracker(experiment):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment)
            with mlflow.start_run() as run:
                mlflow.log_params(kwargs)
                result = fn(*args, **kwargs)
                mlflow.set_tag("train_dict", fn.__name__)
                return result
        return wrapper
    return decorator

#---Class Definition: Stochastic Gradient Optimiser with Adaptive Langevin Dynamics--#
class LangevinTrainer:
    def __init__(self, model, device='cuda', **kwargs):
        self.model = model
        self.device = self.get_device(device)
        self.epochs = kwargs.get('total_epochs', 200)
        self.temp = kwargs.get('langevin_temp', 7.5e-6)
        
        self.objective = ExactObjective(self.model)
        self.orchestrator = ParameterIsolate(model, device=device, **kwargs)
        self.adamw_optimiser, self.sgld_optimiser, self.adam_optimiser, self.debug = self.orchestrator.seperate_params_and_build_optimisers()
        self.kl_weights = self.objective.kl_weights
        self.max_grad_norm = kwargs.get('max_grad_norm', 2.0)
        self.kwargs = kwargs
        self.n_data = kwargs.get('n_data', 38003.0)
        
        self.adamw_params = [p for g in self.adamw_optimiser.param_groups for p in g['params']]
        self.adam_params = [p for g in self.adam_optimiser.param_groups for p in g['params']]
        self.langevin_params = [p for g in self.sgld_optimiser.param_groups for p in g['params']]
        
    def _inject_langevin_noise(self, temp):
        with torch.no_grad():
            for g in self.sgld_optimiser.param_groups:
                lr = g['lr']
                noise_scale = math.sqrt((2 * lr * temp) / self.n_data)
                for p in g['params']:
                    if p.grad is not None:
                        state = self.sgld_optimiser.state[p]
                        if 'square_avg' in state:
                            G = state['square_avg']
                            precond = 1.0 / (torch.sqrt(G) + 1e-8)
                        else:
                            precond = torch.ones_like(p.data)
                        step_noise = torch.randn_like(p.data) * noise_scale * precond
                        p.add_(step_noise)
        return self
    
    def save_checkpoint(self, filepath="langevin_generative_kernel_v1.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.model.gp.likelihood.state_dict(), # Fixed path
            'sgld_optimiser_state_dict': self.sgld_optimiser.state_dict(),
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
    def step_vae(self, x):
        """Standard mini-batch step for Stage 1."""
        self.adamw_optimiser.zero_grad()
        self.sgld_optimiser.zero_grad()
        
        # run_gp=False bypasses the heavy KeOps math
        model_out = self.model(x, run_gp=False) 
        loss, metrics = self.objective(
            model=self.model, x_target=x, ss_history=model_out.history, 
            gp_output=None, gp_target=None # GP is turned off!
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adamw_params, max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.langevin_params, max_norm=self.kwargs.get('langevin_clip_norm', 15.0))

        self.adamw_optimiser.step()
        self.sgld_optimiser.step()
        self._inject_langevin_noise(temp=self.temp)
        
        return metrics

    def step_gp(self, full_x, full_y):
        """Full-batch exact step for Stage 2."""
        self.adam_optimiser.zero_grad()
        
        cholesky_jitter_val = self.kwargs.get('cholesky_jitter', 1e-3)
        
        # The GPyTorch Speed Hacks + Jitter
        with gpytorch.settings.cholesky_jitter(float(cholesky_jitter_val)), \
             gpytorch.settings.max_cg_iterations(100), \
             gpytorch.settings.cg_tolerance(2.0), \
             gpytorch.settings.num_trace_samples(2):
            
            model_out = self.model(full_x, run_gp=True)
            loss, metrics = self.objective(
                model=self.model, x_target=full_x, ss_history=model_out.history, 
                gp_output=model_out.gp_out, gp_target=full_y
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adam_params, max_norm=self.max_grad_norm)
        self.adam_optimiser.step()
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, prefix="train"):
        """Dynamically logs any metric returned by EvidenceLowerBound to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(f"{prefix}_{key}", value, step=epoch)
    
    def evaluate(self, test_loader, epoch=0):
        self.model.eval()
        self.objective.eval() 
        
        test_stats = defaultdict(float)
        n_batches = len(test_loader)

        t_preds, t_targets, t_vars = [], [], []
        cholesky_jitter_val = self.kwargs.get('cholesky_jitter', 1e-3)
        logger.info(f"---Running test eval for epoch: {epoch}---")
        
        with torch.no_grad(), \
             gpytorch.settings.cholesky_jitter(float(cholesky_jitter_val)), \
             gpytorch.settings.fast_pred_var():
             
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                out = self.model(x, run_gp=True)
                
                _, metrics = self.objective(
                    model=self.model,
                    x_target=x,
                    ss_history=out.history,
                    gp_output=None,
                    gp_target=None
                )
                
                for k, v in metrics.items():
                    test_stats[k] += v
                if out.gp_out is not None:
                    W_mat = out.history.lmc_matrices 
                    latent_mean = out.gp_out.mean.t().unsqueeze(-1)
                    projected_mean = torch.bmm(W_mat, latent_mean).squeeze(-1)
                    latent_var = out.gp_out.variance.t().unsqueeze(-1)
                    W_mat_squared = W_mat ** 2
                    projected_var = torch.bmm(W_mat_squared, latent_var).squeeze(-1)
                    batch_y = y.t() if (y.dim() == 2 and y.size(0) == self.model.k_atoms) else y
                    t_preds.append(projected_mean)
                    t_targets.append(batch_y)
                    t_vars.append(projected_var)
        
        mean_test_stats = {k: v / n_batches for k, v in test_stats.items()}

        if len(t_preds) > 0:
            preds = torch.cat(t_preds, dim=-1)
            targets = torch.cat(t_targets, dim=-1)
            vars = torch.cat(t_vars, dim=-1)

            mse = F.mse_loss(preds, targets).item()
            aggregate_uncertainty = vars.mean().item()

            mean_test_stats['mse_accuracy'] = mse
            mean_test_stats['avg_predictive_variance'] = aggregate_uncertainty
            logger.info(f"Val MSE: {mse:.4f} | Avg Uncertainty: {aggregate_uncertainty:.4f}")
        else:
            logger.warning("GP was not run during evaluation. Skipping MSE calculations.")

        self._log_metrics(epoch, mean_test_stats, prefix="val")

        return mean_test_stats

    @tracker(experiment="deepkernels")
    def fit(self, train_loader, full_x, full_y, test_loader=None, warmup_vae_epochs=50, vae_epochs=250, warmup_gp_epochs=50, gp_epochs=200,
            em_macro_cycles=8, e_epochs_per_cycle=3, m_epochs_per_cycle=5, joint_epochs=0):
        """
        full_x: [38003, Features] (Your entire dataset tensor)
        full_y: [30, 38003] (Your entire targets tensor)
        """
        logger.info(f"Starting Two-Stage Training on {self.device}")
        
        # ==========================================
        # STAGE 1 (warmup): Train the VAE alone
        # ==========================================
        #-WARMUP EPOCHS-#

        logger.info(f"--- Entering Stage 0: VAE Warmup ({warmup_vae_epochs} Epochs) ---")
        for epoch in range(1, warmup_vae_epochs + 1):
            train_stats = defaultdict(float)
            
            for x, y in train_loader:
                x = x.to(self.device)
                
                self.objective.kl_weights = {
                    "global_divergence": 2e-5, 
                    "local_divergence": 1e-5, 
                    "alpha_kl": 2e-3, 
                    "lengthscale_kl": 1e-3,
                    "inverse_wishart": 1e-4
                }
                
                metrics = self.step_vae(x) 
                for k, v in metrics.items(): train_stats[k] += v
                    
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch, mean_train_stats, prefix="warmup")
            logger.info(f"Warmup Epoch {epoch}/{warmup_vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
        
        mlflow.pytorch.log_model(self.model, "model_stage1_vae_warmup")
        # ==========================================
        # STAGE 2: Train the VAE (Mini-Batches) with dirichlet module
        # ==========================================
        self.orchestrator.train_vae_and_dirichlet()
        total_steps = vae_epochs * len(train_loader)
        iw_stop_beta = self.kwargs.get('iw_stop_beta', 0.1) # Get from argparse!

        annealers = {
            "global_divergence": StochasticAnnealer(total_steps, n_cycles=4, ratio=0.5, stop_beta=0.1, noise_scale=0.01),
            "local_divergence": StochasticAnnealer(total_steps, n_cycles=4, ratio=0.5, stop_beta=0.1, noise_scale=0.01),
            "alpha_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=iw_stop_beta, noise_scale=0.0)
        }

        logger.info("--- Entering Stage 1 (Full): VAE Training ---")
        global_step = 0
        for epoch in range(1, vae_epochs + 1):
            train_stats = defaultdict(float)
            for x, y in train_loader:
                x = x.to(self.device)
                current_kl_weights = {name: annealer(global_step) for name, annealer in annealers.items()}
                self.objective.kl_weights = current_kl_weights
                metrics = self.step_vae(x) 
                for k, v in metrics.items(): 
                    train_stats[k] += v
                global_step += 1
            
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch + warmup_vae_epochs, mean_train_stats, prefix="vae_train")
            logger.info(f"VAE Epoch {epoch}/{vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
        
        mlflow.pytorch.log_model(self.model, "model_stage2_vae_full")

        # ==========================================
        # STAGE 3: ExactGP Warmup (Mean & Noise Calibration)
        # ==========================================
        self.orchestrator.train_gp_warmup()
        full_x, full_y = full_x.to(self.device), full_y.to(self.device)
        
        logger.info(f"--- Entering Stage 2: GP Warmup ({warmup_gp_epochs} Epochs) ---")
        for epoch in range(1, warmup_gp_epochs + 1):
            self.kwargs['cholesky_jitter'] = 1e-2
            metrics = self.step_gp(full_x, full_y)
            
            epoch_idx = epoch + warmup_vae_epochs + vae_epochs
            self._log_metrics(epoch_idx, metrics, prefix="gp_warmup")
            logger.info(f"GP Warmup Epoch {epoch}/{warmup_gp_epochs} | Exact MLL: {metrics.get('loss_gp', 0.0):.4f}")
        mlflow.pytorch.log_model(self.model, "model_stage3_warmup")
        # ==========================================
        # STAGE 3: Full KeOps ExactGP (Hypernetwork Unfrozen)
        # ==========================================
        self.orchestrator.train_gp_only() #-nkn is now unfrozen-#
        
        logger.info(f"--- Entering Stage 3: Full ExactGP ({gp_epochs} Epochs) ---")
        for epoch in range(1, gp_epochs + 1):
            self.kwargs['cholesky_jitter'] = 2e-4
            metrics = self.step_gp(full_x, full_y)     
            epoch_idx = epoch + warmup_vae_epochs + vae_epochs + warmup_gp_epochs
            self._log_metrics(epoch_idx, metrics, prefix="gp_train")
            logger.info(f"GP Epoch {epoch}/{gp_epochs} | Exact MLL: {metrics.get('loss_gp', 0.0):.4f}")
            
            if test_loader and epoch % 10 == 0:
                self.evaluate(test_loader, epoch_idx)

        mlflow.pytorch.log_model(self.model, "model_stage3_full")
        # ==========================================
        # STAGE 4: Deterministic E-M Alternating Refinement
        # ==========================================
        if em_macro_cycles > 0:
            logger.info(f"--- Entering Stage 4: E-M Cyclical Refinement ({em_macro_cycles} Macro-Cycles) ---")
            

            self.objective.kl_weights = {
                "global_divergence": 0.1, 
                "local_divergence": 0.1, 
                "alpha_kl": 1.0, 
                "lengthscale_kl": 1.0,
                "inverse_wishart": self.kwargs.get('iw_stop_beta', 0.1)
            }

            for cycle in range(1, em_macro_cycles + 1):
                logger.info(f"=== E-M Macro-Cycle {cycle}/{em_macro_cycles} ===")
                
                # --------------------------------------------------
                # E-Step: Refine Latent Representations (Mini-Batches)
                # --------------------------------------------------
                self.orchestrator.train_vae_and_dirichlet() # GP Frozen
                
                for e_epoch in range(1, e_epochs_per_cycle + 1):
                    train_stats = defaultdict(float)
                    for x, y in train_loader:
                        metrics = self.step_vae(x.to(self.device)) 
                        for k, v in metrics.items(): train_stats[k] += v
                    
                    logger.info(f"  [E-Step] VAE Epoch {e_epoch} | Recon: {train_stats['loss_recon']/len(train_loader):.4f}")

                # --------------------------------------------------
                # M-Step: Maximize Marginal Likelihood (Full-Batch)
                # --------------------------------------------------
                self.orchestrator.train_gp_only() # VAE Frozen
                
                for m_epoch in range(1, m_epochs_per_cycle + 1):
                    self.kwargs['cholesky_jitter'] = 1e-3
                    metrics = self.step_gp(full_x, full_y)
                    logger.info(f"  [M-Step] GP Epoch {m_epoch} | Exact MLL: {metrics.get('loss_gp', 0.0):.4f}")
                
            # Calculate continuous MLflow epoch index
                    current_epoch = warmup_vae_epochs + vae_epochs + warmup_gp_epochs + gp_epochs + (cycle * (e_epochs_per_cycle + m_epochs_per_cycle))
                    self._log_metrics(current_epoch, metrics, prefix="em_stage")
            
            mlflow.pytorch.log_model(self.model, "model_stage4_em")

        # ==========================================
        # STAGE 5: End-to-End Joint Fine-Tuning 
        # ==========================================
        if joint_epochs > 0:
            self.orchestrator.train_cyclically() # EVERYTHING UNFROZEN
            
            logger.info(f"--- Entering Stage 5: Fully Unfrozen Joint Training ({joint_epochs} Epochs) ---")
            
            self.kwargs['cholesky_jitter'] = 1e-2
            
            for epoch in range(1, joint_epochs + 1):
                joint_stats = defaultdict(float)
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.adam_optimiser.zero_grad()
                    self.adamw_optimiser.zero_grad()
                    self.sgld_optimiser.zero_grad()
                    with gpytorch.settings.cholesky_jitter(float(self.kwargs['cholesky_jitter'])):
                        model_out = self.model(x, run_gp=True)
                        batch_y = y.t().contiguous() if y.dim() == 2 else y
                        
                        loss, metrics = self.objective(
                            model=self.model, x_target=x, ss_history=model_out.history, 
                            gp_output=model_out.gp_out, gp_target=batch_y
                        )
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.adamw_params + self.adam_params + self.langevin_params, 1.0)
                    
                    self.adam_optimiser.step()
                    self.adamw_optimiser.step()
                    self.sgld_optimiser.step()
                    
                    for k, v in metrics.items(): joint_stats[k] += v
                
                mean_joint_stats = {k: v / len(train_loader) for k, v in joint_stats.items()}
                logger.info(f"Joint Epoch {epoch}/{joint_epochs} | Total Joint Loss: {mean_joint_stats['loss_total']:.4f}")
        
        self.save_checkpoint("deepkernels_model.pth")
        mlflow.pytorch.log_model(self.model, "deepkernels_model")

