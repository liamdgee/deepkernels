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

from deepkernels.train.variational_objective import EvidenceLowerBound
from deepkernels.train.stochastic_annealer import StochasticAnnealer
from typing import Union
from deepkernels.train.trainer import ParameterIsolate
import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

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
            with mlflow.start_run(nested=True) as run:
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
        self.temp = kwargs.get('langevin_temp', 7.5e-6)
        self.n_data = kwargs.get('n_data', 76674.0)
        self.objective = EvidenceLowerBound(self.model)
        self.orchestrator = ParameterIsolate(model, device=device, **kwargs)
        self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser, self.debug = self.orchestrator.seperate_params_and_build_optimisers()
        self.kl_weights = self.objective.kl_weights
        self.max_grad_norm = kwargs.get('max_grad_norm', 2.0)
        self.kwargs = kwargs
        warmup_vae_epochs = kwargs.get('warmup_vae_epochs', 50)
        vae_epochs = kwargs.get('vae_epochs', 250)
        combined_vae_epochs = vae_epochs + warmup_vae_epochs
        self.vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.adamw_optimiser, 
            T_0=max(1, combined_vae_epochs // 4),
            T_mult=1,
            eta_min=1e-5
        )
        gp_epochs = kwargs.get('gp_epochs', 200)
        warmup_gp_epochs = kwargs.get('warmup_gp_epochs', 100)
        combined_gp_epochs = gp_epochs + warmup_gp_epochs
        self.gp_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.adam_optimiser, 
            start_factor=1.0, 
            end_factor=0.01, 
            total_iters=combined_gp_epochs
        )
        self.langevin_clip_norm = self.kwargs.get('langevin_clip_norm', 4.0)
        self.adamw_params = [p for g in self.adamw_optimiser.param_groups for p in g['params']]
        self.adam_params = [p for g in self.adam_optimiser.param_groups for p in g['params']]
        self.langevin_params = [p for g in self.langevin_optimiser.param_groups for p in g['params']]

        self.objective.kl_weights = {
                    "global_divergence": 0.03, 
                    "local_divergence": 0.01, 
                    "alpha_kl": 0.02, 
                    "lengthscale_kl": 0.1,
                    "inverse_wishart": 0.95,
                    "recon_kl": 0.1
                }
        
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
                        step_noise = torch.randn_like(p.data) * noise_scale * precond
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
    def step_vae(self, x, y, ind, global_step, annealers, total_steps, **kwargs):
        """Standard mini-batch step for Stage 1."""
        self.adamw_optimiser.zero_grad()
        self.langevin_optimiser.zero_grad()
        state = self.model(x, indices=ind, run_gp=False, **kwargs)
        loss, metrics = self.objective(
            model=self.model, state_out=state, global_step=global_step, annealers=annealers,
            total_steps=total_steps,
            gp_output= state.gp_out if hasattr(state, 'gp_out') else None, 
            gp_target=y, 
            **kwargs
        )

        if not torch.isfinite(loss):
            logger.warning("NaN/Inf loss detected in VAE step. Skipping batch.")
            self.adamw_optimiser.zero_grad()
            self.langevin_optimiser.zero_grad()
            return metrics

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adamw_params, max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.langevin_params, max_norm=self.langevin_clip_norm)

        self.adamw_optimiser.step()
        self.langevin_optimiser.step()
        self._inject_langevin_noise(temp=self.temp)
        
        return metrics
    
    def step_gp(self, x, y, ind, annealers, global_step, total_steps, **kwargs):
        """Full-batch exact step for Stage 2."""
        self.adam_optimiser.zero_grad()
        
        cholesky_jitter_val = self.kwargs.get('cholesky_jitter', 1e-3)
        
        with gpytorch.settings.cholesky_jitter(float(cholesky_jitter_val)), \
            gpytorch.settings.max_cg_iterations(70), \
            gpytorch.settings.max_preconditioner_size(25), \
            gpytorch.settings.fast_computations(log_prob=True, solves=True), \
            gpytorch.settings.cg_tolerance(2.4), \
            gpytorch.settings.num_trace_samples(2):
            
            state = self.model(x, indices=ind, run_gp=True, **kwargs)
            loss, metrics = self.objective(
                model=self.model,
                gp_output=state.gp_out if hasattr(state, 'gp_out') else None,
                state_out = state,
                gp_target=y,
                global_step=global_step,
                annealers=annealers,
                total_steps=total_steps,
                **kwargs
            )
            
        
        if not torch.isfinite(loss):
            logger.warning("NaN/Inf loss detected in GP step. Skipping batch.")
            self.adam_optimiser.zero_grad()
            return metrics
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.adam_params, max_norm=self.max_grad_norm)
        self.adam_optimiser.step()
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, prefix="train"):
        """Dynamically logs any metric returned by EvidenceLowerBound to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(f"{prefix}_{key}", value, step=epoch)
    
    def evaluate(self, test_loader, epoch=0, **kwargs):
        self.model.eval()
        self.objective.eval()
        k = self.kwargs.get('k_atoms', 30)
        
        test_stats = defaultdict(float)
        n_batches = len(test_loader)

        t_preds, t_targets, t_vars = [], [], []
        cholesky_jitter_val = self.kwargs.get('cholesky_jitter', 1e-3)
        logger.info(f"---Running test eval for epoch: {epoch}---")
        
        with torch.no_grad(), \
             gpytorch.settings.cholesky_jitter(float(cholesky_jitter_val)), \
             gpytorch.settings.fast_pred_var():
             
            for x, y, ind in test_loader:
                x, y, ind = x.to(self.device), y.to(self.device), ind.to(self.device)
                
                state = self.model(x, indices=ind, run_gp=True, **kwargs)
                
                _, metrics = self.objective(
                    model=self.model,
                    x_target=x,
                    gp_output=state.gp_out if hasattr(state, 'gp_out') else None,
                    state_out = state,
                    gp_target=y,
                    **kwargs
                )
                
                
                for k, v in metrics.items():
                    test_stats[k] += v
                if state.gp_out is not None:
                    W_mat = state.state.lmc_matrices 
                    latent_mean = state.gp_out.mean.t().unsqueeze(-1)
                    projected_mean = torch.bmm(W_mat, latent_mean).squeeze(-1)
                    latent_var = state.gp_out.variance.t().unsqueeze(-1)
                    W_mat_squared = W_mat ** 2
                    projected_var = torch.bmm(W_mat_squared, latent_var).squeeze(-1)
                    batch_y = y.t() if (y.dim() == 2 and y.size(0) == k) else y
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
    def fit(self, train_loader, test_loader=None, warmup_vae_epochs=50, vae_epochs=250, warmup_gp_epochs=50, gp_epochs=200,
            em_macro_cycles=8, e_epochs_per_cycle=3, m_epochs_per_cycle=5, joint_epochs=0, **kwargs):
        """
        full_x: [38003, Features] (Your entire dataset tensor)
        full_y: [30, 38003] (Your entire targets tensor)
        """
        logger.info(f"Starting Two-Stage Training on {self.device}")
        total_steps_gp_warmup = warmup_gp_epochs * len(train_loader)
        total_steps_gp_full = gp_epochs * len(train_loader)
        total_steps_full = vae_epochs * len(train_loader)
        total_steps_vae = (warmup_vae_epochs + vae_epochs) * len(train_loader)
        total_steps_warmup = warmup_vae_epochs * len(train_loader)
        total_steps_gp = warmup_gp_epochs + gp_epochs
        iw_stop_beta = self.kwargs.get('iw_stop_beta', 0.1)
        vae_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_vae, n_cycles=4, ratio=0.5, stop_beta=0.15, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(total_steps_vae, n_cycles=4, ratio=0.5, stop_beta=0.11, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.2, stop_beta=0.15, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.2, stop_beta=0.125, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.2, stop_beta=iw_stop_beta, noise_scale=0.0001),
            "recon_kl": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.4, stop_beta=0.05, noise_scale=0.0)
        }
        gp_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.02, stop_beta=1.0, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.03, stop_beta=1.0, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.45, stop_beta=1.0, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.45, stop_beta=1.0, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.075, stop_beta=iw_stop_beta, noise_scale=0.0001),
            "recon_kl": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.4, stop_beta=1.0, noise_scale=0.0)
        }

        warmup_annealers = {
            "global_divergence": StochasticAnnealer(100, n_cycles=1,  stop_beta=0.0, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(100, n_cycles=1, stop_beta=0.0, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(100, n_cycles=1, stop_beta=0.0, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(100, n_cycles=1,  stop_beta=0.0, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(100, n_cycles=1,  stop_beta=0.0, noise_scale=0.0),
            "recon_kl": StochasticAnnealer(100, n_cycles=1, stop_beta=0.0, noise_scale=0.0)
        }
        # ==========================================
        # STAGE 1 (warmup): Train the VAE alone
        # ==========================================
        #-WARMUP EPOCHS-#

        logger.info(f"--- Entering Stage 0: VAE Warmup ({warmup_vae_epochs} Epochs) ---")
        vae_steps = 0
        for epoch in range(1, warmup_vae_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                vae_steps += 1
                metrics = self.step_vae(x, y, ind, annealers=warmup_annealers, global_step=vae_steps, total_steps=total_steps_warmup, **kwargs)
                for k, v in metrics.items(): 
                    train_stats[k] += v
                
            self.vae_scheduler.step()
            current_lr = self.adamw_optimiser.param_groups[0]['lr']
            mlflow.log_metric("vae_lr", current_lr, step=epoch)       
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch, mean_train_stats, prefix="warmup")
            logger.info(f"Warmup Epoch {epoch}/{warmup_vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
        
        mlflow.pytorch.log_model(self.model, "model_stage1_vae_warmup")
        # ==========================================
        # STAGE 2: Train the VAE (Mini-Batches) with dirichlet module
        # ==========================================
        self.orchestrator.train_vae_and_dirichlet()
        vae_steps_full = 0
        logger.info("--- Entering Stage 1 (Full): VAE Training ---")
        for epoch in range(1, vae_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                vae_steps_full += 1
                metrics = self.step_vae(x, y, ind, global_step=vae_steps_full, total_steps=total_steps_full, annealers=vae_annealers, **kwargs)
                for k, v in metrics.items(): 
                    train_stats[k] += v
                
            self.vae_scheduler.step()
            current_lr = self.adamw_optimiser.param_groups[0]['lr']
            mlflow.log_metric("vae_lr", current_lr, step=epoch+warmup_vae_epochs)
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch + warmup_vae_epochs, mean_train_stats, prefix="vae_train")
            logger.info(f"VAE Epoch {epoch}/{vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
        
        mlflow.pytorch.log_model(self.model, "model_stage2_vae_full")

        # ==========================================
        # STAGE 3: ExactGP Warmup (Mean & Noise Calibration)
        # ==========================================
        logger.info("Flushing VRAM before entering full-batch ExactGP phase...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.orchestrator.train_gp_warmup()
        logger.info(f"--- Entering Stage 2: GP Warmup ({warmup_gp_epochs} Epochs) ---")
        gp_steps = 0
        for epoch in range(1, warmup_gp_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            self.kwargs['cholesky_jitter'] = 1e-3
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                gp_steps += 1
                metrics = self.step_gp(x, y, ind, annealers=gp_annealers, global_step=gp_steps, total_steps=total_steps_gp_warmup, **kwargs)
                for k, v in metrics.items(): 
                    train_stats[k] += v
            epoch_idx = epoch + warmup_vae_epochs + vae_epochs
            self.gp_scheduler.step()
            current_lr = self.adam_optimiser.param_groups[0]['lr']
            mlflow.log_metric("gp_lr", current_lr, step=epoch_idx)
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch_idx, mean_train_stats, prefix="gp_warmup")
            logger.info(f"GP Warmup Epoch {epoch}/{warmup_gp_epochs} | Exact MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
        mlflow.pytorch.log_model(self.model, "model_stage3_warmup")
        # ==========================================
        # STAGE 3
        # ==========================================
        self.orchestrator.train_gp_only() #-nkn is now unfrozen-#
        
        logger.info(f"--- Entering Stage 3: Full ExactGP ({gp_epochs} Epochs) ---")
        for epoch in range(1, gp_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            self.kwargs['cholesky_jitter'] = 3e-4
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                gp_steps += 1
                metrics = self.step_gp(x, y, ind, global_step=gp_steps, annealers=gp_annealers, total_steps=total_steps_gp_full, **kwargs)
                for k, v in metrics.items():
                    train_stats[k] += v     
            epoch_idx = epoch + warmup_vae_epochs + vae_epochs + warmup_gp_epochs
            self.gp_scheduler.step()
            current_lr = self.adam_optimiser.param_groups[0]['lr']
            mlflow.log_metric("gp_lr", current_lr, step=epoch_idx)
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch_idx, mean_train_stats, prefix="gp_train")
            logger.info(f"GP Epoch {epoch}/{gp_epochs} | Exact MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
        
            if test_loader and epoch % 10 == 0:
                self.evaluate(test_loader, epoch_idx)

        mlflow.pytorch.log_model(self.model, "model_stage3_full")
        # ==========================================
        # STAGE 4: Deterministic E-M Alternating Refinement
        # ==========================================
        if em_macro_cycles > 0:
            logger.info(f"--- Entering Stage 4: E-M Cyclical Refinement ({em_macro_cycles} Macro-Cycles) ---")

            current_epoch = warmup_vae_epochs + vae_epochs + warmup_gp_epochs + gp_epochs
            e_steps = 0
            m_steps = 0
            total_e_steps = em_macro_cycles * e_epochs_per_cycle * len(train_loader)
            total_m_steps = em_macro_cycles * m_epochs_per_cycle * len(train_loader)
            for cycle in range(1, em_macro_cycles + 1):
                logger.info(f"=== E-M Macro-Cycle {cycle}/{em_macro_cycles} ===")
                train_stats = defaultdict(float)
                # --------------------------------------------------
                # E-Step: Refine Latent Representations (Mini-Batches)
                # --------------------------------------------------
                self.orchestrator.train_vae_and_dirichlet()
                for e_epoch in range(1, e_epochs_per_cycle + 1):
                    self.model.train()      # <--- ADD THIS
                    self.objective.train()
                    current_epoch += 1 # Step forward in MLflow time
                    train_stats = defaultdict(float)
                    for x, y, ind in train_loader:
                        x = x.to(self.device, dtype=torch.float64)
                        y = y.to(self.device, dtype=torch.float64)
                        ind = ind.to(self.device)
                        e_steps += 1
                        metrics = self.step_vae(x, y, ind, annealers=vae_annealers, global_step=e_steps, total_steps=total_e_steps, **kwargs) 
                        for k, v in metrics.items(): train_stats[k] += v
                    
                    mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()} 
                    self._log_metrics(current_epoch, mean_train_stats, prefix="em_e_step")
                    logger.info(f"  [E-Step] VAE Epoch {e_epoch} | Recon: {mean_train_stats['loss_recon']:.4f}")

                # --------------------------------------------------
                # M-Step: Maximize Marginal Likelihood (Full-Batch)
                # --------------------------------------------------
                self.orchestrator.train_gp_only() # VAE Frozen
                
                for m_epoch in range(1, m_epochs_per_cycle + 1):
                    self.model.train()      # <--- ADD THIS
                    self.objective.train()
                    train_stats = defaultdict(float)
                    current_epoch += 1 # Step forward in MLflow time
                    self.kwargs['cholesky_jitter'] = 7e-5
                    for x, y, ind in train_loader:
                        x = x.to(self.device, dtype=torch.float64)
                        y = y.to(self.device, dtype=torch.float64)
                        ind = ind.to(self.device)
                        m_steps += 1
                        metrics = self.step_gp(x, y, ind, annealers=gp_annealers, global_step=m_steps, total_steps=total_m_steps, **kwargs)
                        for k, v in metrics.items(): # <--- 2. ADD THIS
                            train_stats[k] += v

                    mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()} 
                    self._log_metrics(current_epoch, mean_train_stats, prefix="em_m_step")
                    logger.info(f"  [M-Step] GP Epoch {m_epoch} | MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
            
        mlflow.pytorch.log_model(self.model, "model_stage4_em")
        cycles = 0
        total_joint_steps = joint_epochs * len(train_loader)
        if joint_epochs > 0:
            self.orchestrator.train_cyclically() 
            logger.info(f"--- Entering Stage 5: Fully Unfrozen Joint Training ({joint_epochs} Epochs) ---")
            self.kwargs['cholesky_jitter'] = 1e-6
            
            for epoch in range(1, joint_epochs + 1):
                self.model.train()      
                self.objective.train()
                train_stats = defaultdict(float)
                
                with gpytorch.settings.cholesky_jitter(float(self.kwargs['cholesky_jitter'])), \
                    gpytorch.settings.max_cg_iterations(70), \
                    gpytorch.settings.max_preconditioner_size(25), \
                    gpytorch.settings.fast_computations(log_prob=True, solves=True), \
                    gpytorch.settings.cg_tolerance(2.4), \
                    gpytorch.settings.num_trace_samples(2):
                        
                        for x, y, ind in train_loader:
                            # 1. Zero grads INSIDE the loop
                            self.adam_optimiser.zero_grad()
                            self.adamw_optimiser.zero_grad()
                            self.langevin_optimiser.zero_grad()
                            cycles += 1
                            x = x.to(self.device, dtype=torch.float64)
                            y = y.to(self.device, dtype=torch.float64)
                            ind = ind.to(self.device)
                            state = self.model(x, indices=ind, run_gp=True, **kwargs)
                            loss, metrics = self.objective(self.model, gp_output=state.gp_out, state_out=state, gp_target=y, annealers=gp_annealers, global_step=cycles, total_steps=total_joint_steps, **kwargs)
                            
                            if not torch.isfinite(loss):
                                logger.warning("NaN/Inf loss detected. Skipping batch.")
                                continue
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.adamw_params + self.adam_params + self.langevin_params, 1.0)
                            
                            self.adam_optimiser.step()
                            self.adamw_optimiser.step()
                            self.langevin_optimiser.step()
                            
                            for k, v in metrics.items(): 
                                train_stats[k] += v
                mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                current_epoch_idx = epoch + warmup_vae_epochs + vae_epochs + warmup_gp_epochs + gp_epochs
                self._log_metrics(current_epoch_idx, mean_train_stats, prefix="joint")
                logger.info(f"Joint Epoch {epoch}/{joint_epochs} | Total Loss: {mean_train_stats.get('loss_total', float('nan')):.4f}")

        self.save_checkpoint("deepkernels_model.pth")
        mlflow.pytorch.log_model(self.model, "deepkernels_model")

