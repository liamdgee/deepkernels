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
from deepkernels.train.trainer import ParameterIsolate, TrainerConfig
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
        def wrapper(self, *args, **kwargs): # <--- Explicitly catch 'self'
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
        self.model = model
        
        self.n_data = kwargs.get('n_data', 76674.0)
        
        self.config = config if config is not None else TrainerConfig()
        self.device = self.get_device(device)
        self.temp = self.config.langevin_temp
        
        self.objective = EvidenceLowerBound(self.model)
        self.orchestrator = ParameterIsolate(model, device=device)
        self.adamw_optimiser, self.langevin_optimiser, self.adam_optimiser, self.debug = self.orchestrator.seperate_params_and_build_optimisers()
        self.kl_weights = self.objective.kl_weights
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
        
        self.combined_gp_epochs = self.gp_epochs + self.warmup_gp_epochs
        self.gp_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.adam_optimiser, 
            start_factor=1.0, 
            end_factor=0.01, 
            total_iters=self.combined_gp_epochs
        )
        self.langevin_clip_norm = self.config.langevin_clip_norm

        self.adamw_params = [p for g in self.adamw_optimiser.param_groups for p in g['params']]
        self.adam_params = [p for g in self.adam_optimiser.param_groups for p in g['params']]
        self.langevin_params = [p for g in self.langevin_optimiser.param_groups for p in g['params']]

        self.objective.kl_weights = {
                    "global_divergence": 0.0, 
                    "local_divergence": 0.0, 
                    "alpha_kl": 0.0, 
                    "lengthscale_kl": 0.0,
                    "inverse_wishart": 0.0,
                    "recon_kl": 0.0
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
    
    def step_vae(self, x, y, ind, global_step, annealers, total_steps):
        """Standard mini-batch step for Stage 1."""
        self.adamw_optimiser.zero_grad()
        self.langevin_optimiser.zero_grad()
        state, mvn, _ = self.model.forward(x, indices=ind, steps=3, features_only=True)
        loss, metrics = self.objective(
            model=self.model, 
            state_out=state, 
            global_step=global_step, 
            annealers=annealers,
            total_steps=total_steps,
            gp_output=mvn, 
            gp_target=y.view(-1) if y.dim() > 1 else y
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
    
    def step_gp(self, x, y, ind, annealers, global_step, total_steps, cholesky_jitter_val = 1e-2):
        """Full-batch exact step for Stage 2."""
        self.adam_optimiser.zero_grad()
        
        with gpytorch.settings.cholesky_jitter(float(cholesky_jitter_val)), \
            gpytorch.settings.max_cg_iterations(70), \
            gpytorch.settings.max_preconditioner_size(25), \
            gpytorch.settings.fast_computations(log_prob=True, solves=True), \
            gpytorch.settings.cg_tolerance(2.4), \
            gpytorch.settings.num_trace_samples(2):
            
            state, mvn, _ = self.model.forward(x, indices=ind, steps=1, features_only=False)
            loss, metrics = self.objective(
                model =self.model,
                gp_output=mvn,
                state_out=state,
                gp_target=y.view(-1) if y.dim() > 1 else y,
                global_step=global_step,
                annealers=annealers,
                total_steps=total_steps
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
    
    def evaluate(self, test_loader, epoch=0, cholesky_jitter_val = 1e-3):
        self.model.eval()
        self.objective.eval()
        k = self.k_atoms
        
        test_stats = defaultdict(float)
        n_batches = len(test_loader)

        t_preds, t_targets, t_vars = [], [], []
        logger.info(f"---Running test eval for epoch: {epoch}---")
        
        with torch.no_grad(), \
             gpytorch.settings.cholesky_jitter(float(cholesky_jitter_val)), \
             gpytorch.settings.fast_pred_var():
             
            for x, y, ind in test_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                
                state, mvn, _ = self.model.forward(x, indices=ind, features_only=False)
                
                _, metrics = self.objective(
                    model=self.model,
                    gp_output=mvn,
                    state_out=state,
                    gp_target=y.view(-1) if y.dim() > 1 else y,
                    total_steps=None,
                    annealers=None,
                    global_step=None
                )
                
                for k, v in metrics.items():
                    test_stats[k] += v
                
                if mvn is not None:
                    W_mat = state.lmc_matrices 
                    latent_mean = mvn.mean.t().unsqueeze(-1)
                    projected_mean = torch.bmm(W_mat, latent_mean).squeeze(-1)
                    latent_var = mvn.variance.t().unsqueeze(-1)
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
    def fit(self, train_loader, test_loader=None):
        logger.info(f"Starting Two-Stage Training on {self.device}")
        total_steps_gp_warmup = self.warmup_gp_epochs * len(train_loader)
        total_steps_gp_full = self.gp_epochs * len(train_loader)
        total_steps_full = self.vae_epochs * len(train_loader)
        total_steps_vae = (self.warmup_vae_epochs + self.vae_epochs) * len(train_loader)
        total_steps_warmup = self.warmup_vae_epochs * len(train_loader)
        total_steps_gp = self.warmup_gp_epochs + self.gp_epochs
        vae_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_vae, n_cycles=4, ratio=0.5, stop_beta=0.15, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(total_steps_vae, n_cycles=4, ratio=0.5, stop_beta=0.11, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.2, stop_beta=0.15, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.2, stop_beta=0.125, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.2, stop_beta=0.05, noise_scale=0.0001),
            "recon_kl": StochasticAnnealer(total_steps_vae, n_cycles=1, ratio=0.4, stop_beta=0.05, noise_scale=0.0)
        }
        gp_annealers = {
            "global_divergence": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.02, stop_beta=1.0, noise_scale=0.0001),
            "local_divergence": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.03, stop_beta=1.0, noise_scale=0.0001),
            "alpha_kl": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.45, stop_beta=1.0, noise_scale=0.0),
            "lengthscale_kl": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.45, stop_beta=1.0, noise_scale=0.0),
            "inverse_wishart": StochasticAnnealer(total_steps_gp, n_cycles=1, ratio=0.075, stop_beta=0.05, noise_scale=0.0001),
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

        logger.info(f"--- Entering Stage 0: VAE Warmup ({self.warmup_vae_epochs} Epochs) ---")
        vae_steps = 0
        for epoch in range(1, self.warmup_vae_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            self.vae_scheduler.step()
            current_lr = self.adamw_optimiser.param_groups[0]['lr']
            mlflow.log_metric("vae_lr", current_lr, step=epoch)       
            
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                vae_steps += 1
                metrics = self.step_vae(x, y, ind, annealers=warmup_annealers, global_step=vae_steps, total_steps=total_steps_warmup)
                for k, v in metrics.items(): 
                    train_stats[k] += v
            
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch, mean_train_stats, prefix="warmup")
            logger.info(f"Warmup Epoch {epoch}/{self.warmup_vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
        
        mlflow.pytorch.log_model(self.model, "model_stage1_vae_warmup")
        # ==========================================
        # STAGE 2: Train the VAE (Mini-Batches) with dirichlet module
        # ==========================================
        self.orchestrator.train_vae_and_dirichlet()
        vae_steps_full = 0
        logger.info("--- Entering Stage 1 (Full): VAE Training ---")
        for epoch in range(1, self.vae_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            self.vae_scheduler.step()
            current_lr = self.adamw_optimiser.param_groups[0]['lr']
            mlflow.log_metric("vae_lr", current_lr, step= (epoch + self.warmup_vae_epochs))
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                vae_steps_full += 1
                metrics = self.step_vae(x, y, ind, global_step=vae_steps_full, total_steps=total_steps_full, annealers=vae_annealers)
                for k, v in metrics.items(): 
                    train_stats[k] += v
            
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics((epoch + self.warmup_vae_epochs), mean_train_stats, prefix="vae_train")
            logger.info(f"VAE Epoch {epoch}/{self.vae_epochs} | Recon Loss: {mean_train_stats.get('loss_recon', 0.0):.4f}")
        
        mlflow.pytorch.log_model(self.model, "model_stage2_vae_full")

        # ==========================================
        # STAGE 3: ExactGP Warmup (Mean & Noise Calibration)
        # ==========================================
        logger.info("Flushing VRAM before entering full-batch ExactGP phase...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.orchestrator.train_gp_warmup()
        logger.info(f"--- Entering Stage 2: GP Warmup ({self.warmup_gp_epochs} Epochs) ---")
        gp_steps = 0
        for epoch in range(1, self.warmup_gp_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            epoch_idx = epoch + self.warmup_vae_epochs + self.vae_epochs
            self.gp_scheduler.step()
            current_lr = self.adam_optimiser.param_groups[0]['lr']
            mlflow.log_metric("gp_lr", current_lr, step=epoch_idx)
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                gp_steps += 1
                metrics = self.step_gp(x, y, ind, annealers=gp_annealers, global_step=gp_steps, total_steps=total_steps_gp_warmup, cholesky_jitter_val=1e-2)
                for k, v in metrics.items(): 
                    train_stats[k] += v
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch_idx, mean_train_stats, prefix="gp_warmup")
            logger.info(f"GP Warmup Epoch {epoch}/{self.warmup_gp_epochs} | VariationalELBO: {mean_train_stats.get('loss_gp', 0.0):.4f}")
        mlflow.pytorch.log_model(self.model, "model_stage3_warmup")
        # ==========================================
        # STAGE 3
        # ==========================================
        self.orchestrator.train_gp_only() #-nkn is now unfrozen-#
        
        logger.info(f"--- Entering Stage 3: Full GP ({self.gp_epochs} Epochs) ---")
        for epoch in range(1, self.gp_epochs + 1):
            self.model.train()
            self.objective.train()
            train_stats = defaultdict(float)
            epoch_idx = epoch + self.warmup_vae_epochs + self.vae_epochs + self.warmup_gp_epochs
            self.gp_scheduler.step()
            current_lr = self.adam_optimiser.param_groups[0]['lr']
            mlflow.log_metric("gp_lr", current_lr, step=epoch_idx)
            for x, y, ind in train_loader:
                x = x.to(self.device, dtype=torch.float64)
                y = y.to(self.device, dtype=torch.float64)
                ind = ind.to(self.device)
                gp_steps += 1
                metrics = self.step_gp(x, y, ind, global_step=gp_steps, annealers=gp_annealers, total_steps=total_steps_gp_full, cholesky_jitter_val=4e-3)
                for k, v in metrics.items():
                    train_stats[k] += v
            mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
            self._log_metrics(epoch_idx, mean_train_stats, prefix="gp_train")
            logger.info(f"GP Epoch {epoch}/{self.gp_epochs} | MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
        
            if test_loader and epoch % 10 == 0:
                self.evaluate(test_loader, epoch_idx)

        mlflow.pytorch.log_model(self.model, "model_stage3_full")
        # ==========================================
        # STAGE 4: Deterministic E-M Alternating Refinement
        # ==========================================
        if self.em_macro_cycles > 0:
            logger.info(f"--- Entering Stage 4: E-M Cyclical Refinement ({self.em_macro_cycles} Macro-Cycles) ---")

            current_epoch = self.warmup_vae_epochs + self.vae_epochs + self.warmup_gp_epochs + self.gp_epochs
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
                self.orchestrator.train_vae_and_dirichlet()
                for e_epoch in range(1, self.e_epochs_per_cycle + 1):
                    self.model.train()
                    self.objective.train()
                    current_epoch += 1 # Step forward in MLflow time
                    train_stats = defaultdict(float)
                    for x, y, ind in train_loader:
                        x = x.to(self.device, dtype=torch.float64)
                        y = y.to(self.device, dtype=torch.float64)
                        ind = ind.to(self.device)
                        e_steps += 1
                        metrics = self.step_vae(x, y, ind, annealers=vae_annealers, global_step=e_steps, total_steps=total_e_steps) 
                        for k, v in metrics.items(): train_stats[k] += v
                    
                    mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()} 
                    self._log_metrics(current_epoch, mean_train_stats, prefix="em_e_step")
                    logger.info(f"  [E-Step] VAE Epoch {e_epoch} | Recon: {mean_train_stats['loss_recon']:.4f}")

                # --------------------------------------------------
                # M-Step: Maximize Marginal Likelihood (Full-Batch)
                # --------------------------------------------------
                self.orchestrator.train_gp_only() # VAE Frozen
                
                for m_epoch in range(1, self.m_epochs_per_cycle + 1):
                    self.model.train()      # <--- ADD THIS
                    self.objective.train()
                    train_stats = defaultdict(float)
                    current_epoch += 1 # Step forward in MLflow time
                    for x, y, ind in train_loader:
                        x = x.to(self.device, dtype=torch.float64)
                        y = y.to(self.device, dtype=torch.float64)
                        ind = ind.to(self.device)
                        m_steps += 1
                        metrics = self.step_gp(x, y, ind, annealers=gp_annealers, global_step=m_steps, total_steps=total_m_steps, cholesky_jitter_val=6e-5)
                        for k, v in metrics.items():
                            train_stats[k] += v

                    mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()} 
                    self._log_metrics(current_epoch, mean_train_stats, prefix="em_m_step")
                    logger.info(f"  [M-Step] GP Epoch {m_epoch} | MLL: {mean_train_stats.get('loss_gp', 0.0):.4f}")
            
        mlflow.pytorch.log_model(self.model, "model_stage4_em")
        cycles = 0
        total_joint_steps = self.joint_epochs * len(train_loader)
        if self.joint_epochs > 0:
            self.orchestrator.train_cyclically() 
            logger.info(f"--- Entering Stage 5: Fully Unfrozen Joint Training ({self.joint_epochs} Epochs) ---")
            
            for epoch in range(1, self.joint_epochs + 1):
                self.model.train()      
                self.objective.train()
                train_stats = defaultdict(float)
                
                with gpytorch.settings.cholesky_jitter(1e-6), \
                    gpytorch.settings.max_cg_iterations(70), \
                    gpytorch.settings.max_preconditioner_size(25), \
                    gpytorch.settings.fast_computations(log_prob=True, solves=True), \
                    gpytorch.settings.cg_tolerance(2.4), \
                    gpytorch.settings.num_trace_samples(2):
                        
                        for x, y, ind in train_loader:
                            self.adam_optimiser.zero_grad()
                            self.adamw_optimiser.zero_grad()
                            self.langevin_optimiser.zero_grad()
                            cycles += 1
                            x = x.to(self.device, dtype=torch.float64)
                            y = y.to(self.device, dtype=torch.float64)
                            ind = ind.to(self.device)
                            state, mvn, _ = self.model.forward(x, indices=ind, features_only=False)
                            loss, metrics = self.objective(self.model, gp_output=mvn, state_out=state, gp_target=y.view(-1) if y.dim() > 1 else y, annealers=gp_annealers, global_step=cycles, total_steps=total_joint_steps)
                            
                            if not torch.isfinite(loss):
                                logger.warning("NaN/Inf loss detected. Skipping batch.")
                                continue
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.adamw_params + self.adam_params + self.langevin_params, 1.20)
                            
                            self.adam_optimiser.step()
                            self.adamw_optimiser.step()
                            self.langevin_optimiser.step()
                            
                            for k, v in metrics.items(): 
                                train_stats[k] += v
                mean_train_stats = {k: v / len(train_loader) for k, v in train_stats.items()}
                current_epoch_idx = epoch + self.warmup_vae_epochs + self.vae_epochs + self.warmup_gp_epochs + self.gp_epochs
                self._log_metrics(current_epoch_idx, mean_train_stats, prefix="joint")
                logger.info(f"Joint Epoch {epoch}/{self.joint_epochs} | Total Loss: {mean_train_stats.get('loss_total', float('nan')):.4f}")

        self.save_checkpoint("deepkernels_model.pth")
        mlflow.pytorch.log_model(self.model, "deepkernels_model")

