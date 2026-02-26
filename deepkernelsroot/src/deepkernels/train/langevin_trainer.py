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

from deepkernels.train.objective import EvidenceLowerBound
from deepkernels.train.stochastic_annealer import StochasticAnnealer
from typing import Union
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
    def __init__(self, model, adam_optimiser, sgld_optimiser, device='cuda', **kwargs):
        self.model = model
        self.device = self.get_device(device)
        self.epochs = kwargs.get('total_epochs', 200)
        self.temp = kwargs.get('langevin_temp', 7.5e-6)
        self.objective = EvidenceLowerBound(gp_model=self.model.gp, num_data=38003).to(self.device)
        self.adam_optimiser = adam_optimiser
        self.sgld_optimiser = sgld_optimiser
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        
    def _inject_langevin_noise(self, temp):
        for g in self.sgld_optimiser.param_groups:
            noise_scale = math.sqrt(2 * g['lr'] * temp)
            for p in g['params']:
                if p.grad is not None:
                    state = self.sgld_optimiser.state[p]
                    if 'sum' in state:
                        G = state['sum']
                    else:
                        G = torch.ones_like(p.grad)
                    precond = 1.0 / (torch.sqrt(G) + 1e-8)
                    langevin_noise = torch.randn_like(p.grad) * noise_scale * torch.sqrt(precond)
                    p.grad.add_(langevin_noise)
    
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
    
    def step(self, x, targets):
        self.model.train()
        
        self.adam_optimiser.zero_grad()
        self.sgld_optimiser.zero_grad()
        
        model_out = self.model(x)

        loss, metrics = self.objective(
            model=self.model,
            x_target=x,
            ss_history=model_out.history,
            gp_output=model_out.gp_out.mvn,
            gp_target=targets
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self._inject_langevin_noise(temp=self.temp)

        self.adam_optimiser.step()
        self.sgld_optimiser.step()
        
        return metrics
    
    def _log_metrics(self, epoch, metrics, prefix="train"):
        """Dynamically logs any metric returned by EvidenceLowerBound to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(f"{prefix}_{key}", value, step=epoch)
    
    def evaluate(self, test_loader, epoch=0):
        self.model.eval()
        self.objective.eval() # Ensure loss modules (like GPyTorch MLL) are in eval mode
        
        test_stats = defaultdict(float)
        n_batches = len(test_loader)

        t_preds, t_targets, t_vars = [], [], []

        logger.info(f"---Running test eval for epoch: {epoch}---")
        with torch.no_grad():
            for x, y, ind in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                out = self.model(x)
                
                _, metrics = self.objective(
                    model=self.model,
                    x_target=x,
                    ss_history=out.history,
                    gp_output=out.gp_out.mvn,
                    gp_target=y
                )
                
                for k, v in metrics.items():
                    test_stats[k] += v
                
                t_preds.append(out.gp_out.mu)
                t_targets.append(y)
                t_vars.append(out.gp_out.var)
        
        mean_test_stats = {k: v / n_batches for k, v in test_stats.items()}

        preds = torch.cat(t_preds)
        targets = torch.cat(t_targets)
        vars = torch.cat(t_vars)

        mse = F.mse_loss(preds, targets).item()
        aggregate_uncertainty = vars.mean().item()

        # Add global metrics to the dictionary before logging
        mean_test_stats['mse_accuracy'] = mse
        mean_test_stats['avg_predictive_variance'] = aggregate_uncertainty

        self._log_metrics(epoch, mean_test_stats, prefix="val")
        logger.info(f"Val MSE: {mse:.4f} | Avg Uncertainty: {aggregate_uncertainty:.4f}")

        return mean_test_stats
    
    @tracker(experiment="deepkernels")
    def fit(self, train_loader, test_loader=None):
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
        
        total_steps = self.epochs * len(train_loader)
        annealers = {
            "dirichlet_global_kl": StochasticAnnealer(total_steps, n_cycles=4, stop_beta=0.1, noise_scale=0.01),
            "dirichlet_local_kl": StochasticAnnealer(total_steps, n_cycles=4, stop_beta=0.1, noise_scale=0.01),
            "lengthscale_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0),
            "alpha_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0)
        }
        global_step=0
        for epoch in range(1, self.epochs + 1):
            train_stats = defaultdict(float)
            n_batches = len(train_loader)
            
            for batch_idx, (x, y, ind) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                current_kl_weights = {name: annealer(global_step) for name, annealer in annealers.items()}
                self.objective.kl_weights = current_kl_weights
                metrics = self.step(x, y) 
                
                for k, v in metrics.items():
                    train_stats[k] += v
                global_step += 1
            
            mean_train_stats = {k: v / n_batches for k, v in train_stats.items()}
            self._log_metrics(epoch, mean_train_stats, prefix="train")
            mlflow.log_metric("langevin_temp", self.temp, step=epoch)

            total_loss = mean_train_stats.get('loss_total', 0.0)
            logger.info(f"Epoch {epoch}/{self.epochs} | Loss: {total_loss:.4f}")
            
            if test_loader:
                self.evaluate(test_loader, epoch)
        
        mlflow.pytorch.log_model(self.model, "deepkernels_model")