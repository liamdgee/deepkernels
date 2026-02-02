#---Dependencies---#
import os
import logging
import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import math
import torch.nn as nn
import torcn.nn.functional as F
import functools
import mlflow
import xgboost as xgb

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

#---Class Definition: Stochastic Gradient Optimiser with Adaptive Langevin Dynamics--#
@tracker(kernel_experiment="Dirichlet_Mixture_Proj")
class LangevinTrainer:
    def __init__(self, model, **kwargs):
        self.model = model
        self.device = kwargs.get('device', 'mps')
        self.epochs = kwargs.get('total_epochs', 200)

        #--Hyperparams---#
        self.eta_a = kwargs.get('lr_atoms', 2.15e-3)
        self.eta_w = kwargs.get('lr_weights', 1.25e-2)
        self.temp = kwargs.get('langevin_temp', 7.5e-6)
        self.lr_decoder = kwargs.get('stable_rkhs_lr', 2e-4)
        self.beta = kwargs.get('kl_div_beta', 0.2)
        self.rkhs_w = kwargs.get('rkhs_weights', 10.0)

        #--Param Grouping--#
        param_groups = [
            {'params': [self.model.dirichlet.mu_atom], 'lr': self.eta_a},
            {'params': [self.model.dirichlet.log_sigma_atom], 'lr': self.eta_a},
            {'params': [self.model.dirichlet.v_k], 'lr': self.eta_w},
            {'params': [self.model.dirichlet.v_j], 'lr': self.eta_w},
            {'params': self.model.decoder.parameters(), 'lr': self.lr_decoder}
        ]

        self.optimiser = optim.Adagrad(param_groups, lr=self.eta_a)
    
    def comp_kl_divergence_for_dirichlet_module(self):
        """
        Calculates divergence between learned sticks (q) and dirichlet prior (p) 
        defined by concentrations alpha & gamma
        """
        #--Fetch learned stick breaking params--#
        v_k = torch.sigmoid(self.model.dirichlet.v_k) #--Global--#
        v_j = torch.sigmoid(self.model.dirichlet.v_j) #--Local--#

        #--Dirichlet Priors from model---#
        gamma = torch.exp(self.model.dirichlet.gamma)
        alpha = torch.exp(self.model.dirichlet.alpha)

        #--KL Divergence for the GEM dist (stick breaking process mechanism)--#
        kl_global = (v_k.log() + (gamma - 1) * (1 - v_k).log()).sum()
        kl_local = (v_j.log() + (alpha - 1) * (1 - v_j).log()).sum()
        divergence = -(kl_global + kl_local)
        return divergence

    def comp_loss(self, out, **kwargs):
        """Computes loss and stores in dictionary"""

        #--Stage A: Multi-Objective Evidence Collection--
        y = kwargs.get('target_var')
        h_vit_observed = kwargs.get('h_vit_observed')

        #--Negative Log likelihood loss--#
        safe_sig = torch.clamp(out['var'], min=1e-9)
        dist = Normal(out['mu'], torch.sqrt(safe_sig))
        likelihood = dist.log_prob(y).mean() #--positive log likelihood--#

        #--KL_term--#
        kl_div = self.comp_kl_divergence_for_dirichlet_module()

        #--RKHS Loss---#
        rkhs_loss = F.mse_loss(out['h_recon'], h_vit_observed)

        #--Variational Objective (Negative ELBO)---#
        neg_elbo = -likelihood + (self.beta * kl_div) + (self.rkhs_w * rkhs_loss)

        loss_stats = {
            "total": neg_elbo,
            "nll": likelihood,
            "kl": kl_div,
            "rkhs": rkhs_loss
        }

        mlflow.log_metric("ELBO_loss")

        return loss_stats
    
    def _inject_langevin_noise(self, temp):
        noise_scale = math.sqrt(2 * self.optimiser.param_groups[0]['lr'] * temp)
        for g in self.optimiser.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    state = self.optimiser.state[p]
                    G = state.get('sum_square', torch.ones_like(p.grad))
                    precond = 1.0 / (torch.sqrt(G) + 1e-8)
                    langevin_noise = torch.randn_like(p.grad) * noise_scale * torch.sqrt(precond)
                    p.grad.add_(langevin_noise)
    
    def step(self, x, local_idx, y):
        """Stages B + C: Langevin step with adagrad preconditioning"""

        self.y = y
        self.local_idx = local_idx

        self.model.train()

        self.optimiser.zero_grad()

        with torch.no_grad():
            h_vit_observed = self.model.transformer.vit_model(x)
        
        out = self.model(x, local_idx=local_idx, y=y)

        loss_dict = self.comp_loss(out, y=y, h_vit_observed=h_vit_observed)

        loss_dict['total'].backward()

        self._inject_langevin_noise(self.temp)

        self.optimiser.step()

        return loss_dict
    
    def reinitialise(self):
        """Maps new sliced params (by DirichletPruner) to fresh Adagrad buffers"""

        logger.info("Reinitialising Adagrad for update in number of dirichlet cluster atoms")

        new_groups = [
            {'params': [self.model.dirichlet.mu_atom], 'lr': self.eta_a},
            {'params': [self.model.dirichlet.log_sigma_atom], 'lr': self.eta_a},
            {'params': [self.model.dirichlet.v_k], 'lr': self.eta_w},
            {'params': [self.model.dirichlet.v_j], 'lr': self.eta_w},
            {'params': self.model.decoder.parameters(), 'lr': self.lr_decoder}
        ]

        self.optimiser = optim.Adagrad(new_groups, lr=self.eta_a)

        logger.info(f"Optimiser re-initialised after clustering change. Current atoms in CRP 'menu': {self.model.dirichlet.mu_atom.shape[0]}")
    
    def evaluate(self, test_loader, epoch=0):
        self.model.eval()
        test_stats = {"total": 0, "nll": 0, "kl": 0, "rkhs": 0}
        n_batches = len(test_loader)

        t_preds = []
        t_targets = []
        t_vars = []

        logger.info(f"---Running test eval for epoch: {epoch}---")
        with torch.no_grad():
            for x, y, ind in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                h_vit_observed = self.model.transformer.vit_model(x)
                out = self.model(x, local_idx=ind, y=y)
                loss_dict_test_set = self.comp_loss(out, y=y, h_vit_observed=h_vit_observed)
                for k in test_stats:
                    test_stats[k] += loss_dict_test_set[k].item()
                
                t_preds.append(out['mu'])
                t_targets.append(y)
                t_vars.append(out['var'])
        
        mean_test_stats = {k : v / n_batches for k, v in test_stats.items()}

        preds = torch.cat(t_preds)
        targets = torch.cat(t_targets)
        vars = torch.cat(t_vars)

        mse = F.mse_loss(preds, targets).item()
        aggregate_uncertainty = vars.mean().item()

        self._log_val(epoch, mean_test_stats, mse, aggregate_uncertainty)

        logger.info(f"Val MSE: {mse:.4f} | Avg Uncertainty: {aggregate_uncertainty:.4f}")

        return mean_test_stats
    
    def _log_val(self, epoch, stats, mse, uncertainty):
        """Standardized MLflow logging for validation phase"""
        mlflow.log_metric("val_total_loss", stats["total"], step=epoch)
        mlflow.log_metric("val_nll", stats["nll"], step=epoch)
        mlflow.log_metric("val_rkhs_mse", stats["rkhs"], step=epoch)
        mlflow.log_metric("val_mse_accuracy", mse, step=epoch)
        mlflow.log_metric("val_avg_predictive_variance", uncertainty, step=epoch)

    def _log_epoch(self, epoch, stats):
        """Standardized MLflow logging for DKL/HDP metrics"""
        mlflow.log_metric("train_total_loss", stats["total"], step=epoch)
        mlflow.log_metric("train_nll", stats["nll"], step=epoch)
        mlflow.log_metric("train_kl_divergence", stats["kl"], step=epoch)
        mlflow.log_metric("train_rkhs_mse", stats["rkhs"], step=epoch)
        mlflow.log_metric("langevin_temp", self.temp, step=epoch)

    def fit(self, train_loader, test_loader=None):
        logger.info(f"Starting trainng for {self.epochs} epochs on {self.device}")
        for epoch in range(1, self.epochs+1):
            stats_per_epoch = {"total_loss": 0, "nll": 0, "kl": 0, "rkhs": 0}
            n_batches = len(train_loader)
            for batch_idx, (x, y, ind) in enumerate(train_loader): #---idx currently unused--#
                x, y = x.to(self.device), y.to(self.device)
                loss_dict = self.step(x, y, ind)
                for k in stats_per_epoch:
                    stats_per_epoch[k] += loss_dict[k].item()
            mean_train_stats = {k: v / n_batches for k, v in stats_per_epoch.items()}

            #--Mlflow logging--#
            self._log_epoch(epoch, mean_train_stats)
            active_atoms = self.model.dirichlet.mu_atom.shape[0]
            mlflow.log_metric("active_atoms", active_atoms, step=epoch)

            logger.info(f"Epoch {epoch}/{self.epochs} | Loss: {mean_train_stats['total']:.4f} | Atoms: {active_atoms}")
            if test_loader:
                self.evaluate(test_loader, epoch)
        
        mlflow.pytorch.log_model(self.model, "final_mixture_model")

