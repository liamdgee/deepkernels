#filename: multi_stage_dkl_trainer.py

#filename: Staged_trainer.py

#filename: train.py

#---Dependencies---#
import numpy as np
import torch
from torch.utils.data import DataLoader
import gpytorch
import logging
import os
import json
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score
)

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---Device Detection Function---#
def assign_device() -> str:
    """
    Detects the best availanble device for training & inference;
    Priorities: if NVIDIA GPU -> CUDA, elif M-chip Apple -> MPS, else CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

#---Class Definition: Configuration for Training Module---#
@dataclass
class ConfigTrainer:
    nn_only_epochs: int = 20
    gp_only_epochs: int = 80
    epochs: int = 100
    lr: float = 3e-3
    w_decay: float = 1e-6
    checkpoint_dir: str = "./checkpoints"
    device: str = field(default_factory=assign_device)
    patience: int = 8
    batch_size: int = 256
    save_best_only: bool = True

#---Class Definition: Deep Kernel Learning Trainer Class---#
class DKLModelTrainer:
    """
    OOP framework for training Deep Kernel Learning models.
    Assumes architecture: Deep nn feature extractor -> sparse variational GP.
    Outputs logits for classification tasks.
    Explicitly handles: 
    - Variational ELBO loss function as a proxy measure for KL divergence
    - Optimisation of neural network and Gaussian Process hyperparameters
    - Checkpointing & early termination of training where no progress is made

    Flow:
    - Stage A: train nn, freeze gp
    - Stage B: freeze nn, train gp
    - Stage C: Train nn + gp
    """

    def __init__(
            self, 
            model: gpytorch.models.ApproximateGP,
            likelihood: gpytorch.likelihoods.Likelihood,
            train_loader: DataLoader,
            test_loader: DataLoader,
            config: ConfigTrainer
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
        self.optim = None
        self.mll = self._config_loss_fn()
    
    def _config_loss_fn(self) -> gpytorch.mlls.VariationalELBO:
        """
        Sets up the evidence lower bound (variational ELBO);
        approximates marginal log likelihood (param: 'mll')
        """
        n_data_points = len(self.train_loader.dataset)
        return gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=n_data_points)
    
    def _separate_params(self):
        """
        Seperates neural net and gp params/hyperparams
        Assumes standardised GpyTorch setup with 'variational_strategy',
        'mean_module', 'covar_module' or 'likelihood'
        """

        gp_core_params = ['variational_strategy', 'mean_module', 'covar_module', 'likelihood']
        nn_params = []
        gp_params = []

        #---Model Params---#
        for name, param in self.model.named_parameters():
            if any(gp_name in name for gp_name in gp_core_params):
                gp_params.append(param)
            else:
                nn_params.append(param)

        #---GP Likelihood Params---#
        for param in self.likelihood.parameters():
            gp_params.append(param)
        
        return nn_params, gp_params
    
    def _define_stage(self, stage: str) -> None:
        """
        Configures optimiser and gradient freezing for relevant stage
        """
        nn_params, gp_params = self._separate_params()
        if stage == 'A':
            logger.info("Stage A: Freeze GP weights to warmup nn feature extractor.")
            for p in nn_params: p.requires_grad = True
            for p in gp_params: p.requires_grad = False
            trainable_params = nn_params
            current_stage = 'A'
        
        elif stage == 'B':
            logger.info("Stage B: Freezing neural net weights, training GP.")
            for p in nn_params: p.requires_grad = False
            for p in gp_params: p.requires_grad = True
            trainable_params = gp_params
            current_stage = 'B'
        
        elif stage == 'C':
            logger.info("Stage C: Final Setup with all params trainable.")
            for p in nn_params: p.requires_grad = True
            for p in gp_params: p.requires_grad = True
            trainable_params = nn_params + gp_params
            current_stage = 'C'
        
        else:
            raise ValueError(f"Not assigned a valid stage defined in arch: {stage}")
        
        self.optim = torch.optim.Adam(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.w_decay
        )
    

    def run_epoch(self, current_stage: str) -> float:
        """Runs a single training instance"""
        self.model.train()
        self.likelihood.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Training ({current_stage})", leave=False)
        for X_batch, y_batch in progress:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optim.zero_grad()
            dist = self.model(X_batch)
            loss = -self.mll(dist, y_batch)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        mean_loss = total_loss / len(self.train_loader)
        return mean_loss
    
    def validate(self) -> Dict[str, float]:
        """
        Evauates model on valuation (test) set
        returns: Average loss, accuracy
        """
        self.model.eval()
        self.likelihood.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                #---Forward Pass---#
                dist_out = self.model(X_batch)
                #---Loss Calculation---#
                loss = -self.mll(dist_out, y_batch)
                total_loss += loss.item()
                #---Probabilities & Predictions---#
                pred_dist = self.likelihood(dist_out)
                probs = pred_dist.mean
                preds = (probs > 0.5).float()
                #---Store Results---#
                all_targets.append(y_batch.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        #---Concatenate results---#
        y_pos = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        #---Metrics Calculation---#
        metrics = {}
        #---Loss---#
        metrics['test_loss'] = total_loss / len(self.test_loader)
        #---Sklearn Metrics---#
        try:
            metrics['acc'] = accuracy_score(y_pos, y_pred)
            metrics['f1'] = f1_score(y_pos, y_pred, zero_division=0)
            metrics['brier'] = brier_score_loss(y_pos, y_prob)
            metrics['recall'] = recall_score(y_pos, y_pred, zero_division=0)
            metrics['precision'] = precision_score(y_pos, y_pred, zero_division=0)
            if len(np.unique(y_pos)) > 1:
                metrics['auc'] = roc_auc_score(y_pos, y_prob)
            else:
                metrics['auc'] = float('nan')
        except Exception as e:
            logger.warning(f"Error computing metrics: {e} --dataset likely too small or single-class.")
            metrics['acc'] = float('nan')
            metrics['f1'] = float('nan')
            metrics['brier'] = float('nan')
            metrics['auc'] = float('nan')
            metrics['recall'] = float('nan')
            metrics['precision'] = float('nan')
        return metrics
    
    def _init_stage_loop(self, epochs: int, current_stage: str):
        """runs epochs in loop for specified stage (A/B/C)"""
        if epochs <= 0: return
        logger.info(f"=== Starting {current_stage} | epochs: {epochs} ===")
        self._define_stage(current_stage)

        for e in range(1, epochs + 1):
            train_loss = self.run_epoch(current_stage)
            metric_values = self.validate()
            logger.info(f"[{current_stage}] Epoch {e}/{epochs} | Train Loss: {train_loss:.4f} | Accuracy: {metric_values['acc']:.4f} | Test Loss: {metric_values['test_loss']:.4f} | AUC: {metric_values['auc']:.4f} | F1: {metric_values['f1']:.4f} | Brier: {metric_values['brier']:.4f} | Recall: {metric_values['recall']:.4f} | Precision: {metric_values['precision']:.4f}")

    def fit(self) -> None:
        """
        Main training loop with checkpointing & early stopping
        Orchestrates all 3 stages of training
        """
        logger.info(f"Starting multi-stage training on {self.device} for {self.config.epochs} epochs.")
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        #---Stage A: Train Feature extractor---#
        self._init_stage_loop(self.config.nn_only_epochs, "A")
        #---Stage B: Train GP---#
        self._init_stage_loop(self.config.gp_only_epochs, "B")
        #---Stage C: Model Traning Unconstrained---#
        logger.info("=== Starting Stage C: Full Model Training ===")
        self._define_stage("C")
        best_value_by_metric = float('inf')
        patience_count = 0

        #---Final Stage Core Epoch Loop---#
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.run_epoch("C")
            metric_values = self.validate()
            test_loss = metric_values['test_loss']
            accuracy = metric_values['acc']
            area_under_curve = metric_values['auc']
            test_accuracy = accuracy if not np.isnan(accuracy) else 0.0
            f1_score = metric_values['f1'] if not np.isnan(metric_values['f1']) else 0.0
            brier = metric_values['brier'] if not np.isnan(metric_values['brier']) else 0.0
            recall = metric_values['recall'] if not np.isnan(metric_values['recall']) else 0.0
            precision = metric_values['precision'] if not np.isnan(metric_values['precision']) else 0.0

            #---History Logging---#
            self.history["train_loss"].append(train_loss)
            for i, j in metric_values.items():
                if i not in self.history: self.history[i] = []
                self.history[i].append(j)
            
            logger.info(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f} | AUC: {area_under_curve:.4f} | F1: {f1_score:.4f} | Brier: {brier:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")

            #---Early Stopping & Checkpointing---#
            target_metric = test_loss

            if target_metric < best_value_by_metric:
                best_value_by_metric = target_metric
                patience_count = 0
                self._save_checkpoint(epoch, test_loss, is_best=True)
                logger.info(f"New best model found at epoch {epoch} with test loss {test_loss:.4f}")
            else:
                patience_count += 1
                if not self.config.save_best_only:
                    self._save_checkpoint(epoch, test_loss, is_best=False)
                if patience_count >= self.config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        self._save_history()
        logger.info(f"Training Complete. Checkpoints and history saved to {self.config.checkpoint_dir}")
    
    def _save_history(self) -> None:
        """Saves training history to JSON file"""
        history_path = os.path.join(self.config.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        logger.info(f"Training history saved to {history_path}")
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """Saves model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': loss,
            'history': self.history
        }

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            logger.info(f"Best model checkpoint saved to {best_path}")
        if not self.config.save_best_only:
            filename = f'model_ep_{epoch}.pth'
            filepath = os.path.join(self.config.checkpoint_dir, filename)
            torch.save(state, filepath)
            if not is_best:
                logger.info(f"Checkpoint saved to {filepath}")