#filename: train.py

#---Dependencies---#
import torch
from torch.utils.data import DataLoader
import gpytorch
import logging
import os
import json
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field

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
    epochs: int = 80
    lr: float = 3e-3
    w_decay: float = 1e-6
    checkpoint_dir: str = "./checkpoints"
    device: str = field(default_factory=assign_device)
    patience: int = 8,
    batch_size: int = 256,
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
    """

    def __init__(
            self, 
            model: gpytorch.models.ApproximateGP,
            likelihood: gpytorch.likelihoods.BernoulliLikelihood,
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
        self.optim = self._config_optim()
        self.mll = self._config_loss_fn()
    
    def _config_optim(self) -> torch.optim.Adam:
        """Configs optimiser for nn weights & GP params"""
        return torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.config.lr, weight_decay=self.config.w_decay)
    
    def _config_loss_fn(self) -> gpytorch.mlls.VariationalELBO:
        """
        Sets up the evidence lower bound (variational ELBO);
        approximates marginal log likelihood (param: 'mll')
        """
        n_data_points = len(self.train_loader.dataset)
        return gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=n_data_points)
    
    def run_epoch(self) -> float:
        """Runs a single training instance"""
        self.model.train()
        self.likelihood.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc="Training", leave=False)
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
    
    def validate(self) -> Tuple[float, float]:
        """
        Evauates model on valuation (test) set
        returns: Average loss, accuracy
        """
        self.model.eval()
        self.likelihood.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                dist_out = self.model(X_batch)
                loss = -self.mll(dist_out, y_batch)
                total_loss += loss.item()
                pred_dist = self.likelihood(dist_out)
                pred_labels = (pred_dist.mean > 0.5).float()
                correct += (pred_labels == y_batch).sum().item()
                total += y_batch.size(0)
        mean_test_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        return mean_test_loss, accuracy

    def fit(self) -> None:
        """Main training loop with checkpointing & early stopping"""
        logger.info(f"Starting training on {self.device} for {self.config.epochs} epochs.")
        best_test_loss = float('inf')
        patience_count = 0
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.run_epoch()
            test_loss, test_accuracy = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_accuracy)

            logger.info(f"Epoch {epoch}/{self.config.epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Test Loss: {test_loss:.4f} | "
                        f"Test Accuracy: {test_accuracy:.4f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
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