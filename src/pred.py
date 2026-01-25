#filename: inference.py

#---Dependencies---#
import torch
import logging
import os
import numpy as np
from typing import Tuple, Dict, Optional
from src import integrate

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---Class Definition: Inference Module for DKL Models---#
class Inference:
    """
    Makes predictions from a trained Deep Kernel Learning model
    Features:
    - Loads model & likelihood state from training checkpoints
    - numerical integrator via simpsons rule for auditing
    - manifold projection and posterior density recon
    - optimised for mps backend (M series mac)
    """

    def __init__(
            self,
            mod_arch: torch.nn.Module,
            checkpoint_path: str,
            n_steps: int = 3001,
            device: str = "cpu",
            eps: float = 1e-8
    ):
        
        self.device = torch.device(device)
        self.model = mod_arch.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.engine = integrate.IntegrationEngine(n_steps=n_steps, bounds=(-10,10))
        self.eps = eps
        self._load_weights()
        self.model.eval()
    
    def _load_weights(self) -> None:
        """Loads state dicts from checkpoints saved to disk."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        logger.info(f"Loading model weights from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model weights loaded successfully.")
    
    def audit(self, X_tensor: torch.Tensor, lender_idx: int, mu_prior: float = 0.0, sig_prior: float = 1.0) -> Dict:
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor, lender_idx)
            mu_p = out['mu'].mean()
            var_p = out['var'].mean()
        
        p_pdf = self.engine.get_post_density(mu_p, var_p)
        q_pdf = self.engine.get_post_density(mu_prior, sig_prior)

        kl_div = self.engine.compute_elementwise_kl(p_pdf, q_pdf)
        
        audit_dict = {
            "mu": mu_p.item(),
            "var": var_p.item(),
            "kl_divergence": kl_div,
            "p_pdf": p_pdf,
            "q_pdf": q_pdf,
            "x_axis": self.engine.x_ax
        }

        return audit_dict

    def predict(self, X_tensor: torch.Tensor, lender_idx: int) -> Dict:
        """Inference module for MLOPS pipeline"""
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor, lender_idx)

            #--- 2 Confidence Intervals for calc--#
            sig = torch.sqrt(torch.clamp(out['var'], min=self.eps))
            lower_bound = out['mu'] - 1.96 * sig
            upper_bound = out['mu'] + 1.96 * sig
        
        results = {
            "projected_mean": out['mu'].cpu(),
            "predicted_variance": sig.cpu(),
            "95_pct_confidence": (lower_bound.cpu(), upper_bound.cpu()),
            "model_weights": out['weights'].cpu()
        }

        return results