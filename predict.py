#filename: inference.py

#---Dependencies---#
import torch
import gpytorch
import logging
import os
from typing import Tuple, Dict, Optional

#---Init Logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

#---Class Definition: Inference Module for DKL Models---#
class DeepKernelInference:
    """
    Makes predictions from a trained Deep Kernel Learning model
    Features:
    - Loads model & likelihood state from training checkpoints
    - Specific functional call for 'predict_uncertainty' to assess GP confidence
    - Outputs class probabilities for classification
    - Optimised for inference on CPU or GPU devices
    """

    def __init__(
            self,
            mod_arch: gpytorch.models.ApproximateGP,
            lik_arch: gpytorch.likelihoods.Likelihood,
            checkpoint_path: str,
            device: str = "cpu"
    ):
        """Args:"""
        self.device = torch.device(device)
        self.model = mod_arch.to(self.device)
        self.likelihood = lik_arch.to(self.device)
        self.checkpoint_path = checkpoint_path
        self._load_weights()
        self.model.eval()
        self.likelihood.eval()
    
    def _load_weights(self) -> None:
        """Loads state dicts from checkpoints saved to disk."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        logger.info(f"Loading model weights from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully.")
        except RuntimeError as e:
            logger.error(f"Failed to load model weights, possible architecture mismatch? Error: {e}")
            raise
        try:
            self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            logger.info("Likelihood weights loaded successfully.")
        except RuntimeError as e:
            logger.error(f"Failed to load likelihood weights, possible architecture mismatch? Error: {e}")
            raise
    
    def pred(self, X_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard prediction function
        returns: (Predicted class labels, Probability of positive class)
        """
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_dist = self.model(X_tensor)
            pred_dist = self.likelihood(latent_dist)
            probs = pred_dist.mean
            pred_labels = (probs > 0.5).float()
        return pred_labels.cpu(), probs.cpu()
    
    def pred_uncertainty_adjusted(self, X_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Uncertainty-aware prediction function for calibration & active learning
        returns: (Dict with keys: 'predicted_probabilities', 'epistemic_uncertainty', 'labels')
        """
        X_tensor = X_tensor.to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_dist = self.model(X_tensor)
            pred_dist = self.likelihood(latent_dist)
            epistemic_uncertainty = latent_dist.variance
            probs = pred_dist.mean
            pred_labels = (probs > 0.5).float()
        return {
            "predicted_probabilities": probs.cpu(),
            "epistemic_uncertainty": epistemic_uncertainty.cpu(),
            "labels": pred_labels.cpu()
        }