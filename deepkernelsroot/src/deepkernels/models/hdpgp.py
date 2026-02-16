import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import IndependentMultitaskVariationalStrategy, VariationalStrategy
import torch.nn as nn

class BaseVariationalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A dictionary to hold current step's losses
        self._loss_registry = {} 

    def update_added_loss_term(self, name, value):
        # Store the loss tensor (keep gradients attached!)
        self._loss_registry[name] = value

    def get_additional_losses(self):
        # Return the sum of all registered losses
        return sum(self._loss_registry.values())
    
    def clear_losses(self):
        # Clear for the next batch
        self._loss_registry = {}