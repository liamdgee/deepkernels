import torch
import gpytorch
import torch.nn as nn
from gpytorch.mlls import AddedLossTerm

class SimpleLoss(AddedLossTerm):
        def __init__(self, loss):
            self.loss = loss
        def add_loss_term(self, current_loss):
            return current_loss + self.loss