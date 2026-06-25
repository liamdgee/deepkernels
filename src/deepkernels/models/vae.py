import logging
from typing import NamedTuple, Optional
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import StateSpaceOutput

class SpectralVAE(nn.Module):
    """Minimal viable VAE structure to allow the script to run standalone."""
    def __init__(self, input_dim=30):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, input_dim)
        self.last_state = None

    def get_zero_state(self, x, device, batch_size):
        # Initialize zero states for the 8 expected features
        return StateSpaceOutput(*[torch.zeros(batch_size, self.input_dim, device=device) for _ in range(8)])

    def forward(
        self,
        x,
        state: Optional[StateSpaceOutput]=None,
        indices=None,
        batch_shape=torch.Size([]),
        features_only: bool = False,
        generative_mode: bool = False,
        **params,
    ) -> StateSpaceOutput:
        
        out = self.net(x)
        jitter_scale = 0.05

        if state is None:
            last = getattr(self, "last_state", None)
            
            if last is not None:
                state = StateSpaceOutput(
                    recon=out,
                    gates=last.gates + jitter_scale * torch.randn_like(last.gates),
                    linear=last.linear + jitter_scale * torch.randn_like(last.linear),
                    periodic=last.periodic + jitter_scale * torch.randn_like(last.periodic),
                    rational=last.rational + jitter_scale * torch.randn_like(last.rational),
                    polynomial=last.polynomial + jitter_scale * torch.randn_like(last.polynomial),
                    matern=last.matern + jitter_scale * torch.randn_like(last.matern),
                    pi=last.pi + jitter_scale * torch.randn_like(last.pi)
                )
            else:
                state = StateSpaceOutput(
                    recon=out,
                    gates=torch.randn_like(out),
                    linear=torch.randn_like(out),
                    periodic=torch.randn_like(out),
                    rational=torch.randn_like(out),
                    polynomial=torch.randn_like(out),
                    matern=torch.randn_like(out),
                    pi=torch.randn_like(out)
                )
        
        self.last_state = state
        return state
