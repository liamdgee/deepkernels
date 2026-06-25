from typing import NamedTuple
import torch

class StateSpaceOutput(NamedTuple):
    recon: torch.Tensor
    gates: torch.Tensor
    linear: torch.Tensor
    periodic: torch.Tensor
    rational: torch.Tensor
    polynomial: torch.Tensor
    matern: torch.Tensor
    pi: torch.Tensor