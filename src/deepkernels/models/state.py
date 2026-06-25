from typing import NamedTuple
import torch

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union, NamedTuple

import gpytorch

class StateSpaceOutput(NamedTuple):
    recon: torch.Tensor
    gates: torch.Tensor
    linear: torch.Tensor
    periodic: torch.Tensor
    rational: torch.Tensor
    polynomial: torch.Tensor
    matern: torch.Tensor
    pi: torch.Tensor

@dataclass
class ModelOutput:
    """
    Unified return shape for any StateSpaceKernelProcess.forward().

    All fields default to None so subclasses (or features-only paths) fill in
    only what they compute. Unlike NamedTuple, fields are mutable post-
    construction — useful for progressive enrichment patterns where the
    forward pass populates fields in stages.

    Conventions:
      • state    : the VAE/Dirichlet/Decoder StateSpaceOutput passed forward
      • gp_out   : MVN for single head, or Dict[str, MVN] for multi-head
                   (e.g. {'forward': MVN, 'score': MVN} when score head enabled)
      • zz       : the packed [B, L, kernel_input_dim] feature tensor fed to GP
    """
    state:  Optional["StateSpaceOutput"] = None
    gp_out: Optional[Union[
        gpytorch.distributions.MultivariateNormal,
        Dict[str, gpytorch.distributions.MultivariateNormal],
    ]] = None
    zz:     Optional[torch.Tensor] = None

    # ---- convenience constructors --------------------------------------
    @classmethod
    def empty(cls) -> "ModelOutput":
        """Fallback empty return — useful when subclass aborts early."""
        return cls()

    @classmethod
    def features_only(cls, state: "StateSpaceOutput") -> "ModelOutput":
        """Returned when forward(..., features_only=True). Carries state only."""
        return cls(state=state)

    # ---- ergonomic accessors -------------------------------------------
    def as_tuple(self) -> tuple:
        """Back-compat with the (state, gp_out, zz, belief) tuple destructure."""
        return (self.state, self.gp_out, self.zz)

    def as_dict(self) -> Dict[str, Any]:
        """For logging or serialization."""
        return asdict(self)

    def __iter__(self):
        """Allow `state, gp_out, zz, belief = model_out` to keep working."""
        return iter(self.as_tuple())