import torch
import gpytorch
from gpytorch.mlls import AddedLossTerm
from gpytorch.priors import NormalPrior, GammaPrior

class BaseGenerativeModel(gpytorch.Module):
    def __init__(self):
        super().__init__()
    
    def register_constrained_parameter(self, name, parameter, constraint):
        self.register_parameter(name, parameter)
        self.register_constraint(name, constraint)
        return self
    
    def register_priors_for_dirichlet(self):
        if hasattr(self, "h_log_sigma"):
            self.register_prior("global_lengthscale_prior", GammaPrior(concentration=2.5, rate=3.0), lambda m: m.h_log_sigma.exp(), lambda m, v: None)

        if hasattr(self, "atom_log_sigma"):
            self.register_prior("local_lengthscale_prior", GammaPrior(concentration=3.0, rate=5.0), lambda m: m.atom_log_sigma.exp(), lambda m,v: None)
        
        if hasattr(self, "gamma"):
            self.register_prior("gamma_prior", GammaPrior(2.5, 1.0), lambda m: F.softplus(m.gamma), lambda m, v: None)

    def log_loss(self, name, value):
        """
        Wraps the raw tensor in an AddedLossTerm and updates it.
        usage: self.log_loss("reconstruction_loss", recon_tensor)
        """
        if not hasattr(self, "_added_loss_terms") or name not in self._added_loss_terms:
             raise RuntimeError(f"Loss term '{name}' not registered in Base __init__")
        
        # Determine if we need to sum (if it's a batch) or take mean
        # You can customize this logic centrally here!
        scalar_loss = value.sum() if value.dim() > 0 else value
        self.update_added_loss_term(name, SimpleLoss(scalar_loss))

    # C. SHARED MATH
    # Common operations like reparameterization should live here, not in the child
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def update_added_loss_term(self, name, value):
        self.added_loss_terms[name] = value

    def collect_internal_losses(self):
        # Sums up everything registered during the forward pass
        if not self.added_loss_terms:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return sum(self.added_loss_terms.values())

    def clear_loss_registry(self):
        self.added_loss_terms = {}
    
    def forward(self, x):
        raise NotImplementedError("Subclass must implement forward()")
        
    def get_variational_strategy(self):
        """Optional: helper to retrieve the GP strategy easily"""
        raise NotImplementedError
