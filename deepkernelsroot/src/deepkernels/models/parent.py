import torch
import gpytorch
from gpytorch.mlls import AddedLossTerm
from gpytorch.priors import NormalPrior, GammaPrior, HorseshoePrior
import torch.distributions as dist
import torch.nn.functional as F
from deepkernels.losses.simple import SimpleLoss
import torch.distributions as dist

class BaseGenerativeModel(gpytorch.Module):
    def __init__(self):
        super().__init__()
    
    def register_constrained_parameter(self, name, parameter, constraint):
        self.register_parameter(name, parameter)
        self.register_constraint(name, constraint)
        return self
    
    def register_priors_for_dirichlet(self):
        if hasattr(self, "gamma"):
            self.register_prior("gamma_prior", GammaPrior(2.5, 1.0), lambda m: F.softplus(m.gamma), lambda m, v: None)
        
        if hasattr(self, "raw_logits"):
            self.register_prior("logit_prior", NormalPrior(loc=0.0, scale=1.0),lambda m: F.softplus(m.raw_logits))
    
    
    def register_kernel_priors(self):
        if hasattr(self, "covar_module"):
            self.register_prior("sparsity_prior", HorseshoePrior(scale=0.1), lambda m: m.raw_inv_bandwidth)

    def log_loss(self, name, value):
        """
        Wraps the raw tensor in an AddedLossTerm and updates it.
        usage: self.log_loss("reconstruction_loss", recon_tensor)
        """
        if not hasattr(self, "_added_loss_terms") or name not in self._added_loss_terms:
             raise RuntimeError(f"Loss term '{name}' not registered in Base __init__")
        scalar_loss = value.sum() if value.dim() > 0 else value
        self.update_added_loss_term(name, SimpleLoss(scalar_loss))

    
    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, vae_out, steps=None, batch_shape=torch.Size([]), **params):
        raise NotImplementedError("Subclass must implement forward()")
        
    def get_variational_strategy(self):
        """Optional: helper to retrieve the GP strategy easily"""
        raise NotImplementedError
    
    def multivariate_projection(self, mu, factor, diag):
        """for alpha params: input projections from three alpha heads"""
        mvn = dist.LowRankMultivariateNormal(
            loc=mu,
            cov_factor=factor,
            cov_diag=diag
        )
        jitter = 1e-6
        logits = mvn.rsample()
        alpha = torch.nn.functional.softplus(logits) + jitter
        
        return alpha
    
    def lowrankmultivariatenorm(self, mu, factor, diag):
        mvn = torch.distributions.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        logits = mvn.rsample()
        return logits # Return logits, softplus them later
    
    def apply_softplus(self, x, jitter=1e-6):
        return torch.nn.functional.softplus(x) + jitter
    
    def dynamic_random_fourier_features(self, z, omega, pi=None):
        B, D = z.shape
        if pi is None:
            pi = torch.full((B, self.K), 1.0/self.K, device=z.device)
            pi = F.softmax(pi, dim=-1)
        
        # Determine if omega is dynamic
        if omega.dim() == 4:
            proj = (z.view(B, 1, 1, D) * omega).sum(dim=-1) 
        else:
            W = omega.view(-1, D)
            proj = F.linear(z, W).view(B, self.K, self.M)
        
        proj = proj + self.noise_bias.unsqueeze(0)
        scale = 1.0 / math.sqrt(self.M)
        
        # Harmonics
        cos_proj = torch.cos(proj) * scale
        sin_proj = torch.sin(proj) * scale

        # Mixing
        pi_scl = torch.sqrt(pi).unsqueeze(-1)
        cos_proj = cos_proj * pi_scl
        sin_proj = sin_proj * pi_scl
        
        feats = torch.stack([cos_proj, sin_proj], dim=-1)
        return feats.flatten(1) 
        
    def get_omega(self, bw):
        # bw: [B, K, M, D]
        # Broadcasting: [1, 1, D] + [K, 1, D] + ([K, M, D] * [B, K, 1, D])
        # Note: self.noise_weights is [K, M, D]. 
        # To broadcast with [B, K, 1, D], we need noise to be [1, K, M, D]
        omega = self.h_mu + self.atom_mu + (self.noise_weights.unsqueeze(0) * bw)
        return torch.clamp(omega, -100.0, 100.0)
    
    def init_weights_nkn(self):
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.uniform_(self.linear.bias, 0.0, 2.0)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)
        nn.init.orthogonal_(self.complex_interactions.weight, gain=1.41)

        for module in self.latent_kernel_heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.41)
    
    def get_latent_kernels(self, kernel_features):
        return [latent(kernel_features) for latent in self.latent_kernel_heads]
    
    def feed_dirichlet_gate(self, kernel_features):
        return self.spectral_feedback_loop(kernel_features)
    
    def get_cov_matrices(self, latent_kernels):
        stacked_features = self.stack_features(latent_kernels)
        cov_matrices = torch.einsum('lbd, lcd -> lbc', stacked_features, stacked_features)
        return cov_matrices #-[8, B, B]
    
    def stack_features(self, latent_kernels):
        return torch.stack(latent_kernels)
    
    def compute_kernel_interactions(self, lin, per, rbf, rat):
        "input stack: [B, 4, 32]"
        stack = torch.stack([lin, per, rbf, rat], dim=1)
        mask = torch.sigmoid(self.selection_weights).unsqueeze(-1)
        stack_safe = torch.abs(stack) + 1e-6
        log_stack = torch.log(stack_safe)
        # Weighted Einsum in log space: b p d (batch, primitives, dim), k p   (products, primitives) -> b k d (batch, products, dim)
        log_product = torch.einsum('bpd, kp -> bkd', log_stack, mask.squeeze(-1))
        product_features = torch.exp(log_product) #-[B, 12, 32]
        interactions_matrix = product_features.flatten(start_dim=1) #-[B, 384]
        return self.complex_interactions(interactions_matrix) #-[B, products, 128]
    
    def compute_primitives(self, x):
        lin = self.linear(x) * self.linear_scale
        per = torch.cos(self.periodic(x))
        rbf = torch.exp(-torch.pow(self.rbf(x), 2))
        rat = 1.0 / (1.0 + torch.pow(self.rational(x), 2))
        return lin, per, rbf, rat