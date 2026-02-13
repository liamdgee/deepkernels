import torch
import torch.nn as nn
import math
import torch.nn.utils.parametrizations as P
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import torch
import math
import gpytorch
from gpytorch.kernels import Kernel
from linear_operator.operators import LowRankRootLinearOperator, RootLinearOperator, MatmulLinearOperator
from src.deepkernels.models.encoder import RecurrentEncoder

class DynamicMixtureMean(gpytorch.means.Mean):
    def __init__(self, covar_access, base_means=None, jitter=1e-6):
        """
        Args:
            base_means (list): A list of K gpytorch.means.Mean modules.
            hypernet (nn.Module): The same hypernetwork used in your kernel.
        """
        super().__init__()
        self.base_means = nn.ModuleList(base_means)
        self.covar_access = covar_access
        self.hypernet = DeepKernel(mean_access=self.covar_access)
        self.jitter = jitter

    def forward(self, x):
        # 1. Get Mixture Weights from Hypernetwork (reuse kernel logic)
        # Note: We only need w1 (weights) here, not lengthscales
        w, _ = self.hypernet(x)  # w shape: [..., N, K]
        
        # Normalize to probability space (pi)
        S = w.sum(dim=-1, keepdim=True)
        pi = torch.clamp(w / (S + self.jitter), min=self.jitter) # [..., N, K]

        # 2. Compute the K individual means
        # Stack them to get shape [..., N, K]
        # We iterate through the list of means you provided
        mean_outputs = [mean_module(x).unsqueeze(-1) for mean_module in self.base_means]
        mean_stack = torch.cat(mean_outputs, dim=-1) # [..., N, K]
        
        # 3. Weighted Sum (Gating)
        # Sum_k ( pi_k(x) * mu_k(x) )
        final_mean = (pi * mean_stack).sum(dim=-1) # [..., N]
        
        return final_mean


class DeepKernel(Kernel):
    """
    A unified Deep Kernel module.
    1. Neural Phase: Transforms latent 'z' into high-dimensional task features.
    2   . Kernel Phase: Maps task features into an approximate RBF kernel using Orthogonal Random Features.
    """
    def __init__(self, mean_access, **kwargs):
        super().__init__(**kwargs)
        
        # --- Config ---
        self.mean_access = mean_access
        self.input_dim = 7680
        self.H = 16
        self.n_experts = 30
        self.output_dim_per_cluster = 256
        self.deep_feat_dim = 30 * 256
        self.n_samples = 512

        # ==========================================
        # Neural Network Architecture
        # ==========================================
        self.linear = sn(nn.Linear(self.input_dim, self.H))
        self.linear_scale = nn.Parameter(torch.tensor(0.13))
        self.periodic = sn(nn.Linear(self.input_dim, self.H))
        self.rbf = sn(nn.Linear(self.input_dim, self.H))
        self.rational = sn(nn.Linear(self.input_dim, self.H))
        self.constant = nn.Parameter(torch.randn(1, self.H) * 0.13)
        self.primitive_norm = nn.LayerNorm(self.H * 9)

        self.mixer = nn.Sequential(
            sn(nn.Linear(self.H * 9, self.H * 12)),
            nn.SiLU(),
            nn.LayerNorm(self.H * 12),
        )

        #-Parallel Task Heads-#
        self.parallel_task_layers = nn.ModuleList()
        for _ in range(self.n_experts):
            expert_layer = sn(nn.Linear(self.H * 12, self.output_dim_per_cluster))
            self.mean_access.parallel_task_layers.append(expert_layer)

        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        self.log_amplitude = nn.Parameter(torch.zeros(self.n_experts))
        self._init_weights()

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def _init_weights(self):
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.orthogonal_(self.periodic.weight, gain=1.41)
        nn.init.orthogonal_(self.rbf.weight, gain=2.0)
        nn.init.uniform_(self.rbf.bias, -1.0, 1.0)
        nn.init.orthogonal_(self.rational.weight, gain=1.41)

        for module in self.mixer.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('silu'))

        for layer in self.parallel_task_layers:
            nn.init.orthogonal_(layer.weight, gain=1.0)
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    
    def _compute_primitives_and_interactions(self, x):
        """
        Runs the Neural Network: x -> Primitives -> Mixer -> Task Heads -> Flattened Features
        """
        # A. Primitives
        lin_out = self.linear(x) * self.linear_scale
        period_out = torch.cos(self.periodic(x))
        rbf_out = torch.exp(-torch.pow(self.rbf(x), 2))
        rat_out = 1.0 / (1.0 + torch.pow(self.rational(x), 2))
        const_out = self.constant.expand(x.size(0), -1)
        
        # B. Interactions
        seasonal = lin_out * period_out
        local = lin_out * rbf_out
        decay = period_out * rbf_out
        tails = period_out * rat_out
        
        combined = torch.cat([lin_out, period_out, rbf_out, rat_out, const_out, seasonal, local, decay, tails], dim=-1)
        mixed = self.mixer(self.primitive_norm(combined))

        #-heads-#
        task_outputs = []
        for layer in self.parallel_task_layers:
            task_outputs.append(layer(mixed))
        
        #-Stack [Batch, K, D_per_k]
        stacked = torch.stack(task_outputs, dim=1)
        
        return stacked
    
    def forward(self, x1, x2, diag=False, **params):

        features_x1 = self._compute_primitives_and_interactions(x1)
        
        # 2. Permute to [K, Batch, D] 
        features_x1 = features_x1.permute(1, 0, 2) # [30, B, 256]
        
        if x2 is None or torch.equal(x1, x2):
            features_x2 = features_x1
            symmetric = True
        else:
            features_x2 = self._compute_primitives_and_interactions(x2)
            features_x2 = features_x2.permute(1, 0, 2)
            symmetric = False

        # We use the log_amplitude vector (size 30)
        amplitude = self.log_amplitude.exp().view(-1, 1, 1) # [30, 1, 1]
        
        # Scale by amplitude and normalize by sqrt(D) for RFF stability
        scale_factor = amplitude / math.sqrt(self.output_dim_per_cluster)
        
        features_x1 = features_x1 * scale_factor
        if not symmetric:
            features_x2 = features_x2 * scale_factor
        else:
            features_x2 = features_x1
        #input is 3D [K, B, D], RootLinearOperator automatically 
        if diag:
            # Returns [K, Batch]
            return (features_x1 * features_x2).sum(-1)
            
        if symmetric:
            # Returns Batched LinearOperator representing [K, Batch, Batch]
            return RootLinearOperator(features_x1)
        else:
             # Returns Batched LinearOperator representing [K, Batch_1, Batch_2]
            return MatmulLinearOperator(features_x1, features_x2.transpose(-1, -2))