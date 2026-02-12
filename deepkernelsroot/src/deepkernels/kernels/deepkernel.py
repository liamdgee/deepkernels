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


class DeepKernel(Kernel):
    """
    A unified Deep Kernel module.
    1. Neural Phase: Transforms latent 'z' into high-dimensional task features.
    2   . Kernel Phase: Maps task features into an approximate RBF kernel using Orthogonal Random Features.
    """
    def __init__(self, input_dim=7680, n_experts=30, features_per_expert=256, 
                 hidden_dim=16, orf_num_samples=512, **kwargs):
        super().__init__(**kwargs)
        
        # --- Config ---
        self.input_dim = input_dim
        self.H = hidden_dim
        self.n_experts = n_experts
        self.output_dim_per_cluster = features_per_expert
        self.deep_feat_dim = self.n_experts * self.output_dim_per_cluster
        self.n_samples = orf_num_samples

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
            self.parallel_task_layers.append(expert_layer)

        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        
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
        
        #-Flatten [Batch, K * D_per_k] --> feature vector
        flat_features = stacked.view(x.size(0), -1)
        
        return flat_features
    
    def forward(self, x1, x2, diag=False, **params):
        features_x1 = self._compute_primitives_and_interactions(x1)
        if x2 is None:
            features_x2 = features_x1
            symmetric = True
        else:
            if torch.equal(x1, x2):
                features_x2 = features_x1
                symmetric = True
            else:
                features_x2 = self._compute_primitives_and_interactions(x2)
                symmetric = False
        if self.outputscale is not None:
            scale = self.outputscale.sqrt()
            features_x1 = features_x1 * scale
            if symmetric:
                features_x2 = features_x1
            else:
                features_x2 = features_x2 * scale
        if diag:
            return (features_x1 * features_x2).sum(-1)
        if symmetric:
            #--Returns [Batch, N, N] as V @ V.T--#
            return RootLinearOperator(features_x1)
        else:
            #--Returns [Batch, N, M] as A @ B.T --#
            return MatmulLinearOperator(features_x1, features_x2.transpose(-1, -2))