import torch
import math
import itertools
import gpytorch
from gpytorch.kernels import Kernel
from pykeops.torch import LazyTensor
from linear_operator.operators import KeOpsLinearOperator


class GenerativeKeOpsHybridKernel(Kernel):
    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_primitives = 4     #--> 4 Primitives + 6 Pairs + 4 Self-Squared + 2 Triples = 16 Kernels-#
        self.kernels_out = 16 

    def forward(self, x1, x2, diag=False, **params):
        """
        x1: (Batch, N, D)
        x2: (Batch, M, D)
        params: Must contain 'ls_rbf', 'ls_per', 'p_per', 'ls_mat', 
                'w_sm', 'mu_sm', 'v_sm', and 'gates'.
                All params should be pre-shaped by the VAE to (Batch, 1, 1, ...)
        """
        if diag:
            return self._forward_diag_fallback(x1, x2, **params)
        
        hyperparams = params.get("gp_params", None)
        if hyperparams is None:
            raise ValueError("Missing Kernel hyperparameters")
        
        ls_rbf = hyperparams['ls_rbf']
        ls_per = hyperparams['ls_per']
        p_per = hyperparams['p_per']
        ls_mat = hyperparams['ls_mat']
        w_sm = hyperparams['w_sm']
        mu_sm = hyperparams['mu_sm']
        v_sm = hyperparams['v_sm']
        gates = hyperparams['gates']

        # --DISTANCE MATRICES ---# x_i: (B, N, 1, D)---# x_j: (B, 1, M, D)--#
        x_i = LazyTensor(x1.unsqueeze(-2))
        x_j = LazyTensor(x2.unsqueeze(-3))

        # d2: (B, N, M) (Squared Distance) -- # - d:  (B, N, M) - (Absolute Distance)-#
        x_diff = x_i - x_j
        d2 = (x_diff ** 2).sum(-1) 
        d = d2.sqrt()

        #-PRIMITIVES-#
        prims = []

        #-RBF Primitive: exp(-0.5 * d^2 / ls^2) wrapped as (B, 1, 1, 1)
        ls_rbf = LazyTensor(ls_rbf.view(-1, 1, 1, 1))
        k_rbf = (-0.5 * d2 / (ls_rbf ** 2)).exp()
        prims.append(k_rbf)

        #-Spectral Mixture Primitive-- isotropic SM formulation over distances wrapped as (B, 1, 1, Q)
        # sum_q [ w_q * cos(2pi * d * mu_q) * exp(-2pi^2 * d^2 * v_q) ]
        w_sm = LazyTensor(w_sm.unsqueeze(-2).unsqueeze(-2))
        mu_sm = LazyTensor(mu_sm.unsqueeze(-2).unsqueeze(-2))
        v_sm = LazyTensor(v_sm.unsqueeze(-2).unsqueeze(-2))
        
        #-Should be handled by KeOps natively-#
        #- for maximum stability in combinatorial trees, compute the Q terms and sum.
        arg = d * mu_sm * (2 * math.pi)
        cos_term = arg.cos()
        exp_arg = (d2 * v_sm) * (-2 * (math.pi ** 2))
        exp_term = exp_arg.exp()
        k_sm = (w_sm * cos_term * exp_term).sum(-1) # Summing over Q dimension
        prims.append(k_sm)

        #- Periodic Primitive: exp(-2 * sin^2(pi * d / p) / ls^2)
        p_per = LazyTensor(p_per.view(-1, 1, 1, 1))
        ls_per = LazyTensor(ls_per.view(-1, 1, 1, 1))
        sine_term = (math.pi * d / p_per).sin() ** 2
        k_per = (-2.0 * sine_term / (ls_per ** 2)).exp()
        prims.append(k_per)

        #- Matern-1/2 Primitive for linear & exponential trends-
        ls_mat = LazyTensor(ls_mat.view(-1, 1, 1, 1))
        k_mat = (-d / ls_mat).exp()
        prims.append(k_mat)

        # ---KERNEL COMBINATORICS--- #
        interactions = []
        
        for k_a, k_b in itertools.combinations(prims, 2):
            interactions.append(k_a * k_b)
        
        for k in prims:
            interactions.append(k * k)
        
        interactions.append(prims[0] * prims[1] * prims[2])
        interactions.append(prims[0] * prims[1] * prims[3])
        
        kernels = prims + interactions 

        # --- 4. GATED ROUTING ---shape: (Batch, 1, 1, 16)
        gates = gates.view(x1.size(0), 1, 1, self.kernels_out)
        gate_0 = LazyTensor(gates[..., 0:1])
        k_final = gate_0 * kernels[0]
        
        for i in range(1, self.kernels_out):
            gate_i = LazyTensor(gates[..., i:i+1])
            k_final = k_final + (gate_i * kernels[i])

        return KeOpsLinearOperator(k_final) #-keops lazy tensor-#

    def _forward_diag_fallback(self, x1, x2, **params):
        batch_size, n, _ = x1.shape
        device = x1.device
        
        hyperparams = params.get("gp_params", params)
        
        diag_ones = torch.ones(batch_size, n, device=device)
        diag_sm = hyperparams['w_sm'].sum(-1).view(batch_size, 1).expand(batch_size, n)
        
        prims_diag = [diag_ones, diag_sm, diag_ones, diag_ones]
        
        interactions_diag = []
        for k_a, k_b in itertools.combinations(prims_diag, 2):
            interactions_diag.append(k_a * k_b)
        for k in prims_diag:
            interactions_diag.append(k * k)
        interactions_diag.append(prims_diag[0] * prims_diag[1] * prims_diag[2])
        interactions_diag.append(prims_diag[0] * prims_diag[1] * prims_diag[3])
        
        all_diag = prims_diag + interactions_diag
        
        gates = hyperparams['gates'].view(batch_size, self.kernels_out)
        k_final_diag = gates[:, 0:1] * all_diag[0]
        for i in range(1, self.kernels_out):
            k_final_diag += gates[:, i:i+1] * all_diag[i]
            
        return k_final_diag

class ProbabilisticMixtureMean(gpytorch.means.Mean):
    def __init__(self, k_atoms=30):
        super().__init__()
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(k_atoms, 1) * 0.1)
        )

    def forward(self, x, **params):
        """
        x: [Batch, N, D]
        # pi shape: [Batch, k_atoms]
        # cluster_constants shape: [k_atoms, 1]
        # batch_mean shape: [Batch, 1]
        """
        batch_size, n, _ = x.shape
        
        pi = params.get("pi", None)
        
        if pi is None:
            return torch.zeros(batch_size, n, device=x.device)
            
        
        batch_mean = pi @ self.cluster_constants
        
        return batch_mean.expand(batch_size, n)