import torch

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']

import pykeops
pykeops.clean_pykeops()
pykeops.config.cuda_standalone = True
pykeops.config.use_OpenMP = False

import math
import itertools
import gpytorch
from gpytorch.kernels import Kernel
from pykeops.torch import LazyTensor
import linear_operator
from linear_operator.operators import LinearOperator
from gpytorch.operators import KeOpsLinearOperator
from gpytorch.kernels.keops import RBFKernel



class GenerativeKernel(Kernel):
    has_lengthscale = False

    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.num_primitives = 4     #--> 4 Primitives + 6 Pairs + 4 Self-Squared + 2 Triples = 16 Kernels-#
        self.kernels_out = 16
        self.base_kernel = RBFKernel(batch_shape=batch_shape)

        self.register_parameter(
            name="raw_outputscale",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape))
        )

    def forward(self, x1, x2, diag=False, **params) -> LinearOperator:
        if diag:
            return self._forward_diag_fallback(x1, x2, **params)
        
        hyperparams = params.get("gp_params", None)
        if hyperparams is None:
            raise ValueError("Missing Kernel hyperparameters")
        
        # --- THE TEMPORAL SHIELD ---
        param_keys = ['ls_rbf', 'ls_per', 'p_per', 'ls_mat', 'w_sm', 'mu_sm', 'v_sm', 'gates']
        
        pooled_params = {}
        for k in param_keys:
            val = getattr(hyperparams, k)
            if val.dim() == 4:
                val = val.mean(dim=2)
            pooled_params[k] = val
                
        ls_rbf = pooled_params['ls_rbf']
        ls_per = pooled_params['ls_per']
        p_per = pooled_params['p_per']
        ls_mat = pooled_params['ls_mat']
        w_sm = pooled_params['w_sm']
        mu_sm = pooled_params['mu_sm']
        v_sm = pooled_params['v_sm']
        gates = pooled_params['gates']

        # --- THE RECIPE FOR GPYTORCH ---
        def covar_func(x1_, x2_, **inner_params):
            ls_rbf_in = inner_params['ls_rbf']
            ls_per_in = inner_params['ls_per']
            p_per_in = inner_params['p_per']
            ls_mat_in = inner_params['ls_mat']
            w_sm_in = inner_params['w_sm']
            mu_sm_in = inner_params['mu_sm']
            v_sm_in = inner_params['v_sm']
            gates_in = inner_params['gates']

            x_i = LazyTensor(x1_.unsqueeze(-2))
            x_j = LazyTensor(x2_.unsqueeze(-3))

            x_diff = x_i - x_j
            d2 = (x_diff ** 2).sum(-1) 
            d = d2.sqrt()

            prims = []

            #-RBF
            ls_rbf_lazy = LazyTensor(ls_rbf_in.unsqueeze(-2).unsqueeze(-2))
            k_rbf = (-0.5 * d2 / (ls_rbf_lazy ** 2)).exp()
            prims.append(k_rbf)

            #-Spectral Mixture
            w_sm_lazy = LazyTensor(w_sm_in.unsqueeze(-2).unsqueeze(-2))
            mu_sm_lazy = LazyTensor(mu_sm_in.unsqueeze(-2).unsqueeze(-2))
            v_sm_lazy = LazyTensor(v_sm_in.unsqueeze(-2).unsqueeze(-2))
            
            arg = d * mu_sm_lazy * (2 * math.pi)
            cos_term = arg.cos()
            exp_arg = (d2 * v_sm_lazy) * (-2 * (math.pi ** 2))
            exp_term = exp_arg.exp()
            k_sm = (w_sm_lazy * cos_term * exp_term).sum(-1) 
            prims.append(k_sm)

            #-Periodic
            p_per_lazy = LazyTensor(p_per_in.unsqueeze(-2).unsqueeze(-2))
            ls_per_lazy = LazyTensor(ls_per_in.unsqueeze(-2).unsqueeze(-2))
            sine_term = (math.pi * d / p_per_lazy).sin() ** 2
            k_per = (-2.0 * sine_term / (ls_per_lazy ** 2)).exp()
            prims.append(k_per)

            #-Matern-1/2
            ls_mat_lazy = LazyTensor(ls_mat_in.unsqueeze(-2).unsqueeze(-2))
            k_mat = (-d / ls_mat_lazy).exp()
            prims.append(k_mat)

            # ---KERNEL COMBINATORICS---
            interactions = []
            for k_a, k_b in itertools.combinations(prims, 2):
                interactions.append(k_a * k_b)
            for k in prims:
                interactions.append(k * k)
            
            interactions.append(prims[0] * prims[1] * prims[2])
            interactions.append(prims[0] * prims[1] * prims[3])
            
            kernels = prims + interactions 

            # --- GATED ROUTING ---
            gates_lazy = gates_in.unsqueeze(-2).unsqueeze(-2)
            weighted_kernels = [
                LazyTensor(gates_lazy[..., i:i+1]) * kernels[i] 
                for i in range(self.kernels_out)
            ]

            return sum(weighted_kernels)
        
        keops_op = KeOpsLinearOperator(
            x1.contiguous(), x2.contiguous(), covar_func,
            ls_rbf=ls_rbf, ls_per=ls_per, p_per=p_per, ls_mat=ls_mat,
            w_sm=w_sm, mu_sm=mu_sm, v_sm=v_sm, gates=gates
        )

        #-batch aware output scale-#
        outputscale = torch.nn.functional.softplus(self.raw_outputscale)
        batch_dims = x1.shape[:-2] 
        pad_dims = len(batch_dims) - len(self.batch_shape)
        for _ in range(pad_dims):
            outputscale = outputscale.unsqueeze(0)
            
        outputscale = outputscale.unsqueeze(-1).unsqueeze(-1)
        
        return keops_op * outputscale

    def _forward_diag_fallback(self, x1, x2, **params):
        target_shape = x1.shape[:-1]
        device = x1.device
        
        hyperparams = params.get("gp_params", params)
        pooled_params = {}
        param_keys = ['w_sm', 'gates']
        for k in param_keys:
            val = getattr(hyperparams, k)
            if val.dim() == 4:
                val = val.mean(dim=2)
            pooled_params[k] = val
        
        # --- SHIELD FOR DIAG FALLBACK ---
        w_sm = pooled_params['w_sm']
        gates = pooled_params['gates']
        if w_sm.dim() == 4:
            w_sm = w_sm.mean(dim=2)
            gates = gates.mean(dim=2)
            
        diag_ones = torch.ones(*target_shape, device=device)
        diag_sm = w_sm.sum(-1).unsqueeze(-1).expand(*target_shape)
        
        prims_diag = [diag_ones, diag_sm, diag_ones, diag_ones]
        
        interactions_diag = []
        for k_a, k_b in itertools.combinations(prims_diag, 2):
            interactions_diag.append(k_a * k_b)
        for k in prims_diag:
            interactions_diag.append(k * k)
        interactions_diag.append(prims_diag[0] * prims_diag[1] * prims_diag[2])
        interactions_diag.append(prims_diag[0] * prims_diag[1] * prims_diag[3])
        
        all_diag = prims_diag + interactions_diag
        
        k_final_diag = gates[..., 0:1] * all_diag[0]
        for i in range(1, self.kernels_out):
            k_final_diag += gates[..., i:i+1] * all_diag[i]
            
        outputscale = torch.nn.functional.softplus(self.raw_outputscale)
        
        while outputscale.dim() < k_final_diag.dim():
            outputscale = outputscale.unsqueeze(0)        
        # 3. Final multiplication
        return k_final_diag * outputscale

class ProbabilisticMixtureMean(gpytorch.means.Mean):
    def __init__(self, k_atoms=30, batch_shape=torch.Size([])):
        super().__init__(batch_shape=batch_shape)
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(k_atoms, *batch_shape) * 0.1)
        )

    def forward(self, x, **params):
        target_shape = x.shape[:-1]
        expert_means = params.get("mixture_means_per_expert", None)
        
        # THE SHIELD: Are we evaluating Data or Inducing Points?
        # Data target_shape = [Batch, Experts, SeqLen] (Length 3)
        # Inducing target_shape = [Experts, Num_Inducing] (Length 2)
        if expert_means is not None and len(target_shape) > 2:
            # It's the data! Expand the means over the sequence length.
            return expert_means.unsqueeze(-1).expand(target_shape)
        else:
            # It's the inducing points! Just return 0 to bypass the batch collision.
            return torch.zeros(target_shape, device=x.device)