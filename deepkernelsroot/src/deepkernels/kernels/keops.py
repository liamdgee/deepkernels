import os

import pykeops
import linear_operator
from pykeops.torch import LazyTensor
from linear_operator.operators import LinearOperator, KeOpsLinearOperator


pykeops.config.cuda_standalone = True
pykeops.config.use_OpenMP = False

try:
    if not pykeops.config.gpu_available:
        print("Warning: KeOps GPU not found, falling back to CPU.")
except:
    pass

import torch
import torch.nn as nn
import math
import gpytorch
import torch.nn.functional as F



class GenerativeKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.kernels_out = 32
        self.individual_kernel_input_dim = 32
        self.batch_shape = batch_shape
        #-- register hyperparams-#
        self.register_parameter(name="raw_scale_12", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_scale_32", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_scale_52", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_rq_alpha", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_poly_offset", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_linear_scale", parameter=nn.Parameter(torch.zeros(1)))
        
        #-global hyperparams
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(8)))
        self.register_parameter(name="raw_latent_amplitude", parameter=torch.nn.Parameter(torch.zeros(8)))
        self.register_parameter(name="raw_inv_bandwidth", parameter=nn.Parameter(torch.zeros(32)))
        
        #-register weight params-#
        self.register_parameter(name="raw_nkn_weights", parameter=nn.Parameter(torch.ones(4, 8) * 0.5413))

        # -- register_constraints ---
        self.register_constraint("raw_scale_12", gpytorch.constraints.Interval(0.025, 7.0))
        self.register_constraint("raw_scale_32",  gpytorch.constraints.Interval(0.025, 7.0))
        self.register_constraint("raw_scale_52",  gpytorch.constraints.Interval(0.025, 7.0))
        self.register_constraint("raw_rq_alpha",  gpytorch.constraints.Interval(0.05, 3.5))
        self.register_constraint("raw_linear_scale", gpytorch.constraints.Interval(0.05, 1.5))
        self.register_constraint("raw_poly_offset", gpytorch.constraints.Interval(0.025, 1.5))
        self.register_constraint("raw_latent_amplitude", gpytorch.constraints.Interval(0.01, 4.0))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Interval(0.01, 9.0))
        self.register_constraint("raw_inv_bandwidth", gpytorch.constraints.Interval(2e-3, 1.0))
        self.register_constraint("raw_nkn_weights", gpytorch.constraints.Interval(lower_bound=0.1, upper_bound=1.1))
        
        #-register priors-#
        self.register_prior("scale_12_prior", gpytorch.priors.GammaPrior(4.0, 1.6), lambda m: m.scale_12)
        self.register_prior("scale_32_prior", gpytorch.priors.GammaPrior(6.0, 4.8),lambda m: m.scale_32)
        self.register_prior("scale_52_prior", gpytorch.priors.GammaPrior(9.0, 6.4),lambda m: m.scale_52)
        self.register_prior("rq_alpha_prior", gpytorch.priors.GammaPrior(3.0, 0.75), lambda m: m.rq_alpha)
        
        self.register_prior("linear_scale_prior", gpytorch.priors.HalfCauchyPrior(1.5), lambda m: m.linear_scale)
        self.register_prior("poly_offset_prior", gpytorch.priors.HalfCauchyPrior(1.0), lambda m: m.poly_offset)
        
        self.register_prior("outputscale_prior", gpytorch.priors.GammaPrior(3.0, 0.25), lambda m: m.outputscale)
        self.register_prior("latent_amplitude_prior", gpytorch.priors.GammaPrior(3.0, 0.3), lambda m: m.latent_amplitude)
        self.register_prior("inv_bandwidth_prior", gpytorch.priors.GammaPrior(4.0, 0.8), lambda m: m.latent_amplitude)
       
        self.register_prior("nkn_weights_prior", CustomLaplacePrior(loc=0.0, scale=0.3), "raw_nkn_weights")
        self.raw_latent_amplitude.requires_grad = False
        
        self.initialize(
            raw_scale_12 = torch.tensor(0.0),
            raw_scale_32 = torch.tensor(0.0),
            raw_scale_52 = torch.tensor(0.0),
            raw_rq_alpha = torch.tensor(0.0),
            raw_inv_bandwidth = torch.tensor(-2.0),
            raw_nkn_weights=torch.tensor(0.0),
            raw_outputscale = torch.tensor(-3.0),     
            raw_latent_amplitude = torch.tensor(-3.0),
            raw_poly_offset = torch.tensor(-2.0),
            raw_linear_scale = torch.tensor(-2.0)
        )
    
    @property
    def nkn_weights(self): return self.raw_nkn_weights_constraint.transform(self.raw_nkn_weights)
    @property
    def scale_12(self): return self.raw_scale_12_constraint.transform(self.raw_scale_12)
    @property
    def scale_32(self): return self.raw_scale_32_constraint.transform(self.raw_scale_32)
    @property
    def scale_52(self): return self.raw_scale_52_constraint.transform(self.raw_scale_52)
    @property
    def rq_alpha(self): return self.raw_rq_alpha_constraint.transform(self.raw_rq_alpha)
    @property
    def linear_scale(self): return self.raw_linear_scale_constraint.transform(self.raw_linear_scale)
    @property
    def poly_offset(self): return self.raw_poly_offset_constraint.transform(self.raw_poly_offset)
    @property
    def outputscale(self): return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    @property
    def latent_amplitude(self): return self.raw_latent_amplitude_constraint.transform(self.raw_latent_amplitude)
    @property
    def inv_bandwidth(self): return self.raw_inv_bandwidth_constraint.transform(self.raw_inv_bandwidth)

    def forward(self, x1, x2=None, diag=False, **params) -> LinearOperator:
        params.pop('indices', None)
        params.pop('steps', None)
        params.pop('batch_shape', None)
        params.pop('seq_len', None)
        if diag:
            return self._forward_diag_fallback(x1, x2, **params)
        
        x1 = x1.contiguous()
        x2 = x2.contiguous() if x2 is not None else x1

        if x1.dim() == 2:
            x1 = x1.unsqueeze(0).expand(8, -1, -1)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0).expand(8, -1, -1)
        inv_bw_lt = LazyTensor(self.inv_bandwidth.view(1, 1, 1, 32).contiguous())
        params['inv_bw'] = inv_bw_lt

        ws = F.softplus(self.nkn_weights.contiguous(), beta=1.0) / 8.0
        params['ws'] = ws
        lin_scale_opt = self.linear_scale / 32.0
        params['lin_scale_opt'] = torch.as_tensor(lin_scale_opt, device=x1.device, dtype=x1.dtype)
        s52_sq = self.scale_52 ** 2
        params['s52_sq'] = torch.as_tensor(s52_sq, device=x1.device, dtype=x1.dtype)
        # ==========================================
        # C++ PROGRAM 1: INTERACTION 1
        # ==========================================
        def covar_func(xi, xj, **inner_params):
            inner_params.pop('indices', None)
            inner_params.pop('steps', None)
            inner_params.pop('batch_shape', None)
            inner_params.pop('seq_len', None)
            inv_bw = inner_params.get('inv_bw', 1.0)
            ws = torch.as_tensor(inner_params.get('ws', self.nkn_weights), device=xi.device)
            s52_sq = torch.as_tensor(inner_params.get('s52_sq', 1.0), device=xi.device)
            lin_scale_opt = torch.as_tensor(inner_params.get('lin_scale_opt', 0.03), device=xi.device)
            
            tiny_eps = 1e-12
            
            gates_i = LazyTensor(xi[..., :, None, 0:8].contiguous())
            gates_j = LazyTensor(xj[..., None, :, 0:8].contiguous())
            
            idx = 8
            
            lin_i  = LazyTensor(xi[..., :, None, idx:idx+32].contiguous()); idx += 32
            per_i  = LazyTensor(xi[..., :, None, idx:idx+32].contiguous()); idx += 32
            rat_i  = LazyTensor(xi[..., :, None, idx:idx+32].contiguous()); idx += 32
            poly_i = LazyTensor(xi[..., :, None, idx:idx+32].contiguous()); idx += 32
            mat_i  = LazyTensor(xi[..., :, None, idx:idx+32].contiguous()); idx += 32
            
            idx = 8
            
            lin_j  = LazyTensor(xj[..., None, :, idx:idx+32].contiguous()); idx += 32
            per_j  = LazyTensor(xj[..., None, :, idx:idx+32].contiguous()); idx += 32
            rat_j  = LazyTensor(xj[..., None, :, idx:idx+32].contiguous()); idx += 32
            poly_j = LazyTensor(xj[..., None, :, idx:idx+32].contiguous()); idx += 32
            mat_j  = LazyTensor(xj[..., None, :, idx:idx+32].contiguous()); idx += 32
            
            

            diff_mat = (mat_i - mat_j) * inv_bw
            dist_sq_mat = ((diff_mat) ** 2).sum(-1)
            dist_mat = (dist_sq_mat + tiny_eps).sqrt()

            diff_rat = (rat_i - rat_j) * inv_bw
            dist_sq_rat = ((diff_rat) ** 2).sum(-1)
            
            inner_lin = ((lin_i * inv_bw) * (lin_j * inv_bw)).sum(-1)
            inner_poly = ((poly_i * inv_bw) * (poly_j * inv_bw)).sum(-1)

            diff_per = (per_i - per_j) * inv_bw
            dist_per = (((diff_per) ** 2).sum(-1) + 1e-11).sqrt()

            # 3. Primitives
            k_lin = inner_lin * LazyTensor(lin_scale_opt.view(1, 1, 1, 1).contiguous())
            k_poly = (LazyTensor(self.poly_offset.view(1, 1, 1, 1).contiguous()) + (inner_poly / 32.0)).square()
            k_per = (-2.0 * (math.pi * dist_per).sin().square()).exp()
            
            s_12 = LazyTensor(self.scale_12.view(1, 1, 1, 1).contiguous())
            k_mat12 = (-(dist_mat * s_12)).exp()
            
            s_32 = LazyTensor(self.scale_32.view(1, 1, 1, 1).contiguous())
            sqrt3_d = 1.732 * (dist_mat * s_32)
            k_mat32 = (1.0 + sqrt3_d) * (-sqrt3_d).exp()
            
            s_52 = LazyTensor(self.scale_52.view(1, 1, 1, 1).contiguous())
            s_52_sq_lt = LazyTensor(s52_sq.view(1, 1, 1, 1).contiguous())
            sqrt5_d = 2.236 * (dist_mat * s_52)
            k_mat52 = (1.0 + sqrt5_d + (1.6666667 * dist_sq_mat * s_52_sq_lt)) * (-sqrt5_d).exp()
            
            alpha = LazyTensor(self.rq_alpha.view(1, 1, 1, 1).contiguous())
            k_rq = (1.0 + dist_sq_rat / (2.0 * alpha)) ** (-alpha)
            k_rbf = (-dist_sq_rat).exp()

            primitives = k_lin.concat(k_poly).concat(k_per).concat(k_mat12).concat(k_mat32).concat(k_mat52).concat(k_rq).concat(k_rbf)
            
            base_manifold = gates_i * gates_j * primitives
            
            w0 = LazyTensor(ws[0].view(1, 1, 1, 8).contiguous())
            w1 = LazyTensor(ws[1].view(1, 1, 1, 8).contiguous())
            w2 = LazyTensor(ws[2].view(1, 1, 1, 8).contiguous())
            w3 = LazyTensor(ws[3].view(1, 1, 1, 8).contiguous())
            
            scale = 0.025
            node_jitter = 1e-3
            node0 = (base_manifold * w0).sum(-1) + node_jitter
            node1 = (base_manifold * w1).sum(-1) + node_jitter
            interaction_1 = (node0 * node1) + (scale * 2)
            node2 = (base_manifold * w2).sum(-1) + node_jitter
            node3 = (base_manifold * w3).sum(-1) + node_jitter
            interaction_2 = (node2 * node3) + scale
            return interaction_1 + interaction_2
        
        # ==========================================
        # GPYTORCH COMPOSITION
        # ==========================================
        
        xi = x1.contiguous()
        xj = x2.contiguous()

        base_covar = KeOpsLinearOperator(xi, xj, covar_func, **params)

        curr_batch_size = base_covar.shape[0] 
        n = base_covar.shape[-1]
        rescale = (self.outputscale * self.latent_amplitude).view(curr_batch_size, 1, 1)
        
        scaled_covar = base_covar * rescale
        is_auto_covar = x1.shape == x2.shape and torch.equal(x1, x2)
        
        if is_auto_covar:
            n = base_covar.shape[-1]
            jitter_diag = torch.ones(curr_batch_size, n, device=x1.device, dtype=x1.dtype) * 1e-3
            jitter = linear_operator.operators.DiagLinearOperator(jitter_diag)
            return scaled_covar + jitter
        else:
            return scaled_covar
        
    def _diagonal(self):
        res = self.covar_func(self.x1, self.x1, **self.params)
        if hasattr(res, 'diag'):
            return res.diag() # This usually returns a standard torch Tensor
        return torch.ones(self.x1.shape[:-1], device=self.x1.device, dtype=self.x1.dtype)

    def _forward_diag_fallback(self, x1, x2, **params):
        """Pure PyTorch implementation of the NKN diagonal (distance = 0)"""
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0).expand(8, -1, -1)
        if x2 is not None and not torch.equal(x1, x2):
            raise RuntimeError("Diagonal fallback called with x1 != x2. Distance is not zero! Check your evaluation code.")
        params.pop('indices', None)
        params.pop('steps', None)
        params.pop('batch_shape', None)
        params.pop('seq_len', None)
        params.pop('batch_size', None)
        gates = x1[..., 0:8]
        linear = x1[..., 8:40]
        polynomial = x1[..., 104:136]
        
        inv_bw = self.inv_bandwidth.view(-1)
        
        lin_scaled = linear * inv_bw
        inner_lin = lin_scaled.square().sum(dim=-1) / 32.0
        k_lin_diag = inner_lin * self.linear_scale.view(-1)
        
        poly_scaled = polynomial * inv_bw
        inner_poly = poly_scaled.square().sum(dim=-1) / 32.0
        k_poly_diag = (self.poly_offset.view(-1) + inner_poly).square()
        ones = torch.ones_like(k_lin_diag)
        
        primitives_diag = torch.stack([
            k_lin_diag, k_poly_diag, ones,
            ones, ones, ones,
            ones, ones
        ], dim=-1)

        #-nkn-#
        g_squared = gates.square()
        ws = F.softplus(self.nkn_weights, beta=1.0)
        def get_safe_node(w_idx):
            return ((g_squared * primitives_diag * ws[w_idx]).sum(dim=-1) / 8.0) + 3e-3

        node0 = get_safe_node(0)
        node1 = get_safe_node(1)
        node2 = get_safe_node(2)
        node3 = get_safe_node(3)
        base_diag = (node0 * node1) + (node2 * node3)
        os_view = self.outputscale.view(-1, 1)
        amp_view = self.latent_amplitude.view(-1, 1)
        return base_diag * os_view * amp_view
        

class ProbabilisticMixtureMean(gpytorch.means.Mean):
    def __init__(self, batch_shape=torch.Size([]), k_atoms=30, num_latents=8, **kwargs):
        super().__init__()
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(num_latents, k_atoms) * 0.1)
        )

    def forward(self, x, **params):
        """
        x shape: [num_latents, ..., N, 198]
        """
        target_shape = x.shape[:-1]
        pi = x[..., 168: 198].contiguous()
        c_T = self.cluster_constants.unsqueeze(-2).contiguous()
        latent_means = (pi * c_T).sum(dim=-1)
        return latent_means


class CustomLaplacePrior(gpytorch.priors.Prior):
    def __init__(self, loc, scale, validate_args=False, **kwargs):
        super().__init__(batch_shape=torch.Size([]), validate_args=validate_args, **kwargs)
        self.register_buffer("loc_val", torch.as_tensor(loc))
        self.register_buffer("scale_val", torch.as_tensor(scale))

    def log_prob(self, parameter):
        dist_obj = torch.distributions.Laplace(self.loc_val, self.scale_val)
        return dist_obj.log_prob(parameter)

    def rsample(self, sample_shape=torch.Size()):
        dist_obj = torch.distributions.Laplace(self.loc_val, self.scale_val)
        return dist_obj.rsample(sample_shape)
    
    def __repr__(self):
        return f"CustomLaplacePrior(loc={self.loc_val.item()}, scale={self.scale_val.item()})"

    @property
    def loc(self): return self.loc_val

    @property
    def scale(self): return self.scale_val
    
    @property
    def arg_constraints(self):
        return {}