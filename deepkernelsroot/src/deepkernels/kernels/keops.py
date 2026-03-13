
import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

import pykeops
import linear_operator
from pykeops.torch import LazyTensor
from linear_operator.operators import LinearOperator, KeOpsLinearOperator


pykeops.config.cuda_standalone = True
pykeops.config.use_OpenMP = False

import torch
import torch.nn as nn
import math
import gpytorch
import torch.nn.functional as F
from gpytorch.priors import Prior
from torch.distributions import Laplace


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
        self.register_parameter(name="raw_nkn_weights", parameter=nn.Parameter(torch.randn(4, 8) * 0.1))

        # -- register_constraints ---
        self.register_constraint("raw_scale_12", gpytorch.constraints.GreaterThan(1e-2))
        self.register_constraint("raw_scale_32", gpytorch.constraints.GreaterThan(1e-2))
        self.register_constraint("raw_scale_52", gpytorch.constraints.GreaterThan(1e-2))
        self.register_constraint("raw_rq_alpha", gpytorch.constraints.GreaterThan(1e-2))
        self.register_constraint("raw_linear_scale", gpytorch.constraints.GreaterThan(3e-4))
        self.register_constraint("raw_poly_offset", gpytorch.constraints.GreaterThan(3e-4))
        self.register_constraint("raw_latent_amplitude", gpytorch.constraints.GreaterThan(2e-3))
        self.register_constraint("raw_outputscale", gpytorch.constraints.GreaterThan(2e-3))
        self.register_constraint("raw_inv_bandwidth", gpytorch.constraints.GreaterThan(2e-3))
        self.register_constraint("raw_nkn_weights", gpytorch.constraints.Interval(lower_bound=-2.0, upper_bound=2.0))
        
        #-register priors-#
        self.register_prior("scale_12_prior", gpytorch.priors.GammaPrior(4.0, 1.6), "raw_scale_12")
        self.register_prior("scale_32_prior", gpytorch.priors.GammaPrior(6.0, 4.8), "raw_scale_32")
        self.register_prior("scale_52_prior", gpytorch.priors.GammaPrior(9.0, 6.4), "raw_scale_52")
        self.register_prior("rq_alpha_prior", gpytorch.priors.GammaPrior(3.0, 0.75), "raw_rq_alpha")
        
        self.register_prior("linear_scale_prior", gpytorch.priors.HalfCauchyPrior(1.5), "raw_linear_scale")
        self.register_prior("poly_offset_prior", gpytorch.priors.HalfCauchyPrior(1.0), "raw_poly_offset")
        
        self.register_prior("outputscale_prior", gpytorch.priors.GammaPrior(3.0, 0.25), "raw_outputscale")
        self.register_prior("latent_amplitude_prior", gpytorch.priors.GammaPrior(3.0, 0.3), "raw_latent_amplitude")
        self.register_prior("inv_bandwidth_prior", gpytorch.priors.GammaPrior(4.0, 0.8), "raw_inv_bandwidth")
       
        self.register_prior("nkn_weights_prior", CustomLaplacePrior(loc=0.0, scale=0.45), lambda m: m.nkn_weights)
        
        
        self.initialize(
            raw_scale_12 = torch.tensor(0.0),
            raw_scale_32 = torch.tensor(0.0),
            raw_scale_52 = torch.tensor(0.0),
            raw_rq_alpha = torch.tensor(0.0),
            raw_linear_scale = torch.tensor(0.0),
            raw_poly_offset = torch.tensor(0.0),
            raw_inv_bandwidth = torch.tensor(0.0),
            raw_outputscale = torch.tensor(0.0),
            raw_latent_amplitude = torch.tensor(0.0),
            raw_nkn_weights=torch.tensor(0.0)
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
        if diag:
            return self._forward_diag_fallback(x1, x2, **params)
        params.pop('indices', None)
        params.pop('steps', None)
        params.pop('batch_shape', None)
        params.pop('seq_len', None)
        params.pop('inv_bandwidth', None)
        params.pop('inv_bw', None)
        x1 = x1.contiguous()
        x2 = x2.contiguous() if x2 is not None else x1

        # ==========================================
        # C++ PROGRAM 1: INTERACTION 1
        # ==========================================
        def covar_func_int1(chunkyx1, chunkyx2, **inner_params):
            inner_params.pop('indices', None)
            inner_params.pop('steps', None)
            inner_params.pop('batch_shape', None)
            inner_params.pop('seq_len', None)
            inner_params.pop('inv_bandwidth', None)
            inner_params.pop('inv_bw', None)
            # If chunkyx1 is 2D, make it 3D to match your 8 tasks
            if chunkyx1.dim() == 2:
                chunkyx1 = chunkyx1.unsqueeze(0).expand(8, -1, -1)
            if chunkyx2.dim() == 2:
                chunkyx2 = chunkyx2.unsqueeze(0).expand(8, -1, -1)
            inv_bw = LazyTensor(self.inv_bandwidth.contiguous())
            
            gates_i = LazyTensor(chunkyx1[..., :, None, 0:8].contiguous())
            gates_j = LazyTensor(chunkyx2[..., None, :, 0:8].contiguous())
            
            idx = 8
            
            lin_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            per_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            rat_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            poly_i = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            mat_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            
            idx = 8
            
            lin_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            per_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            rat_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            poly_j = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            mat_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            

            diff_mat = (mat_i - mat_j) * inv_bw
            dist_sq_mat = ((diff_mat) ** 2).sum(-1)
            dist_mat = (dist_sq_mat + 2e-14).sqrt()

            diff_rat = (rat_i - rat_j) * inv_bw
            dist_sq_rat = ((diff_rat) ** 2).sum(-1)
            
            inner_lin = ((lin_i * inv_bw) * (lin_j * inv_bw)).sum(-1)
            inner_poly = ((poly_i * inv_bw) * (poly_j * inv_bw)).sum(-1)

            diff_per = (per_i - per_j) * inv_bw
            dist_per = (((diff_per) ** 2).sum(-1) + 2e-14).sqrt()

            # 3. Primitives
            k_lin = inner_lin * LazyTensor(self.linear_scale.view(1, 1, 1).contiguous())
            k_poly = (LazyTensor(self.poly_offset.view(1, 1, 1).contiguous()) + inner_poly).square()
            k_per = (-2.0 * (math.pi * dist_per).sin().square()).exp()
            
            s_12 = LazyTensor(self.scale_12.view(1, 1, 1).contiguous())
            k_mat12 = (-(dist_mat * s_12)).exp()
            
            s_32 = LazyTensor(self.scale_32.view(1, 1, 1).contiguous())
            sqrt3_d = 1.732 * (dist_mat * s_32)
            k_mat32 = (1.0 + sqrt3_d) * (-sqrt3_d).exp()
            
            s_52 = LazyTensor(self.scale_52.view(1, 1, 1).contiguous())
            sqrt5_d = 2.236 * (dist_mat * s_52)
            k_mat52 = (1.0 + sqrt5_d + (5.0 / 3.0) * (dist_sq_mat * (s_52 ** 2))) * (-sqrt5_d).exp()
            
            alpha = LazyTensor(self.rq_alpha.view(1, 1, 1).contiguous())
            k_rq = (1.0 + dist_sq_rat / (2.0 * alpha)) ** (-alpha)
            k_rbf = (-dist_sq_rat).exp()

            primitives = k_lin.concat(k_poly).concat(k_per).concat(k_mat12).concat(k_mat32).concat(k_mat52).concat(k_rq).concat(k_rbf)
            
            ws = self.nkn_weights.contiguous()
            w0 = LazyTensor(ws[0].view(1, 1, 8).contiguous())
            w1 = LazyTensor(ws[1].view(1, 1, 8).contiguous())

            node0 = (gates_i * gates_j * primitives * w0).sum(-1)
            node1 = (gates_i * gates_j * primitives * w1).sum(-1)
            
            return node0 * node1

        # ==========================================
        # C++ PROGRAM 2: INTERACTION 2
        # ==========================================
        def covar_func_int2(chunkyx1, chunkyx2, **inner_params):
            inner_params.pop('indices', None)
            inner_params.pop('steps', None)
            inner_params.pop('batch_shape', None)
            inner_params.pop('seq_len', None)
            inner_params.pop('inv_bandwidth', None)
            inner_params.pop('inv_bw', None)
            if chunkyx1.dim() == 2:
                chunkyx1 = chunkyx1.unsqueeze(0).expand(8, -1, -1)
            if chunkyx2.dim() == 2:
                chunkyx2 = chunkyx2.unsqueeze(0).expand(8, -1, -1)
            inv_bw = LazyTensor(self.inv_bandwidth.contiguous())
            gates_i = LazyTensor(chunkyx1[..., :, None, 0:8].contiguous())
            gates_j = LazyTensor(chunkyx2[..., None, :, 0:8].contiguous())
            idx = 8
            
            lin_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            per_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            rat_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            poly_i = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            mat_i  = LazyTensor(chunkyx1[..., :, None, idx:idx+32].contiguous()); idx += 32
            
            idx = 8
            
            lin_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            per_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            rat_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            poly_j = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            mat_j  = LazyTensor(chunkyx2[..., None, :, idx:idx+32].contiguous()); idx += 32
            
            # 2. Distance Metrics
            diff_mat = (mat_i - mat_j) * inv_bw
            dist_sq_mat = ((diff_mat) ** 2).sum(-1)
            dist_mat = (dist_sq_mat + 2e-14).sqrt()

            diff_rat = (rat_i - rat_j) * inv_bw
            dist_sq_rat = ((diff_rat) ** 2).sum(-1)
            
            inner_lin = ((lin_i * inv_bw) * (lin_j * inv_bw)).sum(-1)
            inner_poly = ((poly_i * inv_bw) * (poly_j * inv_bw)).sum(-1)

            diff_per = (per_i - per_j) * inv_bw
            dist_per = (((diff_per) ** 2).sum(-1) + 2e-14).sqrt()

            k_lin = inner_lin * LazyTensor(self.linear_scale.view(1, 1, 1).contiguous())
            k_poly = (LazyTensor(self.poly_offset.view(1, 1, 1).contiguous()) + inner_poly).square()
            k_per = (-2.0 * (math.pi * dist_per).sin().square()).exp()
            
            s_12 = LazyTensor(self.scale_12.view(1, 1, 1).contiguous())
            k_mat12 = (-(dist_mat * s_12)).exp()
            
            s_32 = LazyTensor(self.scale_32.view(1, 1, 1).contiguous())
            sqrt3_d = 1.732 * (dist_mat * s_32)
            k_mat32 = (1.0 + sqrt3_d) * (-sqrt3_d).exp()
            
            s_52 = LazyTensor(self.scale_52.view(1, 1, 1).contiguous())
            sqrt5_d = 2.236 * (dist_mat * s_52)
            k_mat52 = (1.0 + sqrt5_d + (5.0 / 3.0) * (dist_sq_mat * (s_52 ** 2))) * (-sqrt5_d).exp()
            
            alpha = LazyTensor(self.rq_alpha.view(1, 1, 1).contiguous())
            k_rq = (1.0 + dist_sq_rat / (2.0 * alpha)) ** (-alpha)
            k_rbf = (-dist_sq_rat).exp()

            primitives = k_lin.concat(k_poly).concat(k_per).concat(k_mat12).concat(k_mat32).concat(k_mat52).concat(k_rq).concat(k_rbf)
            
            ws = self.nkn_weights.contiguous()
            w2 = LazyTensor(ws[2].view(1, 1, 8).contiguous())
            w3 = LazyTensor(ws[3].view(1, 1, 8).contiguous())

            node2 = (gates_i * gates_j * primitives * w2).sum(-1)
            node3 = (gates_i * gates_j * primitives * w3).sum(-1)
            
            return node2 * node3

        # ==========================================
        # GPYTORCH COMPOSITION
        # ==========================================
        op_int1 = KeOpsLinearOperator(x1.contiguous(), x2.contiguous(), covar_func_int1, **params)
        op_int2 = KeOpsLinearOperator(x1.contiguous(), x2.contiguous(), covar_func_int2, **params)
        base_covar = op_int1 + op_int2

        curr_batch_size = base_covar.shape[0] 
        
        rescale = (self.outputscale * self.latent_amplitude).view(curr_batch_size, 1, 1)
        
        return base_covar * rescale
    
    def _forward_diag_fallback(self, x1, x2, **params):
        """Pure PyTorch implementation of the NKN diagonal (distance = 0)"""
        def process_param(p):
            return p.mean(dim=2) if p.dim() == 4 else p
        params.pop('indices', None)
        params.pop('steps', None)
        params.pop('batch_shape', None)
        params.pop('seq_len', None)
        params.pop('latent_dim', None)
        params.pop('batch_size', None)
        gates = x1[..., 0:8].contiguous() 
        linear = x1[..., 8:40].contiguous()
        polynomial = x1[..., 104:136].contiguous()
        
        inv_bw = self.inv_bandwidth.contiguous()
        
        lin_scaled = linear * inv_bw
        inner_lin = lin_scaled.square().sum(dim=-1)
        k_lin_diag = inner_lin * self.linear_scale.view(-1).contiguous()   
        
        poly_scaled = polynomial * inv_bw
        inner_poly = poly_scaled.square().sum(dim=-1)
        k_poly_diag = (self.poly_offset.view(-1) + inner_poly).square()
        ones = torch.ones_like(k_lin_diag)
        
        primitives_diag = torch.stack([
            k_lin_diag, k_poly_diag, ones,
            ones, ones, ones,
            ones, ones
        ], dim=-1)

        #-nkn-#
        g_squared = gates.square().contiguous()
        gated_primitives = g_squared * primitives_diag
        
        nodes = torch.matmul(gated_primitives, self.nkn_weights.t())
        
        interaction_1 = nodes[..., 0] * nodes[..., 1]
        interaction_2 = nodes[..., 2] * nodes[..., 3]
        base_diag = interaction_1 + interaction_2
        os_view = self.outputscale.view(*self.batch_shape, 1)
        amp_view = self.latent_amplitude.view(-1, 1)
        
        return base_diag * os_view * amp_view


class ProbabilisticMixtureMean(gpytorch.means.Mean):
    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__()
        self.k_atoms = 30
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(*batch_shape, self.k_atoms) * 0.1)
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


class CustomLaplacePrior(Prior):
    def __init__(self, loc, scale, validate_args=False, **kwargs):
        super().__init__(batch_shape=torch.Size([]), validate_args=validate_args, **kwargs)
        # Register values as buffers to handle device/dtype moves automatically
        self.register_buffer("loc_val", torch.as_tensor(loc))
        self.register_buffer("scale_val", torch.as_tensor(scale))

    def log_prob(self, parameter):
        # Create the distribution locally, use it, then let it be garbage collected
        dist_obj = Laplace(self.loc_val, self.scale_val)
        return dist_obj.log_prob(parameter)

    def rsample(self, sample_shape=torch.Size()):
        dist_obj = Laplace(self.loc_val, self.scale_val)
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