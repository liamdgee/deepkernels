import torch

import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

import pykeops
import torch.nn.functional as F
from pykeops.torch import LazyTensor
import linear_operator
from linear_operator.operators import LinearOperator, KeOpsLinearOperator
from gpytorch.kernels.keops import RBFKernel


from gpytorch.priors import Prior
from torch.distributions import Laplace

##--pykeops.clean_pykeops()
pykeops.config.cuda_standalone = True
pykeops.config.use_OpenMP = False

import torch.nn as nn
import math
import gpytorch


class GenerativeKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__(batch_shape=batch_shape, **kwargs)
        self.kwargs = kwargs
        self.kernels_out = self.kwargs.get('kernels_out', 32)
        self.individual_kernel_input_dim = self.kwargs.get('individual_kernel_input_dim', 32)
        self.base_kernel = RBFKernel(batch_shape=batch_shape)
        self.batch_shape = batch_shape

        self.SQRT3 = math.sqrt(3.0)
        self.SQRT5 = math.sqrt(5.0)
        scale_constraint = gpytorch.constraints.GreaterThan(1e-4)
        positive_constraint = gpytorch.constraints.Positive()

        #-- register hyperparams-#
        self.register_parameter(name="raw_scale_12", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_scale_32", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_scale_52", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_rq_alpha", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_poly_offset", parameter=nn.Parameter(torch.zeros(1)))
        self.register_parameter(name="raw_linear_scale", parameter=nn.Parameter(torch.zeros(1)))
        
        #-global hyperparams
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(*batch_shape)))
        self.register_parameter(name="raw_latent_amplitude", parameter=torch.nn.Parameter(torch.zeros(*batch_shape)))
        self.register_parameter(name="raw_inv_bandwidth", parameter=nn.Parameter(torch.zeros(1, self.individual_kernel_input_dim)))
        
        #-register weight params-#
        self.register_parameter(name="raw_nkn_weights", parameter=nn.Parameter(torch.zeros(4, 8)))

        # -- register_constraints ---
        self.register_constraint("raw_scale_12", scale_constraint)
        self.register_constraint("raw_scale_32", scale_constraint)
        self.register_constraint("raw_scale_52", scale_constraint)
        self.register_constraint("raw_rq_alpha", scale_constraint)
        self.register_constraint("raw_linear_scale", scale_constraint)
        self.register_constraint("raw_poly_offset", positive_constraint)
        self.register_constraint("raw_latent_amplitude", positive_constraint)
        self.register_constraint("raw_outputscale", positive_constraint)
        self.register_constraint("raw_inv_bandwidth", positive_constraint)
        self.register_constraint("raw_nkn_weights", positive_constraint)
        
        #-register priors-#
        self.register_prior("scale_12_prior", gpytorch.priors.GammaPrior(4.0, 0.6), "raw_scale_12")
        self.register_prior("scale_32_prior", gpytorch.priors.GammaPrior(4.0, 3.0), "raw_scale_32")
        self.register_prior("scale_52_prior", gpytorch.priors.GammaPrior(4.0, 15.0), "raw_scale_52")
        self.register_prior("rq_alpha_prior", gpytorch.priors.GammaPrior(3.0, 0.75), "raw_rq_alpha")
        
        self.register_prior("linear_scale_prior", gpytorch.priors.HalfCauchyPrior(2.0), "raw_linear_scale")
        self.register_prior("poly_offset_prior", gpytorch.priors.HalfCauchyPrior(1.0), "raw_poly_offset")
        
        self.register_prior("outputscale_prior", gpytorch.priors.GammaPrior(2.0, 0.15), "raw_outputscale")
        self.register_prior("latent_amplitude_prior", gpytorch.priors.GammaPrior(2.0, 0.15), "raw_latent_amplitude")
        self.register_prior("inv_bandwidth_prior", gpytorch.priors.GammaPrior(3.0, 1.0), "raw_inv_bandwidth")
       
        self.register_prior("nkn_weights_prior", CustomLaplacePrior(loc=0.0, scale=0.3), lambda m: m.nkn_weights)
        
        initial_weights = (torch.ones(4, 8) / 8.0) + (torch.randn(4, 8) * 0.005)
        
        self.initialize(
            scale_12 = torch.tensor(1.6),
            scale_32 = torch.tensor(0.7),
            scale_52 = torch.tensor(0.1),
            rq_alpha = torch.tensor(1.0),
            linear_scale = torch.tensor(0.5),
            poly_offset = torch.tensor(0.2),
            inv_bandwidth = torch.ones(1, 32) * torch.exp(torch.tensor(0.035)),
            outputscale = torch.exp(torch.tensor(0.015)),
            latent_amplitude = torch.exp(torch.tensor(0.025)),
            nkn_weights=initial_weights
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
    
    #-helper-#
    def _safe_tensor(self, value):
        """Helper to ensure we always pass a tensor to the inverse transform"""
        return value if torch.is_tensor(value) else torch.tensor(value)

    #-setters-#
    @scale_12.setter
    def scale_12(self, value):
        self.initialize(**{"scale_12": self.raw_scale_12_constraint.inverse_transform(self._safe_tensor(value))})
    
    @scale_32.setter
    def scale_32(self, value):
        self.initialize(**{"scale_32": self.raw_scale_32_constraint.inverse_transform(self._safe_tensor(value))})
    
    @scale_52.setter
    def scale_52(self, value):
        self.initialize(**{"scale_52": self.raw_scale_52_constraint.inverse_transform(self._safe_tensor(value))})
    
    @rq_alpha.setter
    def rq_alpha(self, value):
        self.initialize(**{"rq_alpha": self.raw_rq_alpha_constraint.inverse_transform(self._safe_tensor(value))})
    
    @linear_scale.setter
    def linear_scale(self, value):
        self.initialize(**{"linear_scale": self.raw_linear_scale_constraint.inverse_transform(self._safe_tensor(value))})
    
    @poly_offset.setter
    def poly_offset(self, value):
        self.initialize(**{"poly_offset": self.raw_poly_offset_constraint.inverse_transform(self._safe_tensor(value))})
    
    @latent_amplitude.setter
    def latent_amplitude(self, value):
        self.initialize(**{"latent_amplitude": self.raw_latent_amplitude_constraint.inverse_transform(self._safe_tensor(value))})
    
    @outputscale.setter
    def outputscale(self, value):
        self.initialize(**{"outputscale": self.raw_outputscale_constraint.inverse_transform(self._safe_tensor(value))})
    
    @inv_bandwidth.setter
    def inv_bandwidth(self, value):
        self.initialize(**{"inv_bandwidth": self.raw_inv_bandwidth_constraint.inverse_transform(self._safe_tensor(value))})
    
    @nkn_weights.setter
    def nkn_weights(self, value):
        self.initialize(**{"nkn_weights": self.raw_nkn_weights_constraint.inverse_transform(self._safe_tensor(value))})

    
    def forward(self, x1, x2, diag=False, **params) -> LinearOperator:
        if diag:
            return self._forward_diag_fallback(x1, x2, **params)
        
        def covar_func(chunkyx1, chunkyx2, **inner_params):
            idx = 8
            lin_i = LazyTensor(chunkyx1[..., idx:idx+32].unsqueeze(-2)); idx += 32
            per_i = LazyTensor(chunkyx1[..., idx:idx+32].unsqueeze(-2)); idx += 32
            rat_i = LazyTensor(chunkyx1[..., idx:idx+32].unsqueeze(-2)); idx += 32
            poly_i = LazyTensor(chunkyx1[..., idx:idx+32].unsqueeze(-2)); idx += 32
            mat_i = LazyTensor(chunkyx1[..., idx:idx+32].unsqueeze(-2)); idx += 32
            
            idx = 8
            lin_j = LazyTensor(chunkyx2[..., idx:idx+32].unsqueeze(-3)); idx += 32
            per_j = LazyTensor(chunkyx2[..., idx:idx+32].unsqueeze(-3)); idx += 32
            rat_j = LazyTensor(chunkyx2[..., idx:idx+32].unsqueeze(-3)); idx += 32
            poly_j = LazyTensor(chunkyx2[..., idx:idx+32].unsqueeze(-3)); idx += 32
            mat_j = LazyTensor(chunkyx2[..., idx:idx+32].unsqueeze(-3)); idx += 32

            gate_0i = LazyTensor(chunkyx1[..., 0:1].unsqueeze(-2))
            gate_1i = LazyTensor(chunkyx1[..., 1:2].unsqueeze(-2))
            gate_2i = LazyTensor(chunkyx1[..., 2:3].unsqueeze(-2))
            gate_3i = LazyTensor(chunkyx1[..., 3:4].unsqueeze(-2))
            gate_4i = LazyTensor(chunkyx1[..., 4:5].unsqueeze(-2))
            gate_5i = LazyTensor(chunkyx1[..., 5:6].unsqueeze(-2))
            gate_6i = LazyTensor(chunkyx1[..., 6:7].unsqueeze(-2))
            gate_7i = LazyTensor(chunkyx1[..., 7:8].unsqueeze(-2))

            gate_0j = LazyTensor(chunkyx2[..., 0:1].unsqueeze(-3))
            gate_1j = LazyTensor(chunkyx2[..., 1:2].unsqueeze(-3))
            gate_2j = LazyTensor(chunkyx2[..., 2:3].unsqueeze(-3))
            gate_3j = LazyTensor(chunkyx2[..., 3:4].unsqueeze(-3))
            gate_4j = LazyTensor(chunkyx2[..., 4:5].unsqueeze(-3))
            gate_5j = LazyTensor(chunkyx2[..., 5:6].unsqueeze(-3))
            gate_6j = LazyTensor(chunkyx2[..., 6:7].unsqueeze(-3))
            gate_7j = LazyTensor(chunkyx2[..., 7:8].unsqueeze(-3))
            
            # ==========================================
            #-Symbolic Metrics and primitives
            # ==========================================
            inv_bw = LazyTensor(self.inv_bandwidth.view(1, 1, 32))
            #-metrics-#
            #-for matern family-#-
            diff_mat = (mat_i - mat_j) * inv_bw
            dist_sq_mat = ((diff_mat) ** 2).sum(-1)
            dist_mat = dist_sq_mat.sqrt()

            #-for rational & rbf-#
            diff_rat = (rat_i - rat_j) * inv_bw
            dist_sq_rat = ((diff_rat) ** 2).sum(-1)
            
            #- linear
            lin_i_scaled = lin_i * inv_bw
            lin_j_scaled = lin_j * inv_bw
            inner_lin = (lin_i_scaled * lin_j_scaled).sum(-1)

            #-poly
            poly_i_scaled = poly_i * inv_bw
            poly_j_scaled = poly_j * inv_bw
            inner_poly = (poly_i_scaled * poly_j_scaled).sum(-1)

            #-periodic
            diff_per = (per_i - per_j) * inv_bw
            dist_per = ((diff_per) ** 2).sum(-1).sqrt()

            #-primitive kernels-#
            #-linear-#
            k_lin = inner_lin * LazyTensor(self.linear_scale.view(1, 1, 1))
            #-poly-
            k_poly = (LazyTensor(self.poly_offset.view(1, 1, 1)) + inner_poly).square()
            #-periodic
            k_per = (-2.0 * (math.pi * dist_per).sin().square()).exp()
            
            #materns-#
            #-matern 1/2
            s_12 = LazyTensor(self.scale_12.view(1, 1, 1))
            k_mat12 = (-(dist_mat * s_12)).exp()
            
            #-matern 3/2
            s_32 = LazyTensor(self.scale_32.view(1, 1, 1))
            sqrt3_d = self.SQRT3 * (dist_mat * s_32)
            k_mat32 = (1.0 + sqrt3_d) * (-sqrt3_d).exp()
            
            #-matern 5/2-#
            s_52 = LazyTensor(self.scale_52.view(1, 1, 1))
            sqrt5_d = self.SQRT5 * (dist_mat * s_52)
            k_mat52 = (1.0 + sqrt5_d + (5.0 / 3.0) * (dist_sq_mat * (s_52 ** 2))) * (-sqrt5_d).exp()
            
            #-rational quadratic-#
            alpha = LazyTensor(self.rq_alpha.view(1, 1, 1))
            k_rq = (1.0 + dist_sq_rat / (2.0 * alpha)) ** (-alpha)
            
            #-RBF-#
            k_rbf = (-dist_sq_rat).exp()

            #-streamlined kernel algebra-#
            primitives = [k_lin, k_poly, k_per, k_mat12, k_mat32, k_mat52, k_rq, k_rbf]
            
            ws = self.nkn_weights

            node0 = (gate_0i * gate_0j * primitives[0] * LazyTensor(ws[0, 0].view(1, 1, 1)))
            node0 += (gate_1i * gate_1j * primitives[1] * LazyTensor(ws[0, 1].view(1, 1, 1)))
            node0 += (gate_2i * gate_2j * primitives[2] * LazyTensor(ws[0, 2].view(1, 1, 1)))
            node0 += (gate_3i * gate_3j * primitives[3] * LazyTensor(ws[0, 3].view(1, 1, 1)))
            node0 += (gate_4i * gate_4j * primitives[4] * LazyTensor(ws[0, 4].view(1, 1, 1)))
            node0 += (gate_5i * gate_5j * primitives[5] * LazyTensor(ws[0, 5].view(1, 1, 1)))
            node0 += (gate_6i * gate_6j * primitives[6] * LazyTensor(ws[0, 6].view(1, 1, 1)))
            node0 += (gate_7i * gate_7j * primitives[7] * LazyTensor(ws[0, 7].view(1, 1, 1)))

            node1 = (gate_0i * gate_0j * primitives[0] * LazyTensor(ws[1, 0].view(1, 1, 1)))
            node1 += (gate_1i * gate_1j * primitives[1] * LazyTensor(ws[1, 1].view(1, 1, 1)))
            node1 += (gate_2i * gate_2j * primitives[2] * LazyTensor(ws[1, 2].view(1, 1, 1)))
            node1 += (gate_3i * gate_3j * primitives[3] * LazyTensor(ws[1, 3].view(1, 1, 1)))
            node1 += (gate_4i * gate_4j * primitives[4] * LazyTensor(ws[1, 4].view(1, 1, 1)))
            node1 += (gate_5i * gate_5j * primitives[5] * LazyTensor(ws[1, 5].view(1, 1, 1)))
            node1 += (gate_6i * gate_6j * primitives[6] * LazyTensor(ws[1, 6].view(1, 1, 1)))
            node1 += (gate_7i * gate_7j * primitives[7] * LazyTensor(ws[1, 7].view(1, 1, 1)))

            node2 = (gate_0i * gate_0j * primitives[0] * LazyTensor(ws[2, 0].view(1, 1, 1)))
            node2 += (gate_1i * gate_1j * primitives[1] * LazyTensor(ws[2, 1].view(1, 1, 1)))
            node2 += (gate_2i * gate_2j * primitives[2] * LazyTensor(ws[2, 2].view(1, 1, 1)))
            node2 += (gate_3i * gate_3j * primitives[3] * LazyTensor(ws[2, 3].view(1, 1, 1)))
            node2 += (gate_4i * gate_4j * primitives[4] * LazyTensor(ws[2, 4].view(1, 1, 1)))
            node2 += (gate_5i * gate_5j * primitives[5] * LazyTensor(ws[2, 5].view(1, 1, 1)))
            node2 += (gate_6i * gate_6j * primitives[6] * LazyTensor(ws[2, 6].view(1, 1, 1)))
            node2 += (gate_7i * gate_7j * primitives[7] * LazyTensor(ws[2, 7].view(1, 1, 1)))

            node3 = (gate_0i * gate_0j * primitives[0] * LazyTensor(ws[3, 0].view(1, 1, 1)))
            node3 += (gate_1i * gate_1j * primitives[1] * LazyTensor(ws[3, 1].view(1, 1, 1)))
            node3 += (gate_2i * gate_2j * primitives[2] * LazyTensor(ws[3, 2].view(1, 1, 1)))
            node3 += (gate_3i * gate_3j * primitives[3] * LazyTensor(ws[3, 3].view(1, 1, 1)))
            node3 += (gate_4i * gate_4j * primitives[4] * LazyTensor(ws[3, 4].view(1, 1, 1)))
            node3 += (gate_5i * gate_5j * primitives[5] * LazyTensor(ws[3, 5].view(1, 1, 1)))
            node3 += (gate_6i * gate_6j * primitives[6] * LazyTensor(ws[3, 6].view(1, 1, 1)))
            node3 += (gate_7i * gate_7j * primitives[7] * LazyTensor(ws[3, 7].view(1, 1, 1)))
            
            interaction_1 = node0 * node1
            interaction_2 = node2 * node3
        
            return interaction_1 + interaction_2
        
        base_covar = KeOpsLinearOperator(x1.contiguous(), x2.contiguous(), covar_func)

        global_scale = self.outputscale.view(*self.batch_shape, 1, 1)
        num_latents = self.latent_amplitude.size(0)
        multitask_amp = self.latent_amplitude.view(num_latents, 1, 1)
        return base_covar * global_scale * multitask_amp
    
    def _forward_diag_fallback(self, x1, x2, **params):
        """Pure PyTorch implementation of the NKN diagonal (distance = 0)"""
        def process_param(p):
            return p.mean(dim=2) if p.dim() == 4 else p
        
        gates = x1[..., 0:8]
        linear = x1[..., 8:40]
        polynomial = x1[..., 104:136]
        
        inv_bw = self.inv_bandwidth.view(-1)                 # [32]
        
        lin_scaled = linear * inv_bw
        inner_lin = lin_scaled.square().sum(dim=-1)          # [..., N]
        k_lin_diag = inner_lin * self.linear_scale.view(-1)
        
        poly_scaled = polynomial * inv_bw
        inner_poly = poly_scaled.square().sum(dim=-1)        # [..., N]
        k_poly_diag = (self.poly_offset.view(-1) + inner_poly).square()


        #-stationary primitives eval to 1 when dist is 0-#
        ones = torch.ones_like(k_lin_diag)
        
        primitives_diag = torch.stack([
            k_lin_diag, k_poly_diag, ones,                   # lin, poly, per
            ones, ones, ones,                                # mat12, mat32, mat52
            ones, ones                                       # rq, rbf
        ], dim=-1)                                           # [..., N, 8]

        #-nkn-#
        g_squared = gates.square()                           # [..., N, 8]
        gated_primitives = g_squared * primitives_diag       # [..., N, 8]
        
        nodes = torch.matmul(gated_primitives, self.nkn_weights.t())
        
        interaction_1 = nodes[..., 0] * nodes[..., 1]        # [..., N]
        interaction_2 = nodes[..., 2] * nodes[..., 3]        # [..., N]
        base_diag = interaction_1 + interaction_2            # [..., N]
        os_view = self.outputscale.view(*self.batch_shape, 1)
        base_diag = base_diag * os_view
        
        base_diag = base_diag.unsqueeze(0)
        
        num_latents = self.latent_amplitude.size(0)
        amp_shape = [num_latents] + [1] * (base_diag.dim() - 1)
        amp_view = self.latent_amplitude.view(*amp_shape)
        
        return base_diag * amp_view


class ProbabilisticMixtureMean(gpytorch.means.Mean):
    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__()
        self.k_atoms = kwargs.get("k_atoms", 30) #--this is an unncessary headache but it expects k_atoms=30
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(30, self.k_atoms, *batch_shape) * 0.1)
        )

    def forward(self, x, **params):
        """
        x shape: [num_latents, ..., N, 198]
        """
        target_shape = x.shape[:-1]
        pi = x[..., 168: 198]
        c_T = self.cluster_constants.t() 
        for _ in range(pi.dim() - c_T.dim()):
            c_T = c_T.unsqueeze(-2)
        latent_means = (pi * c_T).sum(dim=-1)
        return latent_means


class CustomLaplacePrior(Prior):
    def __init__(self, loc, scale, validate_args=False, **kwargs):
        super().__init__(validate_args=validate_args, **kwargs)
        # Register as buffers so .to("cuda") and .type() work correctly
        self.register_buffer("loc_val", torch.as_tensor(loc, dtype=torch.float32))
        self.register_buffer("scale_val", torch.as_tensor(scale, dtype=torch.float32))

    def log_prob(self, parameter):
        return Laplace(self.loc_val, self.scale_val).log_prob(parameter)

    def rsample(self, sample_shape=torch.Size()):
        return Laplace(self.loc_val, self.scale_val).rsample(sample_shape)
    
    @property
    def loc(self): return self.loc_val

    @property
    def scale(self): return self.scale_val