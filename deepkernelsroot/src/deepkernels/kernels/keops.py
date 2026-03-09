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
from gpytorch.kernels import Kernel


class GenerativeKernel(Kernel):
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
        self.register_parameter(name="raw_nkn_weights", parameter=nn.Parameter(torch.empty(4, 8)))
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

        #-initialising values-#
        self.raw_scale_12.data.fill_(self.raw_scale_12_constraint.inverse_transform(torch.tensor(1.6)).item())
        self.raw_scale_32.data.fill_(self.raw_scale_32_constraint.inverse_transform(torch.tensor(0.7)).item())
        self.raw_scale_52.data.fill_(self.raw_scale_52_constraint.inverse_transform(torch.tensor(0.1)).item())
        self.raw_rq_alpha.data.fill_(self.raw_rq_alpha_constraint.inverse_transform(torch.tensor(1.0)).item())
        self.raw_linear_scale.data.fill_(self.raw_linear_scale_constraint.inverse_transform(torch.tensor(0.5)).item())
        self.raw_poly_offset.data.fill_(self.raw_poly_offset_constraint.inverse_transform(torch.tensor(0.2)).item())
        
        self.raw_inv_bandwidth.data.copy_(self.raw_inv_bandwidth_constraint.inverse_transform(torch.exp(torch.tensor(0.025))))
        self.raw_outputscale.data.copy_(self.raw_outputscale_constraint.inverse_transform(torch.exp(torch.tensor(0.025))))
        self.raw_latent_amplitude.data.copy_(self.raw_latent_amplitude_constraint.inverse_transform(torch.exp(torch.tensor(0.025))))
        
        with torch.no_grad():
            self.raw_nkn_weights.copy_(self.raw_nkn_weights_constraint.inverse_transform(torch.tensor(1.0 / 8)))
            self.raw_nkn_weights.add_(torch.randn(4, 8) * 0.005)
        
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
       
        self.register_prior(
            "nkn_weights_prior", 
            CustomLaplacePrior(loc=0.0, scale=0.1), 
            lambda m: m.nkn_weights
        )

        self.initialize(scale_12=)
    
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

    def forward(self, x1, x2, diag=False, **params) -> LinearOperator:
        if diag:
            return self._forward_diag_fallback(x1, x2, **params)
        
        hyperparams = params.get("gp_params", None)
        if hyperparams is None:
            raise ValueError("Missing Kernel hyperparameters")
        
        test_hyperparams = params.get("gp_params_eval", None)
        
        if test_hyperparams is None:
            test_hyperparams = hyperparams
            is_same = True
        else:
            is_same = False
        
        param_keys = ['gates', 'linear', 'periodic', 'rational', 'polynomial', 'matern']
        
        pooled_params_train = {}
        pooled_params_test = {}

        for k in param_keys:
            train_val = getattr(hyperparams, k)
            if train_val.dim() == 4:
                train_val = train_val.mean(dim=2)
            pooled_params_train[k] = train_val
            if is_same:
                pooled_params_test = pooled_params_train
            if not is_same:
                test_val = getattr(test_hyperparams, k)
                if test_val.dim() == 4:
                    test_val = test_val.mean(dim=2)
                pooled_params_test[k] = test_val
            

                
        gates_i = pooled_params_train['gates']
        linear_i = pooled_params_train['linear']
        periodic_i = pooled_params_train['periodic']
        rational_i = pooled_params_train['rational']
        polynomial_i = pooled_params_train['polynomial']
        matern_i = pooled_params_train['matern']

        if not is_same:
            gates_j = pooled_params_test['gates']
            linear_j = pooled_params_test['linear']
            periodic_j = pooled_params_test['periodic']
            rational_j = pooled_params_test['rational']
            polynomial_j = pooled_params_test['polynomial']
            matern_j = pooled_params_test['matern']
        
        else:
            gates_j, linear_j, periodic_j = gates_i, linear_i, periodic_i
            rational_j, polynomial_j, matern_j = rational_i, polynomial_i, matern_i
        

        def covar_func(x1_params, x2_params, **inner_params):
            mat_i = LazyTensor(pooled_params_train['matern'].unsqueeze(1))
            rat_i = LazyTensor(pooled_params_train['rational'].unsqueeze(1))
            lin_i = LazyTensor(pooled_params_train['linear'].unsqueeze(1))
            poly_i = LazyTensor(pooled_params_train['polynomial'].unsqueeze(1))
            per_i = LazyTensor(pooled_params_train['periodic'].unsqueeze(1))
            gate_i = LazyTensor(pooled_params_train['gates'].unsqueeze(1))
            
            if is_same:
                mat_j = LazyTensor(pooled_params_train['matern'].unsqueeze(0))
                rat_j = LazyTensor(pooled_params_train['rational'].unsqueeze(0))
                lin_j = LazyTensor(pooled_params_train['linear'].unsqueeze(0))
                poly_j = LazyTensor(pooled_params_train['polynomial'].unsqueeze(0))
                per_j = LazyTensor(pooled_params_train['periodic'].unsqueeze(0))
                gate_j = LazyTensor(pooled_params_train['gates'].unsqueeze(0))
            else:
                mat_j = LazyTensor(pooled_params_test['matern'].unsqueeze(0))
                rat_j = LazyTensor(pooled_params_test['rational'].unsqueeze(0))
                lin_j = LazyTensor(pooled_params_test['linear'].unsqueeze(0))
                poly_j = LazyTensor(pooled_params_test['polynomial'].unsqueeze(0))
                per_j = LazyTensor(pooled_params_test['periodic'].unsqueeze(0))
                gate_j = LazyTensor(pooled_params_test['gates'].unsqueeze(0))
            
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

            primitives = [k_lin, k_poly, k_per, k_mat12, k_mat32, k_mat52, k_rq, k_rbf]
            gate_0i = LazyTensor(pooled_params_train['gates'][:, 0].view(-1, 1, 1))
            gate_1i = LazyTensor(pooled_params_train['gates'][:, 1].view(-1, 1, 1))
            gate_2i = LazyTensor(pooled_params_train['gates'][:, 2].view(-1, 1, 1))
            gate_3i = LazyTensor(pooled_params_train['gates'][:, 3].view(-1, 1, 1))
            gate_4i = LazyTensor(pooled_params_train['gates'][:, 4].view(-1, 1, 1))
            gate_5i = LazyTensor(pooled_params_train['gates'][:, 5].view(-1, 1, 1))
            gate_6i = LazyTensor(pooled_params_train['gates'][:, 6].view(-1, 1, 1))
            gate_7i = LazyTensor(pooled_params_train['gates'][:, 7].view(-1, 1, 1))

            if not is_same:
                gate_0j = LazyTensor(pooled_params_test['gates'][:, 0].view(1, -1, 1))
                gate_1j = LazyTensor(pooled_params_test['gates'][:, 1].view(1, -1, 1))
                gate_2j = LazyTensor(pooled_params_test['gates'][:, 2].view(1, -1, 1))
                gate_3j = LazyTensor(pooled_params_test['gates'][:, 3].view(1, -1, 1))
                gate_4j = LazyTensor(pooled_params_test['gates'][:, 4].view(1, -1, 1))
                gate_5j = LazyTensor(pooled_params_test['gates'][:, 5].view(1, -1, 1))
                gate_6j = LazyTensor(pooled_params_test['gates'][:, 6].view(1, -1, 1))
                gate_7j = LazyTensor(pooled_params_test['gates'][:, 7].view(1, -1, 1))
            
            else:
                gate_0j = LazyTensor(pooled_params_train['gates'][:, 0].view(1, -1, 1))
                gate_1j = LazyTensor(pooled_params_train['gates'][:, 1].view(1, -1, 1))
                gate_2j = LazyTensor(pooled_params_train['gates'][:, 2].view(1, -1, 1))
                gate_3j = LazyTensor(pooled_params_train['gates'][:, 3].view(1, -1, 1))
                gate_4j = LazyTensor(pooled_params_train['gates'][:, 4].view(1, -1, 1))
                gate_5j = LazyTensor(pooled_params_train['gates'][:, 5].view(1, -1, 1))
                gate_6j = LazyTensor(pooled_params_train['gates'][:, 6].view(1, -1, 1))
                gate_7j = LazyTensor(pooled_params_train['gates'][:, 7].view(1, -1, 1))
            
            ws = self.nkn_weights
           
            w0 = LazyTensor(ws[:, 0].view(1, 1, 4))
            w1 = LazyTensor(ws[:, 1].view(1, 1, 4))
            w2 = LazyTensor(ws[:, 2].view(1, 1, 4))
            w3 = LazyTensor(ws[:, 3].view(1, 1, 4))
            w4 = LazyTensor(ws[:, 4].view(1, 1, 4))
            w5 = LazyTensor(ws[:, 5].view(1, 1, 4))
            w6 = LazyTensor(ws[:, 6].view(1, 1, 4))
            w7 = LazyTensor(ws[:, 7].view(1, 1, 4))

            combined = 0
            gated_prim0 = gate_0i * gate_0j * primitives[0] * w0
            combined = gated_prim0
            gated_prim1 = gate_1i * gate_1j * primitives[1] * w1
            combined = combined + gated_prim1
            gated_prim2 = gate_2i * gate_2j * primitives[2] * w2
            combined = combined + gated_prim2
            gated_prim3 = gate_3i * gate_3j * primitives[3] * w3
            combined = combined + gated_prim3
            gated_prim4 = gate_4i * gate_4j * primitives[4] * w4
            combined = combined + gated_prim4
            gated_prim5 = gate_5i * gate_5j * primitives[5] * w5
            combined = combined + gated_prim5
            gated_prim6 = gate_6i * gate_6j * primitives[6] * w6
            combined = combined + gated_prim6
            gated_prim7 = gate_7i * gate_7j * primitives[7] * w7
            combined = combined + gated_prim7

            node0 = combined[:, :, 0]
            node1 = combined[:, :, 1]
            node2 = combined[:, :, 2]
            node3 = combined[:, :, 3]
            
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
        hyperparams = params.get("gp_params", None)
        if hyperparams is None:
            raise ValueError("Missing Kernel hyperparameters for diagonal")
            
        N = hyperparams.gates.shape[0]
        device = hyperparams.gates.device
        ones = torch.ones(N, device=device)
        k_per_diag = ones
        k_mat12_diag = ones
        k_mat32_diag = ones
        k_mat52_diag = ones
        k_rq_diag = ones
        k_rbf_diag = ones
        inv_bw = self.inv_bandwidth.view(1, 32)
        
        lin_scaled = hyperparams.linear * inv_bw
        inner_lin = (lin_scaled * lin_scaled).sum(-1)
        k_lin_diag = inner_lin * self.linear_scale.view(1)
        
        poly_scaled = hyperparams.polynomial * inv_bw
        inner_poly = (poly_scaled * poly_scaled).sum(-1)
        k_poly_diag = (self.poly_offset.view(1) + inner_poly).square()
        primitives_diag = torch.stack([
            k_lin_diag, k_poly_diag, k_per_diag, 
            k_mat12_diag, k_mat32_diag, k_mat52_diag, k_rq_diag, k_rbf_diag
        ], dim=-1)
        g_squared = hyperparams.gates.square() # [N, 8]
        gated_primitives = g_squared * primitives_diag
        w_nkn = self.nkn_weights # [4, 8]
        nodes = torch.matmul(gated_primitives, w_nkn.t()) 
        interaction_1 = nodes[:, 0] * nodes[:, 1]
        interaction_2 = nodes[:, 2] * nodes[:, 3]
        base_diag = interaction_1 + interaction_2
        global_scale = self.outputscale.view(-1, 1)        # [e, 1]
        multitask_amp = self.latent_amplitude.view(-1, 1)  # [e, 1]
        return base_diag.unsqueeze(0) * global_scale * multitask_amp


class ProbabilisticMixtureMean(gpytorch.means.Mean):
    def __init__(self, batch_shape=torch.Size([]), **kwargs):
        super().__init__()
        self.num_experts = kwargs.get("num_experts", 8)
        self.k_atoms = kwargs.get("k_atoms", 30)
        self.register_parameter(
            name="cluster_constants", 
            parameter=torch.nn.Parameter(torch.randn(self.k_atoms, *batch_shape) * 0.1)
        )

    def forward(self, x, **params):
        target_shape = x.shape[:-1] 
        pi = params.get("pi", None)

        if pi is not None and len(target_shape) > 2:
            latent_means = pi @ self.cluster_constants
            return latent_means.movedim(-1, 0)
        else:
            return torch.zeros(target_shape, device=x.device)

class CustomLaplacePrior(Prior):
    def __init__(self, loc, scale, validate_args=False, **kwargs):
        loc_tensor = torch.as_tensor(loc, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale,  dtype=torch.float32)
        
        super(Prior, self).__init__(loc_tensor, scale_tensor, validate_args=validate_args)
        
        self._dist = Laplace(loc_tensor, scale_tensor, validate_args=validate_args)

    def log_prob(self, parameter):
        return self._dist.log_prob(parameter)

    def rsample(self, sample_shape=torch.Size()):
        return self._dist.rsample(sample_shape)
    
    @property
    def loc(self):
        return self._dist.loc

    @property
    def scale(self):
        return self._dist.scale