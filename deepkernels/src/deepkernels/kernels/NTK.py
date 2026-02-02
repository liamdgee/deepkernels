import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import Kernel
from torch.func import functional_call, vmap, jacrev

class SparseNTK(Kernel):
    """
    Sparse & Projected Neural Tangent Kernel (NTK).

    approximates empirical infinite-width NTK by:
    1. Computing gradients w.r.t. a subset of parameters (parameter sparsity).
    2. Projecting these gradients into a lower-dimensional space (sketching).
    
    k(x, x') ~ < J(x) @ Omega, J(x') @ Omega >
    """
    def __init__(self, 
                 deep_network, 
                 proj_dim=512, 
                 target_layer_names=None, 
                 freeze_projection=True,
                 **kwargs):
        """
        Args:
            deep_network: nn.Module
            projection_dim (int): The dimension D to project gradients down to.
            target_layer_names (list of str): Substrings of layer names to compute gradients for. 
                                              If None, uses ALL parameters (Slow!).
                                              Example for ViT: ["heads", "encoder.layers.11"]
        """
        super().__init__(**kwargs)
        self.model = deep_network
        self.proj_dim = proj_dim
        
        #--param selecttion--#
        self.param_names = []
        self.tracked_params = {}
        self.untouched_params = {}

        for name, p in self.model.named_parameters():
            if target_layer_names is None or any(t in name for t in target_layer_names):
                self.tracked_params[name] = p
                self.param_names.append(name)
            else:
                self.untouched_params[name] = p
                
        self.buffers = dict(self.model.named_buffers())

         #-calculate activate param count-#
        self.n_active_params = sum(p.numel() for p in self.tracked_params.values())
        if self.n_active_params == 0:
            raise ValueError(f"No parameters found matching {target_layer_names}. -- Check layer names")

        #- initialise random matrix-#
        self.init_proj_mat = torch.randn(self.n_active_params, self.proj_dim) / (proj_dim ** 0.5)

        #-Integration-#
        if freeze_projection:
            #- Random kernel projection: [n_active_params, proj_dim]-#
            #-Johnson-Lindenstrauss lemma-#
            self.register_buffer("projection_matrix", self.init_proj_mat, persistent=True)
        else:
            #-trained linear map for optimiser-#
            self.register_parameter("projection_matrix", nn.Parameter(self.init_proj_mat))
        
        self.freeze_projection = freeze_projection

    def _functional_forward(self, active_params_dict, x_batch):
        """
        Combines active gradients with fixed params.
        """
        full_params = {**self.untouched_params, **active_params_dict}
        return functional_call(self.model, (full_params, self.buffers), x_batch)

    def get_phi(self, x):
        """
        Computes feature map Phi(x).
        Phi(x) = Gradient(x) @ projection_matrix
        """
        jac_fn = jacrev(self._functional_forward, argnums=0)
        batch_jac_fn = vmap(jac_fn, in_dims=(None, 0))
        jdict = batch_jac_fn(self.tracked_params, x) #-dict of gradients per param tensor-#
        #-flatten jacobians -> [batch, n_active_params]-#
        jflat = []
        for name in self.param_names:
            grad = jdict[name] #-- flattens all dims except batch (dim 0)--#
            grad = grad.reshape(grad.shape[0], -1)
            jflat.append(grad) #--scalar outputs: [B, P] ; vector outputs: [B, out, P]
            
        jmat = torch.cat(jflat, dim=1) #-shape: [N, P] @ [P, D] -> [N, D]-#
        phi = torch.matmul(jmat, self.projection_matrix)
        
        return phi

    def forward(self, x1, x2, diag=False, **params):
        phi1 = self.get_phi(x1)
        
        if self.training and torch.equal(x1, x2):
            phi2 = phi1
        else:
            phi2 = self.get_phi(x2)
        if diag:
            return (phi1 * phi2).sum(dim=-1) #-linear kernel on proj-#
        else:
            return torch.matmul(phi1, phi2.transpose(-1, -2))