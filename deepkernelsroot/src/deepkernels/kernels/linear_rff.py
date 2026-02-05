import torch
import math
import gpytorch
from gpytorch.kernels import Kernel
from linear_operator.operators import LowRankRootLinearOperator

class OrthogonalFourierKernel(Kernel):
    """
    A custom kernel that approximates an RBF kernel using Random Fourier Features.
    
    Complexity:
        Storage: O(N * num_samples) instead of O(N^2)
        Compute: O(N) for matrix-vector multiplications
    
    Args:
        num_samples (int): The rank of the approximation (R)
        input_dim (int): The dimension of the input data (D)
    """
    def __init__(self, n_samples, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.input_dim = input_dim

        #-- Qmat is an orthogonal matrix of weights "Q"-#
        Qmat = self._generate_orthogonal_weights(rank= self.n_samples, dim=self.input_dim)

        #-freeze weights and biases and draw random features-#
        self.register_buffer("weights", Qmat)
        self.register_buffer("bias", torch.rand(self.n_samples) * 2 * math.pi)
        
        #-learnable lengthscale-#
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(torch.zeros(1, 1, 1)))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())
        self.lengthscale_initialised = False
    
    def _generate_orthogonal_weights(self, rank, dim):
        assert rank >= dim, "Orthogonal Random Features require rank to be greater than input dimension (Yu et al., 2016)"
        nblocks = int(rank/dim) #-where nblocks = number of full orthogonal blocks-#
        blocks=[]
        #-Uses QR decomposition (matrix factorisation) to generate an orthogonal matrix 'Q'
        for _ in range(nblocks):
            Q, _ = torch.linalg.qr(torch.randn(dim, dim))
            blocks.append(Q)
        
        feats_remaining = rank % dim
        if feats_remaining > 0:
            Q, _ = torch.linalg.qr(torch.randn(dim, dim))
            blocks.append(Q[:, :feats_remaining])

        orth_weights = torch.cat(blocks, dim=1)
        
        #-scaled by chi dist to match a vectorised gaussian-#
        dist = torch.distributions.Chi2(torch.tensor([float(dim)], device=orth_weights.device))
        
        sample = dist.rsample((rank,)).squeeze()
        
        norms = torch.sqrt(sample)

        orth_weights = orth_weights * norms.unsqueeze(0)

        return orth_weights
    
    def _init_lengthscale(self, train_x):
        """
        Sets the initial lengthscale with total variance proxy
        """
        if train_x is None:
            return
        
        if train_x.dim() == 3:
            # Shape: [Batch * N, D]
            flat_x = train_x.reshape(-1, train_x.size(-1))
        else:
            flat_x = train_x
        
        var_per_dim = flat_x.var(dim=0).detach()
        avg_squared_dist = 2 * var_per_dim.sum()
        median_dist_proxy = torch.sqrt(avg_squared_dist)
        
        #-edge case=#
        if median_dist_proxy == 0:
            median_dist_proxy = torch.tensor(1.0, device=train_x.device)

        #-inverse softplus-#
        raw_val = torch.log(torch.expm1(median_dist_proxy))
        self.raw_lengthscale.data.fill_(raw_val.item())
    
    def _init_lengthscale_expensive(self, train_x):
        """
        Sets the initial lengthscale to the median distance between data points
        """
        if train_x is None:
            return
        
        subset = train_x[:1000].detach()
        dists = torch.cdist(subset, subset)
        upper_tri_mask = torch.triu_indices(dists.size(0), dists.size(1), offset=1)
        valid_dists = dists[upper_tri_mask[0], upper_tri_mask[1]]
        if valid_dists.numel() > 0:
            median_dist = valid_dists.median()
        else:
            median_dist = torch.tensor(1.0, device=dists.device)
    
        raw_val = torch.log(torch.expm1(median_dist)) #-inverse softplus-#
    
        self.raw_lengthscale.data.fill_(raw_val.item())


    def forward(self, x1, x2, diag=False, **params):
        if self.training and not self.is_initialized:
            self._init_lengthscale(x1)
            self.lengthscale_initialized = True
        
        lengthscale = self.lengthscale
        
        x1_scl = x1.div(lengthscale) #-[Batch, N, D]
        
        #-fourier proj-#
        z1 = torch.matmul(x1_scl, self.weights)
        z1 = z1.add(self.bias)
        z1 = torch.cos(z1)
        z1 = z1.mul(math.sqrt(2.0 / self.n_samples))
        if x1 is not x2:
            x2_scl = x2.div(lengthscale)
            z2 = torch.matmul(x2_scl, self.weights)
            z2 = z2.add(self.bias)
            z2 = torch.cos(z2)
            z2 = z2.mul(math.sqrt(2.0 / self.n_samples))
        else:
            z2 = z1
        
        if diag:
            return (z1 * z2).sum(-1)
        
        return LowRankRootLinearOperator(z1, z2)