import torch
import gpytorch
from torch.distributions import Normal, TransformedDistribution, kl_divergence, Dirichlet
from torch.distributions.transforms import StickBreakingTransform
from gpytorch.mlls import AddedLossTerm
import math
from src.models.model_config import RootConfig
import torch.nn.functional as F

class KLDivergence(AddedLossTerm):
    """loss term for hierarchical dirichlet process -- KL(q(rho)|p(rho))"""
    def __init__(self, qdist, pdist):
        self.Q = qdist
        self.P = pdist
    def loss(self):
        return torch.distributions.kl_divergence(self.Q, self.P).sum()

from pydantic import BaseModel

class HDPConfig(BaseModel):
    K: int = 30
    M: int = 512
    D: int = 128
    eps: float = 1e-4
    gamma_prior: float = 2.0


class VariationalDirichlet(gpytorch.Module):
    def __init__(self, config:HDPConfig):
        super().__init__()
        self.config = config
        self.K = self.config.K
        self.M = self.config.M #-rff samples per atom-#
        self.D = self.config.D #-input dim to vit-#
        self.gamma_prior = self.config.gamma_prior

        self.gamma = nn.Parameter(torch.tensor(float(self.gamma_prior)))

        #--variational params-#
        #--q(logits) ~ N(mu, sig)

        self.register_parameter(
            "q_mu", 
            nn.Parameter(torch.randn(self.K - 1))
        )
        self.register_parameter(
            "q_log_sigma", 
            nn.Parameter(torch.full((self.K - 1,), -2.0))
        )

        #--prior params--#
        #---uniform on simplex ~ N(0, 1)--#
        self.register_buffer("prior_mu", torch.zeros(self.K - 1))
        self.register_buffer("prior_log_sigma", torch.zeros(self.K - 1))

        #--Spectral Frequencies-#
        self.log_lengthscale_atom = nn.Parameter(torch.randn(self.K, 1, self.D) * 0.5 - 2.0)

        #--fixed RFF phases and noise buffers-#
        #-draw standard normal once and freeze (fundamental random fourier projection assumption)--#
        #-Shape: [K, M, D]
        self.register_buffer("noise_weights", torch.randn(self.K, self.M, self.D))
        self.register_buffer("noise_bias", torch.rand(self.K, self.M) * 2 * math.pi)

        self.stick_break_transform = StickBreakingTransform()
    
    def forward(self, batch_shape=torch.Size([])) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #--variational posterior - q(v)-#
        q_sig = F.softplus(self.q_log_sigma)
        q_dist = Normal(self.q_mu, q_sig)

        #-prior p(v)-#
        p_sig = F.softplus(self.prior_log_sigma)
        p_dist = Normal(self.prior_mu, p_sig)

        #register loss-#
        self.update_added_loss_term("hdp_kl", KLDivergence(q_dist, p_dist))

        #--Sample global weights-#
        #-q(beta) = StickBreaking( q(v) )

        dist = TransformedDistribution(q_dist, [self.stick_break_transform])
        beta = dist.rsample() #-[K]-

        #-sample local weights-#
        #- pi | beta ~ Dir(gamma * beta)

        #--Define Concentration -- gamma strictly positive-#
        gamma = F.softplus(self.gamma)
        concentration = (gamma * beta) + self.eps
        concentration = torch.clamp(concentration, self.eps, 500)

        #-if batch_shape is (), conc: [K]. if batch shape is 128, conc: [128, K]
        if len(batch_shape) > 0:
            concentration = concentration.unsqueeze(0).expand(*batch_shape, -1)
        
        local_dist = torch.distributions.Dirichlet(concentration)
        pi = local_dist.rsample() #-[B, K]-#


        #--Construct spectal frequencies-#
        atomic_scale = self.log_lengthscale_atom.exp() #-[K, 1, D]
        omega = self.noise_weights * atomic_scale #[K, M, D]


        print(f"Type of beta: {type(beta)}")
        print(f"Type of omega: {type(omega)}")
        #--Returns:
        #pi: [B, K] -> local mixing weights
        #beta: [K] -> global prevalence
        #omega: [K, M, D] -> spectral frequencies
        #bias: [K, M] -> spectral phases
        return pi, beta, omega, self.noise_bias