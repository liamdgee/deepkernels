import torch
from gpytorch.kernels import Kernel
from linear_operator.operators import LowRankRootLinearOperator

class LinearKernelON(Kernel):
    """
    A strict O(N) Linear Kernel for use on explicit nn feature map 'z'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_outputscale", 
            parameter=torch.nn.Parameter(torch.tensor(0.0))
        )

    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.raw_outputscale)

    def forward(self, x1, x2, diag=False, **params):
        #-scale by output features-#
        sigma = torch.sqrt(self.outputscale)
        z1 = x1 * sigma
        if x1 is not x2:
            z2 = x2 * sigma
        else:
            z2 = z1
        if diag:
            return (z1 * z2).sum(-1)
            
        return LowRankRootLinearOperator(z1, z2)