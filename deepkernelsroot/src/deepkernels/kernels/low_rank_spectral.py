import gpytorch
from linear_operator.operators import RootLinearOperator

class LowRankSpectralCovariance(gpytorch.kernels.Kernel):
    """
    Complexity: O(N * D_rff^2) or O(N) if D_rff is fixed.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, rff_features):
        return RootLinearOperator(rff_features)