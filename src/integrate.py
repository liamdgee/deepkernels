import os
import logging
import numpy as np
from scipy.stats.norm import pdf
import torch

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---Class Definition: Engine for numerical integration of analytically intractable distributions--#
class Engine:
    """Uses numerical methods to calculate KL diverrgence and Entropy through discretised integration of posterior probability distributions"""
    def __init__(self, n_steps=3001, bounds=(-10,10), eps=1e-12):
        self.bounds = bounds
        self.points = n_steps if n_steps % 2 != 0 else n_steps + 1 #--Simpsons Rule requires odd number of points--#
        self.eps = eps if eps else 1e-12
        self._update()

    def _update(self):
        self.x_ax, self.dx = np.linspace(self.bounds[0], self.bounds[1], self.points, retstep=True)
    
    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asanyarray(tensor)
    
    def integrate(self, y_as_function_values):
        """
        Simpsons 1/3 rule: (dx/3) * (y_0 + 4*sum_{y_odd} + 2*sum_{y_even} + y_n)
        As this tool integrates over function space using parabolic arcs, weights are assigned in a weight vector as follows:
        - 1 to endpoints
        - 4 to odd-index points
        - 2 to even-index points
        """
        y = np.asanyarray(y_as_function_values)
        if len(y) != self.points:
            raise ValueError(f"Input size {len(y)} does not match grid size. grid_size = {self.points}")
        
        #--init weight vector---#
        weights = np.ones(self.points)
        
        #--apply odd-index values--#
        weights[1:-1:2] = 4.0

        #--Apply even-index values--#
        weights[2:-2:2] = 2.0

        #--Compute weighted sum for discrete integration---#
        wsum = np.dot(weights, y)

        #--Simpsons integral computation--#
        integral = (self.dx / 3.0) * wsum

        return integral
    
    def get_post_density(self, mu, var, jitter=1e-7):
        """"Returns probability density curve with jitter for numerical stability"""
        mu = float(self._to_numpy(mu))
        var = float(self._to_numpy(var))
        sigma_jitter = np.sqrt(var + jitter)
        density = pdf(self.x_ax, loc=mu, scale=sigma_jitter)
        return np.maximum(density, self.eps)
    
    def compute_elementwise_kl(self, p_pdf, q_pdf):
        """Computes kl divergence using simpsons rule for precision"""
        p = self._to_numpy(p_pdf)
        q = self._to_numpy(q_pdf)
        kl = p * np.log(p / q) 
        return max(self.eps, self.integrate(kl))
    