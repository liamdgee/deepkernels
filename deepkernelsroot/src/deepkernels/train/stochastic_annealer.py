import torch
import torch.nn as nn
import gpytorch
import random
import os
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"

class StochasticAnnealer:
    def __init__(self, total_steps, n_cycles=4, ratio=0.5, start_beta=0.0, stop_beta=1.0, noise_scale=0.1):
        """
        Args:
            total_steps: Total training steps (Epochs * Batches_Per_Epoch)
            n_cycles: How many times to restart the annealing (e.g., 4)
            ratio: Fraction of the cycle spent annealing (vs. holding at 1.0)
            start_beta: Minimum KL weight (usually 0.0)
            stop_beta: Maximum KL weight (usually 1.0)
            noise_scale: The amplitude of the stochastic jitter added to beta.
        """
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.start_beta = start_beta
        self.stop_beta = stop_beta
        self.noise_scale = noise_scale

    def __call__(self, step):
        period = self.total_steps / self.n_cycles
        step_in_cycle = step % period
        cycle_progress = step_in_cycle / period
        
        if cycle_progress < self.ratio:
            #-Calculate the deterministic base progress
            rel_progress = cycle_progress / self.ratio
            base_beta = self.start_beta + (self.stop_beta - self.start_beta) * rel_progress
            
            #-Inject Stochastic Noise (Zero-mean uniform jitter)
            if self.noise_scale > 0:
                jitter = (random.random() * 2 - 1) * self.noise_scale
                beta = base_beta + jitter
            else:
                beta = base_beta
        else:
            #- Hold steady at the top of the cycle with no noise for covergence
            beta = self.stop_beta
        return max(self.start_beta, min(self.stop_beta, beta))
    
    def get_beta(self, step):
        if self.total_steps == 0: return self.stop_beta
        
        # Use float period for better precision in short epochs
        period = self.total_steps / self.n_cycles
        step_in_cycle = step % period
        
        tau = step_in_cycle / period
        if tau < self.ratio:
            return self.start_beta + (self.stop_beta - self.start_beta) * (tau / self.ratio)
        else:
            return self.stop_beta
    
    def get_noise(self, step):
        # Now it calls the method confirmed above
        beta = self.get_beta(step)
        return self.noise_scale * (1.0 - beta)
    
