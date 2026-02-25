class CyclicalAnnealer:
    def __init__(self, total_steps, n_cycles=4, ratio=0.5, start_beta=0.0, stop_beta=1.0):
        """
        Args:
            total_steps: Total training steps (Epochs * Batches_Per_Epoch)
            n_cycles: How many times to restart the annealing (e.g., 4)
            ratio: Fraction of the cycle spent annealing (vs. holding at 1.0)
            start_beta: Usually 0.0 or 1e-4
            stop_beta: Maximum weight (usually 1.0)
        """
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.start_beta = start_beta
        self.stop_beta = stop_beta

    def __call__(self, step):
        period = self.total_steps / self.n_cycles
        step_in_cycle = step % period
        cycle_progress = step_in_cycle / period
        
        if cycle_progress < self.ratio:
            rel_progress = cycle_progress / self.ratio
            beta = self.start_beta + (self.stop_beta - self.start_beta) * rel_progress
        else:
            beta = self.stop_beta
            
        return beta