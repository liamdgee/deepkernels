import torch
from torch.optim import Optimizer

class SGHMC(Optimizer):
    """
    Stochastic Gradient Hamiltonian Monte Carlo Optimizer (MCMC transition Kernel)
    """
    def __init__(self, params, lr=1e-2, momentum_decay=0.05, num_burn_in=1000):
        defaults = dict(lr=lr, alpha=momentum_decay, burn_in=num_burn_in)
        super().__init__(params, defaults)
        self.steps = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.randn_like(p) #-random init-#
                #-hamiltonian hyperparams-#
                lr = group['lr']
                alpha = group['alpha']
                dp = p.grad.data #-add nll grad term-#
                #-- Hamiltonian dynamics: momentum_t+1 = momentum_t - lr * grad - momentum_decay * momentum_t + noise
                momentum = state['momentum']
                momentum.mul_(1.0 - alpha) #-decay (friction term)-
                momentum.add_(dp, alpha=-lr) #-gradient force-#
                
                # Thermal Noise (Injection)
                # inject thermal noise -- noise ~ N(0, 2 * alpha * lr) for stationarity dist assumptions-#
                sigma = torch.sqrt(torch.tensor(2.0 * alpha * lr))
                noise = torch.randn_like(p) * sigma
                momentum.add_(noise)
                p.data.add_(momentum) #-update position in parameter space-#
        self.steps += 1
        
        return loss