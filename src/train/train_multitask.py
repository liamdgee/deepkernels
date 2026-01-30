def _inject_langevin_noise(self, temp):
    for g in self.opt.param_groups:
        lr = g['lr']
        noise_scale = math.sqrt(2 * lr * temp)
    
    for p in g['params']:
        if p.grad is not None:
            state = self.opt.state[p]
            G = state.get('sum_square', torch.ones_like(p.grad))
            precond = 1.0 / (torch.sqrt(G) + self.eps)
            langevin_noise = torch.randn_like(p.grad) * noise_scale * torch.sqrt(precond)
            p.grad.add_(langevin_noise)