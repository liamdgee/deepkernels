def train_step(model, likelihood, optimizer, x_batch, y_batch):
    model.train()
    likelihood.train()
    optimizer.zero_grad()

    # --- 1. Forward Pass (The Orchestrator) ---
    # expert_outputs: Batch of GPs
    # dist_z: VAE Latent Distribution
    # pi: Mixing Weights
    # topic_recon: Linear Decoder Reconstruction (Optional)
    expert_outputs, dist_z, pi, topic_recon = model(x_batch)

    # --- 2. GP ELBO (The Primary Loss) ---
    # This maximizes P(y|x) via the mixture
    # We manually compute the Mixture Log Likelihood
    log_probs = expert_outputs.log_prob(y_batch) # [K, Batch]
    weighted_log_probs = log_probs + torch.log(pi.T + 1e-10) 
    mixture_log_lik = torch.logsumexp(weighted_log_probs, dim=0).sum()
    
    # --- 3. VAE Reconstruction (The Manifold Loss) ---
    # Does z capture x? (Uses BayesDecoder)
    z_samples = dist_z.rsample()
    loss_recon_x = model.decoder.get_reconstruction_loss(z_samples, x_batch)
    
    # --- 4. Spectral Consistency (The "Dream" Loss) ---
    # Does the Dirichlet Head's internal VAE match the GP's RFFs?
    # This keeps the mixing logic grounded in reality
    # We retrieve this from the module's internal tracking
    loss_spectral = 0
    for name, term in model.dirichlet.named_added_loss_terms():
        loss_spectral += term.loss()

    # --- 5. Latent Regularization (KL) ---
    # Keeps z smooth
    loss_kl_z = torch.distributions.kl_divergence(
        dist_z, 
        torch.distributions.Normal(0, 1)
    ).sum(-1).mean()

    # --- 6. Topic Interpretability (Optional) ---
    # Forces pi to correspond to features in x
    loss_topic = F.mse_loss(topic_recon, x_batch)

    # --- TOTAL LOSS ---
    # Weights are crucial here. Start with these defaults:
    total_loss = (
        -1.0 * mixture_log_lik +   # Maximize GP Likelihood
        1.0 * loss_recon_x +       # Minimize Reconstruction Error
        1.0 * loss_kl_z +          # Regularize Latent Space
        0.1 * loss_spectral +      # Lightly guide Spectral Head
        0.5 * loss_topic           # Moderate Topic Anchoring
    )

    total_loss.backward()
    
    # Clip Gradients (Essential for A100 stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return total_loss.item()




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