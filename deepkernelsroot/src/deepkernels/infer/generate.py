import torch
import gpytorch
import torch.nn as nn

def generate_synthetic_trajectories(pipeline, num_samples=5, seq_length=100):
    pipeline.eval()
    
    with torch.no_grad():
        z_synthetic = torch.randn(num_samples, 16)
        t_grid = torch.linspace(0, 1.0, seq_length).unsqueeze(0).expand(num_samples, -1)
        t_grid = t_grid.unsqueeze(-1) #-shape must match keops-#
        
        #-Generate the KeOps GP Parameters from the latent and feed z straight to the Dirichlet module
        dirichlet_out = pipeline.dirichlet(x=z_synthetic, vae_out=None)
        gp_params = dirichlet_out['gp_params']
        pi = dirichlet_out['pi']
        gp_params['w_sm'] = pi.view(pi.size(0), 1, 1, -1)
        gp_params['ls_rbf'] = dirichlet_out['predicted_lengthscale'].view(-1, 1, 1, 1)
        gp_params['mu_sm'] = dirichlet_out['frequencies'].mean(dim=(2, 3)).view(pi.size(0), 1, 1, -1)
        generative_dist = pipeline.gp_model(
            x=t_grid, 
            gp_params=gp_params, 
            pi=pi
        )
        
        #- rsample() draws a mathematically continuous path from the learned covariance matrix
        synthetic_paths = generative_dist.rsample()
        
    return t_grid.squeeze(-1), synthetic_paths

# --- Execution ---
# time_steps, generated_data = generate_synthetic_trajectories(pipeline)