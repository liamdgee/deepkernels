#---Dependencies---#
import os
import logging
import torch
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import math
import torch.nn as nn
import functools
import mlflow

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#--Tracking Function Decorator using mlflow--#
def tracker(kernel_experiment):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(kernel_experiment)
            with mlflow.start_run() as run:
                mlflow.log_params(kwargs)
                result = fn(*args, **kwargs)
                mlflow.set_tag("train_dict", fn.__name__)
                return result
        return wrapper
    return decorator

#---Class Definition: Stochastic Gradient Optimiser with Adaptive Langevin Dynamics--#
@tracker(kernel_experiment="Dirichlet_Mixture_Proj")

def train_model(
            self,
            model,
            dataloader, 
            criterion, 
            epochs=200, 
            device='cuda'
        ):
        
        self.model = model.to(device) #-where model is an orchestration module-#
        vae_model = vae_model.to(device)
        gp_model = gp_model.to(device)
        criterion = criterion.to(device)
        
        vae_model.train()
        gp_model.train()
        criterion.likelihood.train()

        
        optimiser = optim.AdamW([
            {'params': vae_model.parameters(), 'lr': 2e-3, 'weight_decay': 1.2e-4},
            {'params': gp_model.parameters(), 'lr': 0.015},
            {'params': criterion.likelihood.parameters(), 'lr': 0.015}
        ])

        # 3. Setup Stochastic Annealers (from our previous step)
        total_steps = epochs * len(dataloader)
        annealers = {
            "dirichlet_global_kl": StochasticAnnealer(total_steps, n_cycles=4, stop_beta=0.1, noise_scale=0.01),
            "dirichlet_local_kl": StochasticAnnealer(total_steps, n_cycles=4, stop_beta=0.1, noise_scale=0.01),
            "lengthscale_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0),
            "alpha_kl": StochasticAnnealer(total_steps, n_cycles=1, ratio=0.2, stop_beta=1.0, noise_scale=0.0)
        }

        global_step = 0

        # --- THE EPOCH LOOP ---
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, x_batch in enumerate(loop):
                x_batch = x_batch.to(device) # Shape: [Batch, SeqLen, Features]
                optimiser.zero_grad()
                
                # CRITICAL: Clear the VAE's internal loss dictionary from the previous batch!
                # (Assuming your BaseGenerativeModel has a way to clear these. If it's a list/dict, clear it here)
                if hasattr(model, 'added_loss_terms'):
                    model.added_loss_terms.clear()

                # Step the annealers and update the criterion weights
                current_kl_weights = {
                    name: annealer(global_step) for name, annealer in annealers.items()
                }
                criterion.kl_weights = current_kl_weights

                # --- 1. VAE FORWARD PASS ---
                ss_out = model(x_batch)
                history = ss_out.history

                # --- 2. GP FORWARD PASS ---
                # Extract the bottleneck features to feed the GP
                # Assuming features_per_expert is [Batch, 8, N, 16] or similar
                gp_input = history.bottlenecks  
                
                # Address the shape trap (if N is missing, e.g., [Batch, 8, 16] -> [Batch, 8, 1, 16])
                if gp_input.dim() == 3:
                    gp_input = gp_input.unsqueeze(-2)

                # Package the dynamic hyperparameters from the VAE to the GP
                gp_kwargs = {
                    "gp_params": {
                        "ls_rbf": history.expert_params[..., 0], # Map these to your actual dictionary keys!
                        "w_sm": history.expert_params[..., 1],
                        # ... etc ...
                        "gates": history.gate_weights
                    },
                    "mixture_means_per_expert": history.expert_mixtures,
                    "pi": history.pis
                }

                # Run the KeOps LMC GP!
                gp_output = gp_model(gp_input, **gp_kwargs)

                # --- 3. LOSS COMPUTATION ---
                # Define what the GP is trying to predict (Target). 
                # If it's predicting the expert features themselves, pass them here.
                gp_target = history.expert_params # <-- Update to your specific target tensor
                
                loss, metrics = criterion(
                    model=model,
                    x_target=x_batch,
                    ss_history=history,
                    gp_output=gp_output,
                    gp_target=gp_target
                )

                # --- 4. BACKWARD PASS & OPTIMIZE ---
                loss.backward()
                
                # Optional but highly recommended for RNNs/VAEs: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                global_step += 1

                # --- 5. LOGGING ---
                # Update the progress bar with the most critical metrics
                loop.set_postfix(
                    Loss=f"{metrics['loss_total']:.2f}", 
                    Recon=f"{metrics['loss_recon']:.2f}",
                    GP=f"{metrics['loss_gp']:.2f}",
                    DirKL=f"{metrics.get('loss_dirichlet_global_kl', 0.0):.2f}"
                )
                
                # (Optional) Log to Weights & Biases here:
                # wandb.log(metrics, step=global_step)

        return model, gp_model