#---Dependencies---#

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from scipy import stats
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#---Class Definition: Audit Engine to evaluate parameter relevance from Fisher Information--#
class StatisticalAuditor:
    def __init__(self, model, trainer, **kwargs):
        self.model = model
        self.trainer = trainer
        self.device = kwargs.get('device', 'mps')
        self.eps = 1e-8
    
    def get_fisher_info(self, x, local_idx):
        """Computes diagonal fisher information matrix"""
        self.model.eval()
        out = self.model(x, local_idx=local_idx)
        mu, var = out['mu'], torch.clamp(out['var'], min=self.eps)

        #--logprob score function--#
        Q = torch.distributions.Normal(mu, torch.sqrt(var))
        static_fisher_score = Q.log_prob(mu.detach()).mean()

        #--Compute Gradients of the Fisher Score on dirichlet module--#
        params = [p for n, p in self.model.named_parameters if 'dirichlet' in n]
        scores = grad(static_fisher_score, params, create_graph=True)

        #---Derive Fisher Information--#
        FIM_diag = {}
        dir_params = ((n, p) for n, p in self.model.named_parameters() if 'dirichlet' in n)
        for (n, _), grad in zip(dir_params, scores): 
            FIM_diag[n] = torch.mean(grad**2).item()
        
        return FIM_diag
    
    def get_saliency(self, x, local_idx, mu_target, var_target):
        """
        Computes Matusita Divergence in pixel space to identify visual features that trigger socio-corr/demographic flags
        mu_target and var_target will typically be 0 and 1 for neutral belief
        """

        x.requires_grad = True
        self.model.zero_grad()
        
        #---Get predictive distribution--#
        out = self.model(x, local_idx)
        mu_p, var_p = out['mu'], out['var']

        #--Comp Bhattacharyya Coefficient (rho)---#
        denom = var_p + var_target + self.eps

        spread = torch.sqrt(2 * torch.sqrt(var_p * var_target + self.eps) / denom)

        shift = torch.exp(-0.25 * (mu_p - mu_target)**2 / denom)

        bhatt_rho = spread * shift

        #---Compute Matusita Distance---#
        matusita_dist = torch.sqrt(torch.clamp(2 * (1 - bhatt_rho), min=self.eps))

        #--Backward Pass--#
        matusita_dist.mean().backward()

        #---Gradient Saliency Map---#
        saliency = torch.abs(x.grad).max(dim=1)[0]

        return saliency.detach(), matusita_dist.detach().mean().item()
    
    def render_dashboard(self, x_sample, local_idx, fair_mu, fair_sig):
        """
        Diagnostic dashboard to visualise Bhattacharyya-Matusita Saliency, Fisher sensitivty, Latent Overlap (1D)
        """
        self.model.eval()

        #---Comp Metrics--#
        saliency, mdist = self.get_saliency(x_sample, local_idx, fair_mu, fair_sig)

        fisher = self.get_fisher_info(x_sample, local_idx)

        with torch.no_grad():
            out = self.model(x_sample, local_idx)
            mu_p, var_p = out['mu'].mean().item(), out['var'].mean().item()
        
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        #--Plot A: Matusita-Bhattacharyya Saliency---#
        img = x_sample[0].detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        ax[0].imshow(img)

        sns = ax[0].imshow(saliency[0].cpu(), cmap='jet', alpha=0.45)
        ax[0].set_title(f"Matusita-Bhattacharyya Saliency: -- Matusita Distance: {mdist:.4f} -- ")
        plt.colorbar(sns, ax=ax[0], fraction=0.046, pad=0.04)

        #--Plot B: Fisher Information Sensitivity for hierarchical dirichlet process module--#
        ax[1].bar(fisher.keys(), fisher.values(), color='#000000')
        ax[1].set_title("FIM Diagonal (Global Atom Influence)")
        ax[1].set_ylabel("Fisher Information Score")
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right")

        #--Plot C: Latent Overlap between audit belief (in call of get_saliency as mu_target and var_target) and fair prior beliefs (in this function call as fair_mu & fair_sig)---#
        x_ax = np.linspace(-4, 4, 200)
        p_pdf = stats.norm.pdf(x_ax, mu_p, np.sqrt(var_p)) #--Posterior Distribution (P) across local clusters (lender_idx)---
        q_pdf = stats.norm.pdf(x_ax, fair_mu, np.sqrt(fair_sig)) #---Prior Dist (Q)---#
        ax[2].fill_between(x_ax, p_pdf, color='teal', alpha=0.32, label='Posterior Belief (P)')
        ax[2].fill_between(x_ax, q_pdf, color='orange', alpha=0.32, label='Prior (Q)')
        ax[2].plot(x_ax, p_pdf, color='teal', lw=2)
        ax[2].plot(x_ax, q_pdf, color='orange', lw=2)
        ax[2].set_title("Bhattacharyya Overlap")
        ax[2].legend()
        
        plt.tight_layout()
        plt.show()