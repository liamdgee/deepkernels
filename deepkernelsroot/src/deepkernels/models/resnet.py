#--Dependencies-#
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

#class: pretrained resnet-#
class ResNet(nn.Module):
    """
    A wrapper for ResNet50 that returns feature embeddings instead of class logits.
    
    Args:
        output_dim (int, optional): defaults to 2048
        freeze_grad (bool): If True, prevents gradient updates to the ResNet layers.
    """
    def __init__(self, output_dim=None, freeze_grad=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Identity() #-fc is the final linear layer-#

        if freeze_grad:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        #-optional output dim set by output dim arg-#
        self.output_dim = output_dim

        if output_dim:
            self.proj = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.Tanh(),
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, output_dim),
                nn.Tanh()
            )

            nn.init.orthogonal_(self.proj[-2].weight, gain=0.5)

            for param in self.proj.parameters():
                param.requires_grad = True #-projection is always trainable-#
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.proj(features)
        return embeddings
    
    def _spectral_map(self, x, mu_atom, log_sigma_atom, weights):
        """
        Refined for LMC: Handles broadcasting for Q latent functions.
        x: [batch, latent_dim]
        mu_atom: [Q, K, M, latent_dim]  <-- Note the Q dimension
        """
        batch_dim = x.size(0)
        num_latents = mu_atom.size(0) # Q
        
        #-Reparameterise-#
        sigma = torch.exp(log_sigma_atom)
        eps = torch.randn_like(mu_atom)
        omega = mu_atom + sigma * eps  # [Q, K, M, latent_dim]
        
        #-Reshape-#
        omega_flat = omega.view(num_latents, -1, x.size(-1))
        
        #- Project-# 
        x_expanded = x.unsqueeze(0).expand(num_latents, -1, -1)
        proj = torch.bmm(x_expanded, omega_flat.transpose(-1, -2)) #-[Q, batch, K*M]-#
        
        #-Harmonic Mapping-#
        phi_cos = torch.cos(proj) * self.scaling_constant
        phi_sin = torch.sin(proj) * self.scaling_constant

        #-Dirichlet Weighting-#
        wscl = torch.sqrt(weights + self.eps).unsqueeze(-1).repeat(1, 1, self.M_in).view(num_latents, batch_dim, -1)
        
        phi = torch.cat([phi_cos * wscl, phi_sin * wscl], dim=-1)
        return phi #- output: [Q, batch, 2*K*M]
        
