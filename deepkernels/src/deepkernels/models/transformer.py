#filenmame: transformer.py

import torch
import torch.nn as nn
from src.models.model_config import RootConfig
import torchvision
from torchvision.models import ViT_B_16_Weights, vit_b_16
import torch.nn.functional as F

class VisionTransformerFeatureExtractor(nn.Module):
    def __init__(self, config: RootConfig):
        super().__init__()
        self.config = config
        self.latent_dim = self.config.latent_dim
        pretrained = self.config.model.transformer.pretrained
        freeze_vit = self.config.model.transformer.freeze_vit
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)

        if isinstance(self.backbone.heads.head, nn.Linear):
            self.in_features = self.backbone.heads.head.in_features
        else:
            self.in_features = 768 #-fallback: possibly encode this scalar in config
        
        self.backbone.heads = nn.Identity()

        if freeze_vit:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval() #--freeze batchnorm-#
        
        
        #---Projection Layer for RKHS---#
        self.proj_stack = nn.Sequential(
            nn.Linear(self.in_features, self.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(self.latent_dim)
        )

        #---Orthogonal init for RFF Convergence---#
        nn.init.orthogonal_(self.proj_stack[0].weight)
    
    def forward(self, x, output_token=False):
        token = self.backbone(x) #---[B, 768]---#
        z = self.proj_stack(token) #---[B, latent_dim]---#
        z = F.normalize(z, p=2, dim=1) #-L2 norm for kernel stability-#
        
        if output_token:
            return z, token
        
        return z