#filename: transformer.py

import torch
import torch.nn as nn
from src.deepkernels.models.model_config import RootConfig
import torchvision
from torchvision.models import ViT_B_16_Weights, vit_b_16
import torch.nn.functional as F
from pydantic import BaseModel, Field
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P

class TransformerConfig(BaseModel):
    latent_dim: int = Field(
        default=128, 
        description="Output dim of the feature extractor. Must match VAEConfig.input_dim"
    )
    freeze_vit: bool = Field(
        default=True, 
        description="If True, freezes the backbone but keeps the projection head learnable."
    )
    pretrained: bool = Field(
        default=True, 
        description="Load ImageNet-1k weights"
    )
    model_name: str = Field(
        default="vit_b_16",
        description="Backbone architecture name"
    )

class VisionTransformerFeatureExtractor(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.latent_dim = self.config.latent_dim

        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if self.config.pretrained else None
        self.backbone = vit_b_16(weights=weights)

        if hasattr(self.backbone, 'heads') and isinstance(self.backbone.heads, nn.Sequential):
            for module in self.backbone.heads.modules():
                if isinstance(module, nn.Linear):
                    self.in_features = module.in_features
                    break
            else:
                self.in_features = 768 #-standard fallback for vit16-#
        else:
            self.in_features=768
        
        self.backbone.heads = nn.Identity()

        if self.config.freeze_vit:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        
        #---Projection Layer for RKHS---#
        self.proj_stack = nn.Sequential(
            P.spectral_norm(nn.Linear(self.in_features, 2*self.latent_dim)),
            nn.SiLU(),
            nn.LayerNorm(2*self.latent_dim)
        )

        nn.init.orthogonal_(self.proj_stack[0].weight)

        self.kernel_head = P.spectral_norm(nn.Linear(2*self.latent_dim, self.latent_dim))

        nn.init.orthogonal_(self.kernel_head.weight)
    
    def forward(self, x, output_token=False):
        token = self.backbone(x) #---[B, 768]---#
        z = self.proj_stack(token) #---[B, latent_dim]---#
        z = self.kernel_head(z)
        z = F.normalize(z, p=2, dim=1) #-L2 norm for kernel stability-#

        if output_token:
            return z, token
        
        return z
    
    def train(self, mode=True):
        """
        Overridden to ensure the backbone stays in eval mode (no dropout)
        even when the rest of the model is training.
        """
        super().train(mode)
        if self.config.freeze_vit:
            self.backbone.eval()
        return self