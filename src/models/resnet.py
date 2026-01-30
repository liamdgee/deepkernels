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