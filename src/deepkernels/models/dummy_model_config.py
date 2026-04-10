#--Model Config Module--#

from typing import Optional

from pydantic import BaseModel, Field


class TransformerConfig(BaseModel):
    latent_dim: int = Field(128, description="Latent dimensionality of the transformer")
    freeze_vit: bool = True
    pretrained: bool = True

class DirichletConfig(BaseModel):
    alpha: float = Field(0.75, gt=0, description="Local concentration parameter")
    gamma: float = Field(2.0, gt=0, description="Global concentration parameter")
    sigma_noise: float = Field(0.0002, ge=0)
    n_global: int = Field(30, gt=0)
    n_local: int = Field(3, gt=0)
    learnable_params: bool = True
    fourier_dim: int = Field(512, gt=0, le = 4096, description="RFF dimensionality")
    

class GPConfig(BaseModel):
    num_inducing: int = Field(256, gt=0, description="Inducing points for Woodbury inversion")
    fourier_dim: int = Field(512, gt=0, le = 4096, description="RFF dimensionality")
    latent_dim: int = Field(128, description="Latent dimensionality of the GP")
    bottleneck_dim: int = Field(1024, gt=0, le = 8192, description="compression factor for spectral info") #--for SNVGP arch--#

class RKHSConfig(BaseModel):
    n_anchors: int = Field(256, gt=0)
    transformer_out_dim: int = Field(768, gt=0)
    latent_dim: int = Field(128, description="Latent dimensionality of the reproducing kernel / manifold")

class RFFConfig(BaseModel):
    fourier_dim: int = Field(512, gt=0, le=4096, description="RFF dimensionality")

class SpectralConfig(BaseModel):
    n_mixtures: int = Field(1024, description="placeholder")

class ModelConfig(BaseModel):
    latent_dim: int = Field(128, description="Latent dimensionality of model")
    transformer: TransformerConfig
    dirichlet: DirichletConfig
    gp: GPConfig
    rkhs: RKHSConfig

class RootConfig(BaseModel):
    model: ModelConfig
    eps: float = 1e-9
    latent_dim: int = Field(128, le=1024, ge=16, description="Latent dim of model")
