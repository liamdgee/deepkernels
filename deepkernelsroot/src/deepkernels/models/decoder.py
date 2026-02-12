import torch
import torch.nn as nn

class SpectralDecoder(nn.Module):
    def __init__(self, rff_dim, output_dim):
        super().__init__()
        self.recon = nn.Linear(rff_dim, output_dim)
        
    def forward(self, spectral_features):
        """
        Args:
            spectral_features: The output from dirichlet.dynamic_random_fourier_features
                               Shape: [Batch, K * M * 2]
        """
        recon = self.recon(spectral_features)
        return recon