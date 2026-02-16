import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm

class SpectralDecoder(nn.Module):
    def __init__(self, 
                 input_dim=30,       # Output shape (reconstruction)
                 spectral_dim=256,   # Features per cluster (M*2)
                 num_clusters=30,    # K
                 hidden_dims=None, # Flexible hidden layers
                 dropout=0.1):
        super().__init__()
        
        # e.g., 30 * 256 = 7680
        current_dim = num_clusters * spectral_dim
        hidden_dims = hidden_dims if hidden_dims is not None else [1024, 256]
        
        layers = []
        
        for dim in hidden_dims:
            layers.append(torch.nn.utils.spectral_norm(nn.Linear(current_dim, dim)))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Dropout(dropout))
            current_dim = dim
        
        layers.append(nn.Linear(current_dim, input_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, spectral_features):
        """
        Args:
            spectral_features: [Batch, K*M*2] OR [Batch, K, M*2]
        Returns:
            recon_x: [Batch, input_dim]
        """
        batch_size = spectral_features.size(0)
        # If input is [Batch, K, M*2], flatten to [Batch, K*M*2]
        if spectral_features.dim() == 3:
            features_flat = spectral_features.view(batch_size, -1)
        else:
            features_flat = spectral_features
        
        recon_x = self.network(features_flat)
        
        return recon_x