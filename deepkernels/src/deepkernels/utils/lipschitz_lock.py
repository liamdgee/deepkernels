import torch
import torch.nn as nn

def _apply_recursive_spectral_norm(transformer):
    def apply_sn(m):
        if isinstance(m, nn.Linear):
            if not hasattr(m, 'parametrizations') or 'weight' not in m.parametrizations:
                nn.utils.spectral_norm(m)
        
        elif isinstance(m, nn.Conv2d):
            if not hasattr(m, 'parametrizations') or 'weight' not in m.parametrizations:
                nn.utils.spectral_norm(m)

    transformer.apply(apply_sn)

    return transformer