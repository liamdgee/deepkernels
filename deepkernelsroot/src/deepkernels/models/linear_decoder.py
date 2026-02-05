import torch
import torch.nn as nn
import torch.nn.functional as F

#-where decoder_input_dim = k_atoms * M_inducing_points * 2

class BayesDecoder(nn.Module):
    def __init__(self, decoder_input_dim, features_out=1, sigma_min=1e-5):
        "Mean field variational bayes in weight space / reproducing kernel hilbert space"
        super().__init__()
        self.features_in = decoder_input_dim
        self.features_out = features_out
        self.sigma_min = sigma_min

        #-mean function out (weight vector)-#
        self.mu_fn = nn.Linear(self.features_in, self.features_out)

        #-aleatoric uncertainty (variance function) (learns noise level)-#
        self.logvar_fn = nn.Linear(self.features_in, self.features_out)

    def forward(self, kernel_features):
        """"
        args:
            kernel_features: [batch, feature_dim]
        returns:
            mu: [batch, features_out]
            var: [batch, features_out]
        """
        mu = self.mu_fn(kernel_features)

        logvar = self.logvar_fn(kernel_features)

        var = F.softplus(logvar) + self.sigma_min

        return mu, var
    
    def loss(self, mu_pred, var_pred, targets):
        """negative log likelihood"""
        loss = 0.5 * (torch.log(var_pred) + (targets - mu_pred)**2 / var_pred)
        return loss.mean()