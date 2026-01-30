import torch
import gpytorch
from gpytorch.utils.pivoted_cholesky import pivoted_cholesky

def init_inducing_pivoted_cholesky(model, train_loader, n_inducing, device):
    model.eval()
    all_features = []
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            phi = model.feature_extractor(x.to(device)) 
            all_features.append(phi[0].cpu())
            if i > 5: break 
            
    combined_phi = torch.cat(all_features, dim=0)
    
    #-comp kernel matrix with linear or rbf-
    kernel = gpytorch.kernels.RBFKernel().to('cpu')
    K = kernel(combined_phi).evaluate_kernel() 
    
    #-pivoted chol-
    indices = pivoted_cholesky(K.matrix(), max_iter=n_inducing, error_tol=1e-6)
    
    #-feature vectors out-#
    selected_centroids = combined_phi[indices] # [M, Fdim]
    
    #-Model assign-#
    Q = model.variational_strategy.num_latents
    inducing_points = selected_centroids.unsqueeze(0).repeat(Q, 1, 1)
    model.variational_strategy.base_variational_strategy.inducing_points.copy_(inducing_points)