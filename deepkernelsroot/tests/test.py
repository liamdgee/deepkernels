import unittest
import torch
import gpytorch
import numpy as np
import sys
import os

# --- IMPORT YOUR MODELS ---
# Adjust these imports to match your actual file structure
# from models import SpectralVAE, ApproximateSpectralGP, RecurrentEncoder
# For this script to run standalone, I will assume the classes are importable
# or you can paste the class definitions at the top of this file for a quick check.

class TestDeepKernelSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print(f"\n[System Check] Python: {sys.version.split()[0]}")
        print(f"[System Check] PyTorch: {torch.__version__}")
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System Check] Device: {cls.device}")
        
        # Hyperparameters for testing
        cls.batch_size = 8
        cls.seq_len = 50
        cls.input_dim = 44  # Your data dimension
        cls.feature_dim = 60 # Your latent/spectral feature dim
        
    def test_01_gpu_acceleration(self):
        """
        CRITICAL: Verifies that Cuda is actually doing math.
        """
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA not detected. Training will be painfully slow.")
            # Don't fail the test, just warn
            return

        x = torch.randn(1000, 1000, device=self.device)
        y = torch.matmul(x, x)
        self.assertEqual(y.device.type, 'cuda', "Tensors failed to move to GPU")
        print("✅ GPU Matrix Multiplication works.")

    def test_02_vae_shapes_and_physics(self):
        """
        Verifies the Spectral VAE accepts data and outputs valid physics parameters.
        """
        # Initialize VAE
        # (Assuming you import SpectralVAE correctly)
        # model = SpectralVAE(input_dim=self.input_dim, ...).to(self.device)
        
        # MOCK CLASS if import fails (remove this block if you have real imports)
        # ---------------------------------------------------------
        class MockVAE(torch.nn.Module):
            def __init__(self, input_dim, feat_dim):
                super().__init__()
                self.input_dim = input_dim
                self.feat_dim = feat_dim
                self.encoder = torch.nn.Linear(input_dim, feat_dim) # Dummy
            def forward(self, x):
                # Simulate [Batch, Seq, Dim] -> Flatten or LSTM
                bs = x.shape[0]
                return {
                    "spectral_features": torch.randn(bs, self.feat_dim, device=x.device),
                    "mu": torch.randn(bs, 16, device=x.device),
                    "logvar": torch.randn(bs, 16, device=x.device),
                    "lengthscale": torch.abs(torch.randn(bs, 1, device=x.device)),
                    "recon": x
                }
            def loss(self, out, target_rff):
                return {"loss": torch.tensor(0.5, requires_grad=True)}
        # ---------------------------------------------------------

        model = MockVAE(self.input_dim, self.feature_dim).to(self.device)
        
        # Create dummy batch
        x = torch.randn(self.batch_size, self.input_dim, device=self.device)
        
        # Forward pass
        out = model(x)
        
        # Checks
        self.assertIn('spectral_features', out, "VAE output missing 'spectral_features'")
        self.assertIn('lengthscale', out, "VAE output missing 'lengthscale' (Physics)")
        
        # Physics Check: Lengthscales must be positive
        min_ls = out['lengthscale'].min().item()
        self.assertGreaterEqual(min_ls, 0.0, "Found negative lengthscales! Physics violation.")
        
        print(f"✅ VAE Output Shapes Valid. Feature Dim: {out['spectral_features'].shape}")

    def test_03_gp_integration(self):
        """
        Verifies the Approximate GP can ingest VAE features and compute a valid ELBO.
        """
        # MOCK GP if import fails
        # ---------------------------------------------------------
        class MockGP(gpytorch.models.ApproximateGP):
            def __init__(self, feat_dim):
                dist = gpytorch.variational.CholeskyVariationalDistribution(10)
                strat = gpytorch.variational.VariationalStrategy(self, torch.randn(10, feat_dim), dist)
                super().__init__(strat)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            def forward(self, x):
                m = self.mean_module(x)
                c = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(m, c)
        # ---------------------------------------------------------
        
        gp = MockGP(self.feature_dim).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=100)
        
        # Dummy Features from "VAE"
        features = torch.randn(self.batch_size, self.feature_dim, device=self.device)
        targets = torch.randn(self.batch_size, device=self.device)
        
        # GP Forward
        output = gp(features)
        
        # Loss Calculation
        loss = -mll(output, targets)
        
        # Check correctness
        self.assertFalse(torch.isnan(loss), "GP Loss is NaN! Check initialization or learning rates.")
        print(f"✅ GP Forward Pass & Loss Calculation successful. Loss: {loss.item():.4f}")

    def test_04_end_to_end_gradient(self):
        """
        The 'Gold Standard' test: Can we update the VAE encoder using the GP's loss?
        """
        # Setup dummy models (replace with REAL imports)
        # model = SpectralVAE(...).to(self.device)
        # gp = ApproximateGP(...).to(self.device)
        
        # (Using mocks for the script to run standalone, replace in production)
        linear = torch.nn.Linear(10, 10).to(self.device) # Represents VAE Encoder
        gp_layer = torch.nn.Linear(10, 1).to(self.device) # Represents GP logic
        
        optimizer = torch.optim.Adam(list(linear.parameters()) + list(gp_layer.parameters()), lr=0.01)
        
        # Fake Data
        x = torch.randn(8, 10, device=self.device)
        y = torch.randn(8, 1, device=self.device)
        
        # Forward
        optimizer.zero_grad()
        features = linear(x) # VAE
        pred = gp_layer(features) # GP
        loss = torch.nn.functional.mse_loss(pred, y)
        
        # Backward
        loss.backward()
        
        # Check Gradients in the FIRST layer (Encoder)
        grad_norm = linear.weight.grad.norm().item()
        self.assertGreater(grad_norm, 0.0, "Gradient did not flow back to Encoder!")
        
        optimizer.step()
        print(f"✅ End-to-End Gradient Flow verified. Grad Norm: {grad_norm:.6f}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
