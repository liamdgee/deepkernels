import logging
import torch
import torch.nn.functional as F
import gpytorch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from deepkernels.models.model import ShallowKernels

# --- Init logger --- #
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

# --- Dataset Definition --- #
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=1, pred_len=1, target_col_idx=None):
        """
        Args:
            data: A 2D PyTorch tensor or NumPy array of shape [Total_Timesteps, Features]
            seq_len: The number of historical timesteps to feed the model. 
                     (Set to 1 here assuming model expects 2D [batch, features])
            pred_len: The number of future timesteps to predict.
            target_col_idx: If you only want to predict a specific feature.
        """
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col_idx = target_col_idx

        self.total_windows = len(self.data) - self.seq_len - self.pred_len + 1

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # x_window shape: [seq_len, Features] -> squeezed to [Features] if seq_len=1
        x_window = self.data[idx : idx + self.seq_len].squeeze(0)

        target_start = idx + self.seq_len
        target_end = target_start + self.pred_len

        if self.target_col_idx is not None:
            y_target = self.data[target_start:target_end, self.target_col_idx]
        else:
            y_target = self.data[target_start:target_end]

        return x_window, y_target

# --- Training Loop --- #
def train_model(model, train_loader, epochs=10, device="cuda"):
    """
    Standard training loop for the variational GP and VAE components using a DataLoader.
    """
    model.train()
    model.gp.likelihood.train()
    
    opt = Adam(model.parameters(), lr=0.01)
    
    # CRITICAL: For mini-batch GP training, num_data must be the *total* dataset size, 
    # not the batch size. This scales the KL divergence term correctly.
    mll = gpytorch.mlls.VariationalELBO(
        model.gp.likelihood, model.gp, num_data=len(train_loader.dataset)
    )
    
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        total_loss = 0.0
        
        # --- The Full Circle DataLoader Loop --- #
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            
            # GPs expect 1D targets [batch_size]. If pred_len=1, y_batch is [batch_size, 1].
            # We squeeze it to ensure gpytorch doesn't throw a shape error.
            y_batch = y_batch.to(device).squeeze(-1) 
            
            opt.zero_grad()
            
            # Forward pass
            out = model(x_batch)
            
            # Calculate Losses
            recon_loss = F.mse_loss(out.state.recon, x_batch)
            gp_loss = mll(out.gp_out, y_batch) 
            loss = recon_loss - gp_loss
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
        # Log average loss across the epoch
        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

# --- Execution --- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing run on device: {device}")

    # 1. Setup Architecture
    input_dim = 30
    model = ShallowKernels(input_dim=input_dim, device=device).to(device)

    # 2. Setup Synthetic Data & DataLoader
    logger.info("Initializing Synthetic TimeSeries Dataset...")
    total_timesteps = 1000
    dummy_data = torch.randn(total_timesteps, input_dim)
    
    # Using seq_len=1 to match the 2D input expectation [batch_size, input_dim] of your VAE
    dataset = TimeSeriesDataset(
        data=dummy_data, 
        seq_len=1, 
        pred_len=1, 
        target_col_idx=0 # Predicting just the 0th column for the GP target
    )
    
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Train
    train_model(model, train_loader, epochs=10, device=device)

    # 4. Inference / Trajectory Generation
    logger.info("Testing autoregressive trajectory generation...")
    test_x = torch.randn(4, input_dim).to(device)
    
    mu, var = model.generate_trajectory(test_x, horizon=10, device=device)
    
    logger.info(f"Generation successful. Trajectory Mean Shape: {mu.shape}")
    logger.info("Public model execution finished smoothly.")

if __name__ == "__main__":
    main()