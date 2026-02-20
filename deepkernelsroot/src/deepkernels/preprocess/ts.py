import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=10, pred_len=1, target_col_idx=None):
        """
        Args:
            data: A 2D PyTorch tensor or NumPy array of shape [Total_Timesteps, Features]
            seq_len: The number of historical timesteps to feed the GRU (the look-back window).
            pred_len: The number of future timesteps to predict.
            target_col_idx: If you only want to predict a specific feature (e.g., the 0th column), 
                            pass its index. If None, the target is the full 30-feature row.
        """
        # Ensure data is a float tensor
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col_idx = target_col_idx
        
        # We can only create windows if we have enough data for the history + the future
        self.total_windows = len(self.data) - self.seq_len - self.pred_len + 1

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # 1. Slice the historical window (The input to your GRU)
        # Shape: [seq_len, Features] (e.g., [10, 30])
        x_window = self.data[idx : idx + self.seq_len]
        
        # 2. Slice the target window (What you are trying to reconstruct or predict)
        target_start = idx + self.seq_len
        target_end = target_start + self.pred_len
        
        if self.target_col_idx is not None:
            # Predict only a specific column (e.g., price, temperature)
            y_target = self.data[target_start : target_end, self.target_col_idx]
        else:
            # Predict all 30 features
            y_target = self.data[target_start : target_end]
            
        return x_window, y_target