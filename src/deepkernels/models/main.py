# tests/test_mock_pipeline.py
import logging

import torch
from torch.utils.data import DataLoader, Dataset


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
            y_target = self.data[target_start:target_end, self.target_col_idx]
        else:
            # Predict all 30 features
            y_target = self.data[target_start:target_end]

        return x_window, y_target


# Import from your actual framework
from deepkernelsroot.src.deepkernels.models.lite_model import StateSpaceKernelProcess
from tests.mockdata import MockTimeSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test():
    logger.info("Initializing Synthetic Data...")
    dataset = MockTimeSeries(num_samples=256, seq_len=32, num_targets=8)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    logger.info("Initializing DeepKernels Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateSpaceKernelProcess().to(device)

    logger.info("Testing Forward Pass...")
    try:
        xb, yb, ind = next(iter(loader))
        state, mvn, _ = model(xb.to(device), indices=ind.to(device))

        logger.info(f"Success! Output Mean Shape: {mvn.mean.shape}")
        logger.info(f"Success! Output Variance Shape: {mvn.variance.shape}")

    except Exception as e:
        logger.error(f"Architecture failed on synthetic data: {e}")
        raise


if __name__ == "__main__":
    test()
