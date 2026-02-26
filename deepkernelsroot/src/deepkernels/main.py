
import yaml
from pathlib import Path

import torch

from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import torch
import logging

# --- Your Custom Imports ---
from deepkernels.models.model import StateSpaceKernelProcess
from deepkernels.train.trainer import ParameterIsolate
from deepkernels.train.langevin_trainer import LangevinTrainer
from deepkernels.preprocess.pipe import DataOrchestrator

# Optional: Set up root logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dummy_data(num_samples=1000, seq_len=30, features=16, batch_size=128):
    """Generates dummy data to test the pipeline plumbing."""
    logger.info("Generating dummy dataset...")
    
    #-x: Input sequences [Batch, SeqLen, Features]
    x_data = torch.randn(num_samples, seq_len, features)
    
    y_data = torch.randn(num_samples, seq_len, features) 
    
    ind_data = torch.arange(num_samples)
    
    dataset = TensorDataset(x_data, y_data, ind_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

def main():
    #---------------------------------------------------------
    #-- device config --#
    # ---------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Initializing pipeline on device: {device}")

    #---------------------------------------------------------
    #-- data orchestration -- #
    # ---------------------------------------------------------
    local_data_path1 = "data/lendio_all_methods_sociocorr_lender.dta"
    local_data_path2 = "data/appr_reg_data.dta"
    
    if not os.path.exists(local_data_path1) or not os.path.exists(local_data_path2):
        raise FileNotFoundError(f"Could not find one or both data files. Please check your paths!\nPath 1: {local_data_path1}\nPath 2: {local_data_path2}")
    
    df1 = pd.read_stata(local_data_path1)
    df2 = pd.read_stata(local_data_path2)

    logger.info("Running Data Orchestrator (Parallel Cleaning -> Merging -> Harmonising -> Sequencing)...")
    
    orchestrator = DataOrchestrator()

    train_loader, test_loader = orchestrator.run_pipeline(
        df1=df1,
        df2=df2,
        target_col='lmean_rejected',
        seq_len=32,
        batch_size=128
    )

    # ---------------------------------------------------------
    #-- init model -- #
    # ---------------------------------------------------------
    logger.info("Building StateSpaceKernelProcess...")
    model = StateSpaceKernelProcess().to(device)

    # ---------------------------------------------------------
    #- route params -#
    # ---------------------------------------------------------
    logger.info("Isolating parameters and building split-brain optimizers...")
    setup = ParameterIsolate(
        model=model, 
        device=device,
        base_lr_adamw=1.175e-3,
        langevin_temp=7.5e-6
    )
    
    adam_opt, sgld_opt = setup.seperate_params_and_build_optimisers()

    # ---------------------------------------------------------
    #-- init trainer --#
    # ---------------------------------------------------------
    logger.info("Initializing Langevin Trainer...")
    trainer = LangevinTrainer(
        model=model,
        adam_optimiser=adam_opt,
        sgld_optimiser=sgld_opt,
        device=device,
        total_epochs=100,
        langevin_temp=7.5e-6,
        max_grad_norm=1.0
    )

    # ---------------------------------------------------------
    #- training --#
    # ---------------------------------------------------------
    logger.info("Starting Experiment. Check MLflow dashboard for live metrics!")
    trainer.fit(train_loader, test_loader)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()