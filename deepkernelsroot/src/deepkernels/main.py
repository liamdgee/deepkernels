
import yaml
from pathlib import Path

import torch

from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import torch
import logging
import argparse

# --- Internal imports ---
from deepkernels.models.model import StateSpaceKernelProcess
from deepkernels.train.trainer import ParameterIsolate
from deepkernels.train.langevin_trainer import LangevinTrainer
from deepkernels.preprocess.pipe import DataOrchestrator
from deepkernels.models.exactgp import Simple
from deepkernels.models.variationalautoencoder import SpectralVAE
from deepkernels.train.exact_objective import ExactObjective

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepKernels: Two-Stage Spectral VAE & ExactGP Trainer")
    
    # Data Paths
    parser.add_argument("--data_path1", type=str, default="/home/liam/deepkernels/data/lendio_all_methods_sociocorr_lender.dta", help="Path to the primary lender data")
    parser.add_argument("--data_path2", type=str, default="/home/liam/deepkernels/data/appr_reg_data.dta", help="Path to the secondary regression data")
    
    # Pipeline Dimensions
    parser.add_argument("--target_col", type=str, default="lmean_rejected", help="The ground truth target variable")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for the state-space model")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size for Stage 1 (VAE)")
    parser.add_argument("--test_pct", type=float, default=0.2, help="test split")
    parser.add_argument("--num_workers", type=int, default=4, help="gpu optimisation")

    # Inside parse_args()
    parser.add_argument(
        "--drop_cols", 
        nargs='*',
        default=["lender_id", "rejected", "lmean_rejected"], 
        help="List of column names to drop before processing features (we drop y as we already have y seperate)"
    )

    parser.add_argument(
        "--kl_weights",
        nargs='*'
    )
    
    #-model tweaks
    parser.add_argument("--run_gp", type=bool, default=False, help="boolean to skip complex keops math")
    
    # Optimization Hyperparameters
    parser.add_argument("--vae_epochs", type=int, default=100, help="Number of epochs to train the VAE (Stage 1)")
    parser.add_argument("--gp_epochs", type=int, default=100, help="Number of epochs to train the ExactGP (Stage 2)")
    parser.add_argument("--base_lr_adamw", type=float, default=1.175e-3, help="Base learning rate for AdamW")
    parser.add_argument("--langevin_temp", type=float, default=7.5e-6, help="Temperature for SGLD noise injection")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Base learning rate for AdamW")

    #-niche learning rates-#
    parser.add_argument("--fast_dir", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--med_dir", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--slow_dir", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma_dir", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--ultrasensitive_lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--sensitive_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gp_mean_lr", type=float, default=7.77e-3, help="learning rate")
    parser.add_argument("--gp_likeliehood_lr", type=float, default=1.77e-2, help="learning rate")
    parser.add_argument("--gp_hyper_lr", type=float, default=3.5e-3, help="learning rate")

    return parser.parse_args()

def main():
    #---------------------------------------------------------
    #-- device config --#
    # ---------------------------------------------------------
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Initializing pipeline on device: {device}")

    #---------------------------------------------------------
    #-- data orchestration -- #
    # ---------------------------------------------------------
    if not os.path.exists(args.data_path1) or not os.path.exists(args.data_path2):
        raise FileNotFoundError(f"Missing data files! Check paths:\n1: {args.data_path1}\n2: {args.data_path2}")
    
    df1 = pd.read_stata(args.data_path1)
    df2 = pd.read_stata(args.data_path2)

    logger.info("Running Data Orchestrator (Parallel Cleaning -> Merging -> Harmonising -> Sequencing)...")
    
    orchestrator = DataOrchestrator()

    train_loader, test_loader = orchestrator.run_pipeline(
        df1=df1,
        df2=df2,
        target_col=args.target_col,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        drop_cols=args.drop_cols,
        test_pct=args.test_pct,
        num_workers=args.num_workers
    )

    # ---------------------------------------------------------
    #-- Extract Full Dataset for ExactGP-#
    # ---------------------------------------------------------
    logger.info("Extracting full dataset into memory for KeOps ExactGP...")
    all_x, all_y = [], []
    for batch_idx, (x, y) in enumerate(train_loader):
        all_x.append(x)
        all_y.append(y)
        
    full_x = torch.cat(all_x, dim=0) #- [N, SeqLen, Features]
    full_y = torch.cat(all_y, dim=0) #-[N, 30]
    
    N = full_x.size(0)
    if full_y.dim() == 2 and full_y.size(1) == 30:
        full_y = full_y.t().contiguous() # for lmc: shape == [30, N]
    
    logger.info(f"Full dataset extracted. N={N}. Target shape: {full_y.shape}")

    # ---------------------------------------------------------
    #-- init ExactGP & model -- #
    # ---------------------------------------------------------
    logger.info("Building KeOps ExactGP and StateSpaceKernelProcess...")
    dummy_x = torch.arange(N, dtype=torch.float32)
    

    gp = Simple(
        train_x=dummy_x, 
        train_y=full_y, 
        likelihood=None,
        num_latents=8
    )
    #-could argparse atoms and latents obviously, but this will break the entire model
    
    model = StateSpaceKernelProcess(gp=gp, run_gp=args.run_gp)

    trainer = LangevinTrainer(
        model=model,
        device=device,
        total_epochs=(args.vae_epochs + args.gp_epochs),
        base_lr_adamw=args.base_lr,
        langevin_temp=args.langevin_temp,
        max_grad_norm=args.max_grad_norm,
        n_data=N 
    )

    # ---------------------------------------------------------
    #- route params -#
    # ---------------------------------------------------------
    logger.info("Isolating parameters and building split-brain optimizers...")
    
    objective = ExactObjective(model=model, kl_weights=args.kl_weights)

    setup = ParameterIsolate(
        model=model, 
        device=device,
        objective=objective,
        base_lr_adamw=args.base_lr_adamw,
        langevin_temp=args.langevin_temp,
        fast_dir=args.fast_dir
        med_dir=args.med_dir
        slow_dir=args.slow_dir
        gamma_dir=args.gamma_dir
        ultrasensitive_lr=args.ultrasensitive_lr,
        sensitive_lr=args.sensitive_lr,
        gp_mean_lr=args.gp_mean_lr,
        gp_likelihood_lr=args.gp_likelihood_lr,
        gp_hyper_lr=args.gp_hyper_lr
    )
    
    adam_w_opt, langevin_opt, adam_opt, groups = setup.seperate_params_and_build_optimisers()

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