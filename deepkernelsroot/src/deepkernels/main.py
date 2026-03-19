import os
import sys
KEOPS_CACHE_PATH = ""
if 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
    os.environ['PATH'] = f"{os.environ['CONDA_PREFIX']}/bin:{os.environ['PATH']}"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ['KEOPS_BUILD_DIR'] = KEOPS_CACHE_PATH
import yaml
from pathlib import Path
import gpytorch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
import logging
import argparse
import mlflow
import torch
import mlflow.pytorch
import logging

#---Init logger---#
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#torch.cuda.empty_cache()
#python -c "import pykeops; pykeops.clean_pykeops(); import torch; torch.cuda.empty_cache()"
import pykeops

torch.set_default_dtype(torch.float64)



# --- Internal imports ---
from deepkernels.models.model import StateSpaceKernelProcess
from deepkernels.train.langevin_trainer import LangevinTrainer
from deepkernels.preprocess.pipe import DataOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepKernels: A generative state space process")

    # ==========================================
    # --- Data & Pipeline Execution ---
    # ==========================================
    parser.add_argument("--data_path1", type=str, default="", help="Path to the primary lender data")
    parser.add_argument("--data_path2", type=str, default="", help="Path to the secondary regression data")
    parser.add_argument("--target_col", type=str, default="lmean_rejected", help="The ground truth target variable")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for the state-space model")
    parser.add_argument("--batch_size", type=int, default=2048, help="Mini-batch size for Stage 1 (VAE)")
    parser.add_argument("--test_pct", type=float, default=0.1, help="test split")
    parser.add_argument("--num_workers", type=int, default=4, help="gpu optimisation")
    parser.add_argument(
        "--drop_cols", 
        nargs='*',
        default=["lender_id", "rejected", "lmean_rejected"], 
        help="List of column names to drop before processing features (we drop y as we already have y seperate)"
    )
    
    # ==========================================
    # --- Training Stages & Epochs ---
    # ==========================================
    parser.add_argument("--warmup_vae_epochs", type=int, default=50)
    parser.add_argument("--vae_epochs", type=int, default=250)
    parser.add_argument("--warmup_gp_epochs", type=int, default=50)
    parser.add_argument("--gp_epochs", type=int, default=200)
    parser.add_argument("--em_macro_cycles", type=int, default=8)
    parser.add_argument("--joint_epochs", type=int, default=0)
    parser.add_argument("--e_epochs_per_cycle", type=int, default=3, help="E-step epochs per EM cycle")
    parser.add_argument("--m_epochs_per_cycle", type=int, default=5, help="M-step epochs per EM cycle")

    # ==========================================
    # --- Optimizers & Learning Rates ---
    # ==========================================
    parser.add_argument("--base_lr", type=float, default=1.175e-3, help="Base learning rate for AdamW")
    parser.add_argument("--fast_dir", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--med_dir", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--slow_dir", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma_dir", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--ultrasensitive_lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--sensitive_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gp_mean_lr", type=float, default=7.77e-3, help="learning rate")
    parser.add_argument("--gp_likelihood_lr", type=float, default=1.77e-2, help="learning rate")
    parser.add_argument("--gp_global_hyper_lr", type=float, default=3.5e-3, help="learning rate")
    parser.add_argument("--lmc_lr", type=float, default=8e-4, help="learning rate for lmc params in dirichlet module")
    parser.add_argument("--gp_lengthscale_lr", type=float, default=3.5e-3, help="learning rate")
    parser.add_argument("--gp_kernel_nkn_lr", type=float, default=8e-4, help="learning rate for lmc params in dirichlet module")

    # ==========================================
    # --- Langevin & Gradient Norm Limits ---
    # ==========================================
    parser.add_argument("--langevin_temp", type=float, default=7.5e-6, help="Temperature for SGLD noise injection")
    parser.add_argument("--max_grad_norm", type=float, default=1.5, help="cap for adam (non-langevin) gradients during training")
    parser.add_argument("--langevin_clip_norm", type=float, default=10.0, help="gradient norm for langevin trainer")
    
    # ==========================================
    # --- Model Dimensions & Architecture ---
    # ==========================================
    parser.add_argument("--input_dim", type=int, default=30, help="Number of input features")
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimensionality of the latent space (f)")
    parser.add_argument("--bottleneck_dim", type=int, default=64, help="Dimensionality of the VAE bottleneck")
    parser.add_argument("--k_atoms", type=int, default=30, help="Number of dictionary atoms (k)")
    parser.add_argument("--num_latents", type=int, default=8, help="Number of latent GP processes (e)")
    parser.add_argument("--alpha_factor_rank", type=int, default=3, help="Rank for the Dirichlet alpha factor matrix")
    parser.add_argument(
        "--disentangle_split_override", 
        nargs='+', 
        type=int, 
        default=[4, 30, 30], 
        help="List of dimensions to split the disentangled bottleneck representation"
    )

    # ==========================================
    # --- Spectral, Fourier & NKN Settings ---
    # ==========================================
    parser.add_argument("--num_fourier_features", type=int, default=128, help="Number of random Fourier features (M)")
    parser.add_argument("--spectral_emb_dim", type=int, default=2048, help="Dimensionality of the spectral embedding")
    parser.add_argument(
        "--spectral_compressions", 
        nargs='+', 
        type=int, 
        default=[1024, 512, 128, 64], 
        help="List of dimensions for spectral compression layers in decoder"
    )
    parser.add_argument("--kernels_out", type=int, default=32, help="Number of output features from the hypernetwork")
    parser.add_argument("--individual_kernel_input_dim", type=int, default=32, help="Dimensionality of the input to the primitive kernels")
    parser.add_argument("--individual_kernel_dim_out", type=int, default=32, help="nkn kernel out")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of kernel experts/primitives in the NKN")
    parser.add_argument("--num_primitives", type=int, default=5, help="kernel primitives (note that matern is used for 3 and rq is used for 2, so 5 -> 8 in keops kernel)")

    # ==========================================
    # --- Annealer & Penalty Params ---
    # ==========================================
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weighting for the KL divergence terms (Beta-VAE)")
    parser.add_argument("--iw_stop_beta", type=float, default=0.1, help="Max weight for Inverse-Wishart")
    parser.add_argument("--kl_stop_beta", type=float, default=0.1, help="Max weight for VAE divergences")
    parser.add_argument("--kl_cycles", type=int, default=4, help="Number of cyclical annealing cycles")
    parser.add_argument("--kl_weights", nargs='*', help="Custom weighting for individual KL terms")

    # ==========================================
    # --- Numerical Stability & Bounds ---
    # ==========================================
    
    # --- Variational/Dirichlet Bounds ---
    parser.add_argument("--sigma_lower_bound", type=float, default=1e-4, help="Lower bound clamp for latent sigma")
    parser.add_argument("--sigma_upper_bound", type=float, default=5.0, help="Upper bound clamp for latent sigma")
    parser.add_argument("--mu_lower_bound", type=float, default=-17.0, help="Lower bound clamp for latent mu")
    parser.add_argument("--mu_upper_bound", type=float, default=17.0, help="Upper bound clamp for latent mu")
    
    # --- Clipping & Scaling ---
    parser.add_argument("--eps_clip", type=float, default=2.7, help="Gradient/Value clipping threshold")
    parser.add_argument("--psi_scale", type=float, default=1.0, help="Scaling factor for digamma/psi functions")
    
    # --- Extreme Numerical Tolerances ---
    parser.add_argument("--stick_breaking_epsilon", type=float, default=3e-3, help="Epsilon for Kumaraswamy/Dirichlet stick-breaking")
    parser.add_argument("--uniform_dist_clamp", type=float, default=5e-5, help="Clamp for uniform distribution sampling")
    parser.add_argument("--tiny_eps", type=float, default=3e-8, help="Absolute minimum epsilon to prevent log(0) and div/0")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout probability")
    parser.add_argument("--jitter", type=float, default=1e-6, help="General numerical stability jitter")
    parser.add_argument("--cholesky_jitter", type=float, default=1e-3, help="for use in langevin trainer")
    parser.add_argument("--eps", type=float, default=1e-3, help="General numerical stability epsilon")
    parser.add_argument("--eps_dirichlet", type=float, default=1e-3, help="Epsilon for Dirichlet prior/likelihood stability")
    parser.add_argument("--large_eps", type=float, default=4e-2, help="Larger epsilon for specific numerical bounds")
    parser.add_argument("--posterior_dirichlet_epsilon", type=float, default=4e-5, help="Prevents values from falling off the simplex")
    parser.add_argument("--min_ls", type=float, default=0.05, help="Minimum bounded lengthscale for the GP")
    parser.add_argument("--max_ls", type=float, default=15.0, help="Maximum bounded lengthscale for the GP")
    parser.add_argument("--conc_clamp", type=float, default=30.0, help="Upper clamp limit for Dirichlet concentration")
    parser.add_argument("--gamma_concentration_init", type=float, default=2.5, help="Initial value for Gamma concentration")
    parser.add_argument("--min_noise", type=float, default=1e-4, help="Lower bound constraint for GP likelihood noise")

    return parser.parse_args(), parser

def main():
    #---------------------------------------------------------
    #-- device config --#
    # ---------------------------------------------------------
    args, _ = parse_args()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(6) # Helps with parallelizing non-kernel ops
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"
    # ---------------------------------------------
    os.environ['KEOPS_CHROOT'] = "./.cache/keops"
    device = torch.device('cuda') or torch.device('cuda:0')
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
    
    x_tensor, y_tensor = orchestrator.run_pipeline(
        df1=df1,
        df2=df2,
        target_col=args.target_col,
        drop_cols=args.drop_cols
    )

    seq_x, seq_y = orchestrator.to_seq_data(        
        x_tensor,
        y_tensor, 
        seq_len=args.seq_len
    )

    train_loader, val_loader, test_loader = orchestrator.prepare_data(
            seq_x, seq_y, seq_len=args.seq_len, val_pct=0.1, test_pct=0.1, 
            batch_size=args.batch_size, num_workers=4
        )
    
    N = len(train_loader.dataset)
    args.n_data = N
    logger.info(f"Dataset loaded. Total training samples (N) = {N}")

    # ---------------------------------------------------------
    #-- init Variational GP & model -- #
    # ---------------------------------------------------------
    logger.info("Building StateSpaceKernelProcess...")
    
    model_keys = ['n_data', "input_dim"]
    model_keys_with_seq_len = ['n_data', "input_dim", "seq_len"]

    dynamics = {k: v for k, v in vars(args).items() if k in model_keys}
    trainer_dynamics = {k: v for k, v in vars(args).items() if k in model_keys_with_seq_len}
    
    pykeops.clean_pykeops()
    pykeops.set_build_folder(KEOPS_CACHE_PATH)
    print(f"Ready for Joint Training. Using cache at: {pykeops.get_build_folder()}")

    model = StateSpaceKernelProcess(device=device, **dynamics)

    model_path = "/home/liam/deepkernels/deepkernelsroot/mlruns/2/e0512de7583146329a010afc984a25a5/artifacts/model_checkpoints/best_val_model.pth"
    
    model.to(device)
    
    model.train()
    
    trainer = LangevinTrainer(
        model=model,
        device=device,
        **trainer_dynamics
    )

    if os.path.exists(model_path):
        logger.info(f"Attempting to load partial checkpoint from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            old_state_dict = checkpoint['model_state_dict']
            
            filtered_state_dict = {
                k: v for k, v in old_state_dict.items() 
                if "vae.dirichlet.kernel_network" not in k
            }
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            logger.info(f"[+] Successfully loaded VAE and base GP weights.")
            logger.info(f"[+] Initialized fresh weights for {len(missing_keys)} NKN parameters.")
            logger.warning("[!] Architecture changed: Using FRESH optimizers to prevent momentum shape crashes.")
        except Exception as e:
            logger.error(f"Failed to load partial checkpoint: {e}. Starting entirely fresh.")
    else:
        logger.info("No checkpoint found. Starting fresh training run.")
    # ---------------------------------------------------------
    # -- MLflow Setup & Execution --
    # ---------------------------------------------------------
    
    mlflow.set_tracking_uri("//mlflow.db")
    
    experiment_name = "deepkernels"
    mlflow.set_experiment(experiment_name)

    logger.info("Starting Experiment. Check MLflow dashboard for live metrics!")
    
    try:
        trainer.fit(
            train_loader=train_loader,
            test_loader=val_loader,
            joint_training=False
        )
    except Exception as e:
        logger.error(f"Training interrupted by error: {e}")
        raise
    finally:
        logger.info("Performing safety-save of model weights...")
        final_model_path = "final_model_weights_emergency_save.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved emergency weights to {final_model_path}")
    logger.info("Training session concluded successfully.")

if __name__ == "__main__":
    main()