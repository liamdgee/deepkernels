import os
from pathlib import Path
import sys

# Force single-threading for CPU operations to prevent container thread-locking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

import math
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import gpytorch
import joblib
import pykeops
from scipy.ndimage import gaussian_filter1d
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

#-key imports-#

from src.deepkernels.models.model import StateSpaceKernelProcess
# ==========================================
# LOGGING & PATHS
# ==========================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Docker-safe relative paths
BASE_DIR = Path(os.getenv("APP_BASE_DIR", "/app"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
ORCHESTRATOR_PATH = Path(os.getenv("ORCHESTRATOR_PATH", ARTIFACTS_DIR / "features.pkl"))
WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS", ARTIFACTS_DIR / "princess_weights.pth"))

# ==========================================
# TORCH & KEOPS CONFIGURATION
# ==========================================
KEOPS_CACHE_DIR = Path(os.getenv("PYKEOPS_CACHE_DIR", BASE_DIR / ".cache" / "pykeops"))
KEOPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
pykeops.config.build_folder = str(KEOPS_CACHE_DIR)
pykeops.config.precision = 'float64'
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False 
torch.backends.cudnn.allow_tf32 = False

MEAN_ANIMUS, STD_ANIMUS = 2.26, 2.82
MEAN_IAT, STD_IAT = -1.31, 4.05
MEAN_ISO, STD_ISO = 0.16, 1.1147
MEAN_TENURE, STD_TENURE = 2.0 * 12, 1.15 * 12
SOUGHT_MEAN, SOUGHT_SD = 967333.086, 545944.978
MINTEN, MAXTEN = 12.0, (MEAN_TENURE+1.65*STD_TENURE)
MINASO, MAXASO = 50000.0, (SOUGHT_MEAN+1.65*SOUGHT_SD)
MINAN, MAXAN = (MEAN_ANIMUS-1.65*STD_ANIMUS), (MEAN_ANIMUS+1.65*STD_ANIMUS)
MINISO, MAXISO = (MEAN_ISO-1.65*STD_ISO), (MEAN_ISO+1.65*STD_ISO)
MINIAT, MAXIAT = (MEAN_IAT-1.65*STD_IAT), (MEAN_IAT+1.65*STD_IAT)
BUFFER = 1.035
EPS = 0.1
DAMPING = 0.575

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class SimulationInput(BaseModel):
    tenure_months: float = Field(MEAN_TENURE, ge=MINTEN-0.5, le=MAXTEN+0.5)
    amount_sought: float = Field(SOUGHT_MEAN, ge=MINASO/BUFFER, le=MAXASO*BUFFER)
    lender_type: Literal['fintech', 'bank', 'cdfi', 'creditunion', 'mdi', 'factoringccmca'] = 'fintech'
    animus_proxy: float = Field(MEAN_ANIMUS, ge=MINAN-EPS, le=MAXAN+EPS)
    isolation_proxy: float = Field(MEAN_ISO, ge=MINISO-EPS, le=MAXISO+EPS)
    iat_score: float = Field(MEAN_IAT, ge=MINIAT-EPS, le=MAXIAT+EPS)
    horizon_steps: int = Field(32, ge=16, le=128)
    has_masters: bool = False
    has_postgrad: bool = False
    is_ever_ceo: bool = False
    compare_all_lenders: bool = False


class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    means: List[List[float]]
    variances: List[List[float]]


state = {}

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Booting DeepKernels Inference Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        orch = joblib.load(ORCHESTRATOR_PATH)
        if not hasattr(orch, 'feature_transformer'):
            old_name = next((name for name in ['transformer', 'feature_eng', 'novelty'] if hasattr(orch, name)), None)
            if old_name:
                logger.info(f"🩹 Patching Orchestrator: Aliasing '{old_name}' -> 'feature_transformer'")
                orch.feature_transformer = getattr(orch, old_name)
            else:
                logger.error("❌ ERROR: Could not find any transformer inside the pickle!")
        state["orchestrator"] = orch
        logger.info("✅ Feature Orchestrator Loaded.")
        
        model_categories = set(state["orchestrator"].config.feature.cat_cols)
        schema_categories = set(SimulationInput.model_fields['lender_type'].annotation.__args__)
        if schema_categories - model_categories:
            logger.warning(f"⚠️ WARNING: Schema allows lenders not in training data: {schema_categories - model_categories}")
        
        input_dim = len(orch.config.feature.num_cols) + len(orch.config.feature.cat_cols) + 1
        model = StateSpaceKernelProcess(input_dim=input_dim, n_data=87636.0, device=device)

        logger.info(f"⏳ Loading weights from: {WEIGHTS_PATH.name}...")
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        raw_state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        
        first_key = list(raw_state_dict.keys())[0]
        
        if first_key.startswith('model.'):
            logger.info("🧹 Stripping 'model.' prefix from state_dict keys...")
            clean_state_dict = {k.replace('model.', ''): v for k, v in raw_state_dict.items()}
        elif first_key.startswith('_forward_module.'):
            logger.info("🧹 Stripping '_forward_module.' prefix...")
            clean_state_dict = {k.replace('_forward_module.', ''): v for k, v in raw_state_dict.items()}
        else:
            clean_state_dict = raw_state_dict

        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        model.gp.likelihood.noise_covar.register_constraint(
            "raw_noise", 
            gpytorch.constraints.Interval(1e-5, 0.005)
        )
        model.gp.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.Interval(0.01, 0.5))
        model.gp.covar_module.register_constraint("raw_inv_bandwidth", gpytorch.constraints.Interval(0.01, 0.4))
        if missing:
            logger.warning(f"❓ Missing keys (might be okay if internal GP params): {len(missing)}")
        if unexpected:
            logger.info(f"📦 Ignored {len(unexpected)} extra keys (likely optimizer/meta data).")
        model.to(device).eval()
        state["model"] = model
        state["device"] = device
        logger.info(f"✅ SleepyPrincessv1.0 Ready on {device}.")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL BOOT FAILURE: {e}")
        raise e
    
    yield

    logger.info("🛑 Shutting down DeepKernels Engine...")
    state.clear()

app = FastAPI(title="deepkernelsAPI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Static Telemetry Routes (No SQL needed!) ---

@app.get("/v1/metrics/pulse-check/{run_id}")
async def pulse_check(run_id: str): # Added run_id to match the URL pattern
    return {
        "status": "online",
        "device": str(state.get("device", "cpu")),
        "vram_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    }

@app.get("/v1/metrics/efficiency/{run_id}")
async def efficiency_check(run_id: str):
    return {
        "gpu_util": 85.2, 
        "latency_ms": 120
    }

@app.get("/health")
async def health():
    return {"status": "online", "gpu": torch.cuda.is_available()}

def smooth_data(data: list, std_history: list = None) -> list:
    if not data: return data
    
    # 0.85 is the "Sweet Spot" for UI responsiveness
    base_beta = 0.85
    
    if std_history:
        avg_uncertainty = np.mean(std_history)
        beta = np.clip(base_beta + (avg_uncertainty * 0.07), 0.8, 0.9)
    else:
        beta = base_beta

    smoothed = [data[0]]
    for i in range(1, len(data)):
        # EMA Formula: S_t = (1 - beta) * X_t + beta * S_{t-1}
        smoothed_val = (smoothed[-1] * beta) + (data[i] * (1 - beta))
        smoothed.append(smoothed_val)
    return smoothed

def to_list_safe(data):
    """Ensures data is a list, regardless of whether it's a Tensor, Ndarray, or List."""
    if isinstance(data, list):
        return data
    if hasattr(data, 'tolist'): # Catch-all for NumPy and Torch
        return data.tolist()
    return list(data) # Final fallback
def smooth_vectorized(data, std_data, b_beta=0.82):
    """Applies EMA across the last axis of a 2D or 3D array."""
    v_smoothed = np.zeros_like(data)
    v_smoothed[..., 0] = data[..., 0]
    
    # avg_u and beta are shape (6,)
    avg_u = np.mean(std_data, axis=-1) 
    beta = np.clip(b_beta + (avg_u * 0.1), 0.77, 0.97)
    
    # ONLY add an axis if data is 3D (e.g., if you ever smooth the 10 samples)
    # Because a 3D slice at time 't' is (6, 10), so beta needs to be (6, 1)
    if data.ndim == 3:
        beta = beta[:, np.newaxis]

    for t in range(1, data.shape[-1]):
        # 2D case: (6,) * (6,) = (6,) -> Perfectly fits!
        v_smoothed[..., t] = (v_smoothed[..., t-1] * beta) + (data[..., t] * (1 - beta))
        
    return v_smoothed

@app.post("/v1/inference/simulate")
async def run_simulation(payload: SimulationInput):
    results = {}
    
    model = state.get("model")
    device = state.get("device")
    orch = state.get("orchestrator")

    if not model or not orch:
        raise HTTPException(status_code=503, detail="Model/Orchestrator offline.")
    
    # 2. Dynamic Pool Management
    lender_pool = ['bank', 'fintech', 'creditunion', 'cdfi', 'mdi', 'factoringccmca']
    
    # Ensure the requested lender is actually valid
    if not payload.compare_all_lenders and payload.lender_type not in lender_pool:
        raise HTTPException(status_code=400, detail=f"Invalid lender: {payload.lender_type}")

    active_lenders = lender_pool if payload.compare_all_lenders else [payload.lender_type]
    num_samples = len(active_lenders)

    try:
        batch_dfs = []
        for l_type in active_lenders:
            raw = {
                'time': [0.0],
                'log_amountsought': [np.log((payload.amount_sought)+BUFFER)], 
                'ln_tenure': [np.log((payload.tenure_months/12.0)+BUFFER)],
                'animus_scaled': [payload.animus_proxy],
                'isolation_scaled': [payload.isolation_proxy],
                'iat_score_f_scaled': [payload.iat_score],
                'has_masters': [1 if getattr(payload, 'has_masters', False) else 0],
                'has_postgrad': [1 if getattr(payload, 'has_postgrad', False) else 0],
                'is_ever_ceo': [1 if getattr(payload, 'is_ever_ceo', False) else 0],
                **{col: [1 if col == l_type else 0] for col in orch.config.feature.cat_cols}
            }
            batch_dfs.append(pd.DataFrame(raw))
        full_df = pd.concat(batch_dfs)
        # --- 2. DETERMINISTIC SEEDING (Now fully reactive) ---
        seed_str = f"{payload.amount_sought}-{payload.tenure_months}-{payload.lender_type}-{payload.animus_proxy}-{payload.isolation_proxy}-{payload.iat_score}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        transformed = orch.transform_inference_input(full_df).to(device)
        
        # 3. Proper Tensor Shaping for GPyTorch LMC
        time_tensor = torch.zeros(transformed.size(0), 1, dtype=transformed.dtype, device=device)
        transformed_30 = torch.cat([time_tensor, transformed], dim=1)
        
        horizon = payload.horizon_steps
        # Reshape to [Batch, Seq, Features]
        x_t = transformed_30.view(num_samples, 1, -1).expand(num_samples, horizon, -1)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(1e-6):
            mu_tensor, var_tensor = model.generate_trajectory(x_t, horizon=horizon, device=device)
        

        results = {}
        # 1. GPU Math: Scaling and clamping on CUDA to avoid CPU overhead
        with torch.no_grad():
            mu_tensor = mu_tensor * math.sqrt(0.125)
            var_clamped = torch.clamp(var_tensor * 0.125, min=1e-7, max=0.28)
            std_tensor = torch.sqrt(var_clamped)

        # 2. Single Move: Copy to CPU in one batch
        mu_np = mu_tensor.cpu().numpy()
        std_np = std_tensor.cpu().numpy()

        # 3. Vectorized Noise: Broadcast samples across all lenders/paths
        epsilon = np.random.normal(0, 1, size=(num_samples, 10, horizon))
        epsilon_smooth = gaussian_filter1d(epsilon, sigma=3.0, axis=-1)
        epsilon_smooth /= (np.std(epsilon_smooth, axis=-1, keepdims=True) + 1e-9)
        
        # [num, 1, horiz] * [num, 10, horiz] -> vectorized path generation
        samples_np = mu_np[:, np.newaxis, :] + (epsilon_smooth * std_np[:, np.newaxis, :] * DAMPING)
        mu_smooth_batch = smooth_vectorized(mu_np, std_np)
        # --- CRITICAL FIX: Match the naming for the loop below ---
        for i, lender in enumerate(active_lenders):
            # 1. Secure the baseline
            mu0 = float(mu_np[i][0])
            mu0_safe = mu0 if abs(mu0) > 1e-4 else 1.0
            
            # 2. Extract the NumPy slices for this specific lender
            raw_smooth_mu = mu_smooth_batch[i]
            raw_std = std_np[i]
            raw_samples = samples_np[i]

            # 3. Perform the math using fast NumPy operations
            rel_mu_smooth = (raw_smooth_mu / mu0_safe) - 1.0
            rel_std = raw_std / abs(mu0_safe)
            rel_samples = (raw_samples / mu0_safe) - 1.0

            # 4. Safely convert to lists using your robust fallback function
            l_mu_smooth = to_list_safe(raw_smooth_mu)
            l_rel_mu_smooth = to_list_safe(rel_mu_smooth)
            l_std = to_list_safe(rel_std)
            l_abs_std = to_list_safe(raw_std)
            l_samples = to_list_safe(rel_samples)

            # 5. Assemble the results
            results[lender] = {
                "trajectory": l_mu_smooth,             # Smoothed absolute trajectory
                "relative_trajectory": l_rel_mu_smooth,# Smoothed relative trajectory
                "std_history": l_std,                  # Relative standard deviation
                "absolute_std": l_abs_std,             # Raw absolute standard deviation
                "samples": l_samples,                  # Relative sample paths
                "final_mean": float(l_rel_mu_smooth[-1]),
                "final_std": float(l_std[-1])
            }
            
        if payload.compare_all_lenders:
            return results
        return {payload.lender_type: results.get(payload.lender_type, {})}
    
    except Exception as e:
        logger.error(f"❌ GPU Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Engine Failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn programmatically...")
    uvicorn.run(app, host="127.0.0.1", port=8000)