
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path

repo_root = Path("~/deepkernels/deepkernelsroot").expanduser()

src_path = repo_root / "src" / "deepkernels"
src_naked_path = repo_root / "src"

if str(src_path) not in sys.path or str(src_naked_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(src_naked_path))


import numpy as np
import pandas as pd
import torch
import gpytorch
import joblib

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from typing import List, Literal
from src.deepkernels.models.model import StateSpaceKernelProcess
from app.api.routers import metrics

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path("~/deepkernels").expanduser()
KEOPS_CACHE_DIR = BASE_DIR / ".cache" / "pykeops"
ORCHESTRATOR_PATH = BASE_DIR / "features.pkl"
WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS", BASE_DIR / "princess_weights.pth")).expanduser()
KEOPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import pykeops
pykeops.config.build_folder = str(KEOPS_CACHE_DIR)
KEOPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
pykeops.config.precision = 'float64'
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False 
torch.backends.cudnn.allow_tf32 = False
# ==========================================
# 2. PYDANTIC SCHEMAS
# ==========================================
class SimulationInput(BaseModel):
    tenure_months: float = Field(24.0, ge=1, le=120)
    amount_sought: float = Field(25000.0, ge=5000)
    lender_type: Literal['fintech', 'bank', 'cdfi', 'creditunion', 'mdi', 'factoringccmca'] = 'fintech'
    animus_proxy: float = Field(3.0, ge=0.01, le=7.0)
    isolation_proxy: float = Field(3.0, ge=0.01, le=7.0)
    iat_score: float = Field(3.0, ge=0.01, le=7.0)
    horizon_steps: int = Field(32, ge=8, le=192)
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
        
        if missing:
            logger.warning(f"❓ Missing keys (might be okay if internal GP params): {len(missing)}")
        if unexpected:
            logger.info(f"📦 Ignored {len(unexpected)} extra keys (likely optimizer/meta data).")

        # 5. Finalize Model
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

app.include_router(metrics.router)

@app.get("/health")
async def health():
    return {"status": "online", "gpu": torch.cuda.is_available()}

@app.post("/v1/inference/simulate")
async def run_simulation(payload: SimulationInput):
    model = state.get("model")
    device = state.get("device")
    orch = state.get("orchestrator")

    if not model or not orch:
        raise HTTPException(status_code=503, detail="Model/Orchestrator offline.")
    
    lender_pool = ['bank', 'fintech', 'creditunion', 'cdfi', 'mdi', 'factoringccmca']
    active_lenders = lender_pool if payload.compare_all_lenders else [payload.lender_type]
    num_samples = len(active_lenders)

    try:
        batch_dfs = []
        for l_type in active_lenders:
            raw = {
                'time': [0.0],
                'log_amountsought': [np.log(payload.amount_sought + 1 + 1e-6)], 
                'ln_tenure': [np.log(payload.tenure_months + 1 + 1e-6)],
                'animus_scaled': [payload.animus_proxy],
                'isolation_scaled': [payload.isolation_proxy],
                'iat_score_f_scaled': [payload.iat_score],
                'has_masters': [1 if payload.has_masters else 0],
                'has_postgrad': [1 if payload.has_postgrad else 0],
                'is_ever_ceo': [1 if payload.is_ever_ceo else 0],
                **{col: [1 if col == l_type else 0] for col in orch.config.feature.cat_cols}
            }
            batch_dfs.append(pd.DataFrame(raw))
        full_df = pd.concat(batch_dfs)
        transformed = orch.transform_inference_input(full_df).to(device)
        time_tensor = torch.zeros(transformed.size(0), 1, dtype=torch.float64, device=device)
        transformed_30 = torch.cat([time_tensor, transformed], dim=1)
        horizon = payload.horizon_steps
        x_t = transformed_30.view(num_samples, 1, -1).expand(num_samples, horizon, -1)
        
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3), gpytorch.settings.fast_pred_var(False):
            mu_tensor, var_tensor = model.generate_trajectory(
                x_t, 
                horizon=horizon, 
                device=device
            )
            
        mu = mu_tensor.cpu().numpy()
        absolute_std = np.sqrt(var_tensor.cpu().numpy())
        
        relative_std = absolute_std / (np.abs(mu) + 1e-8)

        results = {}
        for i, lender in enumerate(active_lenders):
            results[lender] = {
                "trajectory": mu[i].tolist(),
                "std_history": relative_std[i].tolist(),
                "final_mean": float(mu[i][-1]),
                "final_std": float(relative_std[i][-1])
            }

        return results if payload.compare_all_lenders else results[payload.lender_type]
    
    except Exception as e:
        logger.error(f"GPU Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn programmatically...")
    uvicorn.run(app, host="127.0.0.1", port=8000)