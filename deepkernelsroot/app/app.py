
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
ORCHESTRATOR_PATH = BASE_DIR / "optimised_features.pkl"
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
        state["orchestrator"] = orch
        logger.info("✅ Feature Orchestrator Loaded.")
        
        model_categories = set(state["orchestrator"].config.feature.cat_cols)
        schema_categories = set(SimulationInput.model_fields['lender_type'].annotation.__args__)
        if schema_categories - model_categories:
            logger.warning(f"⚠️ WARNING: Schema allows lenders not in training data: {schema_categories - model_categories}")
        
        logger.info(f"⏳ Loading SleepyPrincess weights from: {WEIGHTS_PATH.name}...")
        input_dim = len(orch.config.feature.num_cols) + len(orch.config.feature.cat_cols)
        model = StateSpaceKernelProcess(input_dim=input_dim, n_data=87636.0, device=device) 
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.to(device).eval()
        state["model"] = model
        state["device"] = device
        logger.info(f"✅ SleepyPrincessv1.0 Weights Loaded onto {device}.")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load inference assets: {e}")
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

@app.post("/api/v1/inference/simulate")
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
        horizon = payload.horizon_steps
        baseline_X = transformed.view(num_samples, 1, -1).expand(num_samples, horizon, -1)
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3), gpytorch.settings.fast_pred_var(False):
            dream_state, _, _ = model.forward(
                baseline_X, vae_out=None, steps=horizon, features_only=True, generative_mode=True
            )
            _, mvn, _ = model.forward(
                baseline_X, dream_state, steps=0, features_only=False
            )
            
            mu = mvn.mean.cpu().numpy()

            std = np.sqrt(mvn.variance.cpu().numpy())
            
        if payload.compare_all_lenders:
            return {
                lender: {
                    "trajectory": mu[i].tolist(),
                    "std_history": std[i].tolist(),
                    "final_mean": float(mu[i][-1]),
                    "final_std": float(std[i][-1])
                } for i, lender in enumerate(active_lenders)
            }
        else:
            return {
                "trajectory": mu[0].tolist(),
                "std_history": std[0].tolist(),
                "final_mean": float(mu[0][-1]),
                "final_std": float(std[0][-1])
            }
            
    except Exception as e:
        print(f"GPU Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))