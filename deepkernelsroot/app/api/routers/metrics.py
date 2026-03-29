#-dependencies-#
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncConnection
from app.core.database import get_db
from app.services.telemetry import (
    fetch_money_graph_data,
    fetch_model_pulse,
    fetch_gradient_data,
    fetch_calibration_history,
    fetch_hw_telemetry,
    get_batch_latency_stats,
    fetch_dirichlet_stats,
    fetch_nlpd_stats,
    fetch_gp_paths
)

#-router-#
router = APIRouter(
    prefix="/api/v1/metrics",
    tags=["Real-time Telemetry"]
)

# ==========================================
# 1. THE DASHBOARD ENDPOINTS (Required by Dash)
# ==========================================

@router.get("/money-stats/{run_id}")
async def money_stats_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    """Serves the main trajectory vs ground truth graph."""
    try:
        return await fetch_money_graph_data(run_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pulse-check/{run_id}")
async def pulse_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)): 
    """Serves the top-left Health Gauge."""
    try:
        return await fetch_model_pulse(run_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/posterior-paths/{run_id}")
async def simulation_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)): 
    """Serves the top-left Health Gauge."""
    try:
        return await fetch_gp_paths(run_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gradient-flow/{run_id}")
async def gradient_flow_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    """Serves the horizontal gradient magnitude bar chart."""
    try:
        return await fetch_gradient_data(run_id, db) # Fixed name mismatch here
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/calibration-stats/{run_id}")
async def calibration_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    """Serves BOTH the ECE and NLPD line charts in one call to save bandwidth."""
    try:
        return await fetch_calibration_history(run_id, db) # Maps perfectly to UI now
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. HARDWARE & DIAGNOSTIC ENDPOINTS
# ==========================================

@router.get("/efficiency/{run_id}")
async def gpu_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    try:
        return await fetch_hw_telemetry(run_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch-latency/{run_id}")
async def latency_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    try:
        return await get_batch_latency_stats(run_id, db) # Fixed name mismatch here
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dirichlet-dynamics/{run_id}")
async def dirichlet_convergence_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    try:
        return await fetch_dirichlet_stats(run_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlpd-smoothed/{run_id}")
async def nlpd_smoothed_endpoint(run_id: str, db: AsyncConnection = Depends(get_db)):
    """Returns the single smoothed NLPD metric, separate from the history array."""
    try:
        return await fetch_nlpd_stats(run_id, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))