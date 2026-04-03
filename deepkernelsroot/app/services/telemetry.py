
import asyncio
from typing import Dict, List, Any
from sqlalchemy import text
import json
from sqlalchemy.ext.asyncio import AsyncConnection

# -- HARD CODED VALUE FOR TRAINING BATCH SIZE FOR DASHBOARD LATENCY OPTIMISATION --#
BATCH_SIZE = 1024

async def fetch_money_graph_data(run_id: str, db: AsyncConnection, sample_size: int = 200) -> dict:
    """Fetches raw prediction vs. truth samples for 'Money Graphs'."""
    query = text("""
        SELECT 
            step,
            MAX(CASE WHEN key = 'sample_y_true' THEN value END) as y_true,
            MAX(CASE WHEN key = 'sample_y_pred' THEN value END) as y_pred,
            MAX(CASE WHEN key = 'sample_y_std' THEN value END) as y_std
        FROM metrics
        WHERE run_uuid = :run_id
          AND key IN ('sample_y_true', 'sample_y_pred', 'sample_y_std')
        GROUP BY step
        ORDER BY step DESC
        LIMIT :limit
    """)
    result = await db.execute(query, {"run_id": run_id, "limit": sample_size})
    rows = result.fetchall()

    return {
        "run_id": run_id,
        "points": [
            {"step": r.step, "true": r.y_true, "pred": r.y_pred, "std": r.y_std} 
            for r in rows if r.y_true is not None
        ]
    }

async def fetch_model_pulse(run_id: str, db: AsyncConnection) -> dict:
    """Aggregates high-level health scores for the top-left UI Gauge."""
    query = text("""
        SELECT key, value, step 
        FROM metrics
        WHERE run_uuid = :run_id
          AND key IN ('gp_warmup_loss_total', 'val_ece', 'val_nlpd', 'system/gpu_utilization_percentage')
        ORDER BY step DESC
    """)
    result = await db.execute(query, {"run_id": run_id})
    rows = result.fetchall()
    
    latest = {}
    max_step = 0
    for r in rows:
        if r.key not in latest:
            latest[r.key] = r.value
        if r.step > max_step:
            max_step = r.step
            
    score = 100
    if latest.get('val_ece', 0) > 0.05: score -= 20
    if latest.get('val_nlpd', 0) > 2.0: score -= 15
    if latest.get('gp_warmup_loss_total', 0) > 200: score -= 10
    
    grade = "A" if score > 85 else "B" if score > 70 else "C"

    return {
        "summary": {"model_grade": grade, "health_score": score, "current_step": max_step, "status": "OPTIMIZING"},
        "performance": {
            "loss": round(latest.get('gp_warmup_loss_total', 0), 2),
            "calibration_ece": round(latest.get('val_ece', 0), 4),
            "predictive_density_nlpd": round(latest.get('val_nlpd', 0), 4)
        },
        "infrastructure": {
            "gpu_util": f"{round(latest.get('system/gpu_utilization_percentage', 0), 1)}%", 
            "compute_mode": "KeOps-NKN-Custom"
        }
    }

async def fetch_gradient_data(run_id: str, db: AsyncConnection) -> dict:
    """Fetches gradient L2 norms to check for vanishing/exploding gradients."""
    query = text("SELECT key, value FROM latest_metrics WHERE run_uuid = :run_id AND key LIKE 'grads/%'")
    result = await db.execute(query, {"run_id": run_id})
    
    # Strip the "grads/" prefix and "_norm" suffix right here in the backend
    gradients = {row[0].split('/')[-1].replace('_norm', ''): row[1] for row in result.fetchall()}
    return {"gradients": gradients}

async def fetch_calibration_history(run_id: str, db: AsyncConnection) -> dict:
    """Fetches the history of ECE and NLPD for the 3-column UI grid."""
    query = text("""
        SELECT step, key, value 
        FROM metrics 
        WHERE run_uuid = :run_id AND key IN ('val_ece', 'val_nlpd')
        ORDER BY step ASC
    """)
    result = await db.execute(query, {"run_id": run_id})
    rows = result.fetchall()
    
    ece_history = [{"step": r.step, "val": r.value} for r in rows if r.key == 'val_ece']
    nlpd_history = [{"step": r.step, "val": r.value} for r in rows if r.key == 'val_nlpd']
    
    return {
        "ece": ece_history,
        "nlpd": nlpd_history
    }

async def fetch_gp_paths(run_id: str, db: AsyncConnection) -> dict:
    """BACKEND: Fetches the raw arrays for the GP Posterior Manifold."""
    
    query = text("""
        SELECT key, value 
        FROM latest_metrics 
        WHERE run_uuid = :run_id 
          AND key IN ('gp_posterior_steps', 'gp_posterior_mean', 'gp_posterior_std', 'gp_posterior_samples')
    """)
    result = await db.execute(query, {"run_id": run_id})
    rows = result.fetchall()
    
    data_map = {row[0]: json.loads(row[1]) if isinstance(row[1], str) else row[1] for row in rows}
    
    if not data_map or 'gp_posterior_mean' not in data_map:
        return {}
        
    return {
        "steps": data_map.get("gp_posterior_steps", []),
        "mean": data_map.get("gp_posterior_mean", []),
        "std": data_map.get("gp_posterior_std", []),
        "samples": data_map.get("gp_posterior_samples", []) 
    }
# ==========================================
# 2. HARDWARE & THROUGHPUT METRICS
# ==========================================

async def fetch_hw_telemetry(run_id: str, db: AsyncConnection) -> dict:
    """Analyzes system telemetry to detect PyKeOps compilation states."""
    query = text("SELECT key, value FROM latest_metrics WHERE run_uuid = :run_id AND key LIKE 'system/%'")
    result = await db.execute(query, {"run_id": run_id})
    m = {row[0]: row[1] for row in result.fetchall()}
    
    gpu_util = m.get('system/gpu_utilization_percentage', 0)
    cpu_util = m.get('system/cpu_utilization_percentage', 0)
    vram_usage = m.get('system/gpu_memory_usage_percentage', 0)
    
    is_compiling = cpu_util > 90 and gpu_util < 10
    
    return {
        "device": "NVIDIA A100-40GB", 
        "metrics": {
            "gpu_util": f"{round(gpu_util, 1)}%",
            "vram_usage": f"{round(vram_usage, 1)}%",
            "cpu_util": f"{round(cpu_util, 1)}%"
        },
        "keops_status": "JIT_COMPILING" if is_compiling else "CACHED_EXECUTION",
        "bottleneck": "NVCC_COMPILER" if is_compiling else ("CPU_IO_BOUND" if gpu_util < 70 else "NONE"),
        "recommendation": "Increase num_workers" if not is_compiling and gpu_util < 70 else "Optimal"
    }

async def get_batch_latency_stats(run_id: str, db: AsyncConnection) -> dict:
    """Computes JIT Compilation overhead vs. Raw Execution speed."""
    query = text("""
        SELECT step, timestamp 
        FROM metrics 
        WHERE run_uuid = :run_id AND (key LIKE '%loss_gp' OR key LIKE '%gp_loss')
        ORDER BY step ASC
    """)
    result = await db.execute(query, {"run_id": run_id})
    rows = result.fetchall() # FIX: Changed from fetchone() to fetchall()
        
    if len(rows) < 2:
        return {"status": "AWAITING_DATA", "current_step": len(rows)}
        
    jit_compile_time = (rows[1][1] - rows[0][1]) / 1000.0 / 60.0 # in minutes
    
    # steady-state latency
    recent_rows = rows[-6:]
    latencies = []
    for i in range(len(recent_rows) - 1):
        diff = (recent_rows[i+1][1] - recent_rows[i][1]) / 1000.0 # in seconds
        latencies.append(diff)
    
    avg_steady_latency = sum(latencies) / len(latencies) if latencies else 0
    speedup = (jit_compile_time * 60) / (avg_steady_latency + 1e-6)

    return {
        "run_id": run_id,
        "compilation_tax_min": round(jit_compile_time, 2),
        "steady_state_latency_sec": round(avg_steady_latency, 3),
        "throughput_boost": f"{round(speedup, 1)}x Faster",
        "optimization_mode": "KeOps-CUDA-JIT",
        "status": "OPTIMIZED" if avg_steady_latency < 5.0 else "HEAVY_COMPUTE"
    }

async def fetch_dirichlet_stats(run_id: str, db: AsyncConnection) -> dict:
    """Compares Global vs Local concentration to assess kernel specialization."""
    query = text("""
        SELECT 
            step,
            MAX(CASE WHEN key = 'gp_warmup_loss_vae.dirichlet.global_divergence' THEN value END) as global_conc,
            MAX(CASE WHEN key = 'gp_warmup_loss_vae.dirichlet.local_divergence' THEN value END) as local_conc
        FROM metrics
        WHERE run_uuid = :run_id
          AND key IN ('gp_warmup_loss_vae.dirichlet.global_divergence', 'gp_warmup_loss_vae.dirichlet.local_divergence')
        GROUP BY step
        ORDER BY step DESC
        LIMIT 1;
    """)
    result = await db.execute(query, {"run_id": run_id})
    row = result.fetchone()

    if not row or row.global_conc is None or row.local_conc is None:
        return {"status": "WARMUP", "msg": "Dirichlet metrics still stabilizing."}
    
    divergence_ratio = row.local_conc / (row.global_conc + 1e-6)

    return {
        "step": row.step,
        "global_concentration": round(row.global_conc, 4),
        "local_concentration": round(row.local_conc, 4),
        "specialization_ratio": round(divergence_ratio, 2),
        "gating_mode": "Global/Universal" if divergence_ratio < 5.0 else "High-Local-Specialisation",
        "health": "STABLE" if row.global_conc > 0.05 else "VANISHING_DIVERSITY"
    }

async def fetch_nlpd_stats(run_id: str, db: AsyncConnection) -> dict:
    """Fetches smoothed NLPD stats."""
    query = text("""
        SELECT 
            step, 
            value,
            AVG(value) OVER (ORDER BY step ROWS BETWEEN :window_size PRECEDING AND CURRENT ROW) as smoothed
        FROM metrics
        WHERE run_uuid = :run_id AND key LIKE '%val_nlpd'
        ORDER BY step DESC LIMIT 1
    """)
    result = await db.execute(query, {"run_id": run_id, "window_size": 5})
    row = result.fetchone()
        
    if not row:
        return {"status": "AWAITING_EVAL", "msg": "NLPD is usually logged at the end of an epoch."}

    return {
        "step": row[0],
        "current_nlpd": round(row[1], 4),
        "smoothed_nlpd": round(row[2], 4),
        "interpretation": "Lower is better. Represents the quality of uncertainty intervals."
    }