// public/static/script.js

// Keep track of our live-refresh timer
let liveRefreshTimer;

// --- 1. The Hacker Console Logger ---
function addLog(message, type = "INFO") {
    const consoleDiv = document.getElementById('log-console');
    if (!consoleDiv) return;

    const time = new Date().toLocaleTimeString();
    
    let color = "#3fb950"; // Default Green (INFO)
    if (type === "WARN") color = "#d29922"; // Yellow
    if (type === "ERROR") color = "#f85149"; // Red
    if (type === "SYSTEM") color = "#58a6ff"; // Blue

    const logEntry = document.createElement('div');
    logEntry.style.color = color;
    logEntry.innerHTML = `[${time}] [${type}] ${message}`;
    
    consoleDiv.appendChild(logEntry);
    consoleDiv.scrollTop = consoleDiv.scrollHeight;

    // Keep only the last 50 logs to prevent memory lag
    if (consoleDiv.childNodes.length > 50) {
        consoleDiv.removeChild(consoleDiv.firstChild);
    }
}

// --- 2. The Money Graph (GP Uncertainty) ---
async function drawMoneyGraph(runId) {
    try {
        const response = await fetch(`/api/v1/metrics/money-stats/${runId}`);
        const data = await response.json();

        if (!data.points || data.points.length === 0) return;

        const steps = data.points.map((p, i) => i);
        const y_true = data.points.map(p => p.true);
        const y_pred = data.points.map(p => p.pred);
        const y_std = data.points.map(p => p.std);

        const truthTrace = {
            x: steps, y: y_true,
            type: 'scatter', mode: 'markers', name: 'Ground Truth',
            marker: { color: '#ffffff', size: 4 }
        };

        const meanTrace = {
            x: steps, y: y_pred,
            type: 'scatter', name: 'GP Mean Prediction',
            line: { color: '#58a6ff', width: 3 }
        };

        const uncertaintyTrace = {
            x: [...steps, ...steps.slice().reverse()],
            y: [
                ...y_pred.map((v, i) => v + (1.96 * y_std[i])),
                ...y_pred.map((v, i) => v - (1.96 * y_std[i])).reverse()
            ],
            fill: 'toself',
            fillcolor: 'rgba(88, 166, 255, 0.2)', // Adjusted to match your blue aesthetic
            line: { color: 'transparent' },
            name: '95% Confidence Interval',
            type: 'scatter'
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e0e6ed' },
            margin: { t: 30, r: 20, b: 30, l: 40 },
            showlegend: true
        };

        Plotly.react('money-graph', [uncertaintyTrace, meanTrace, truthTrace], layout);
    } catch (err) {
        addLog("Failed to update Money Graph", "ERROR");
    }
}

// --- 3. The KL Divergence Graph ---
async function drawKLGraph(runId) {
    try {
        const response = await fetch(`/api/v1/metrics/kl-evolution/${runId}`);
        const data = await response.json();

        if (!data.metrics) return;

        const traces = [];
        for (const [key, points] of Object.entries(data.metrics)) {
            traces.push({
                x: points.map(p => p.step),
                y: points.map(p => p.val),
                mode: 'lines',
                name: key.split('.').pop()
            });
        }

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e0e6ed' },
            title: 'KL Divergence Tracking',
            margin: { t: 40, r: 20, b: 30, l: 40 }
        };

        Plotly.react('kl-graph', traces, layout);
    } catch (error) {
        addLog("Failed to update KL Graph", "WARN");
    }
}

// --- 4. The Hardware Gauge & Top Metrics ---
async function updateHardwareStats(runId) {
    try {
        const response = await fetch(`/api/v1/metrics/efficiency-stats/${runId}`);
        const data = await response.json();

        if (data.detail) return;

        // Update the top text metrics in your HTML
        document.getElementById('gpu-value').innerText = data.metrics.gpu_util;
        
        // Change KeOps text color based on status
        const keopsElement = document.getElementById('keops-value');
        keopsElement.innerText = data.keops_status;
        if (data.keops_status === "JIT_COMPILING") {
            keopsElement.style.color = "#d29922"; // Yellow while compiling
        } else {
            keopsElement.style.color = "#3fb950"; // Green when cached
        }

    } catch (err) {
        console.error("Hardware API error:", err);
    }
}

// --- 5. The Master Controller ---
async function updateAllDashboards() {
    const runId = window.ACTIVE_RUN_ID;
    if (!runId || runId === "NO_RUNS_FOUND") {
        addLog("No active runs detected in database.", "WARN");
        return;
    }

    addLog(`Pinging A100 Engine for Run: ${runId.substring(0,8)}...`, "INFO");

    // Run all fetches concurrently for speed
    await Promise.all([
        drawMoneyGraph(runId),
        drawKLGraph(runId),
        updateHardwareStats(runId)
    ]);
}

// --- Event Listeners & Initialization ---

document.addEventListener("DOMContentLoaded", () => {
    addLog("Dashboard initialized. Connecting to FastAPI...", "SYSTEM");
    
    // Initial fetch
    updateAllDashboards();
    
    // Start the 5-second live refresh loop
    liveRefreshTimer = setInterval(updateAllDashboards, 5000);
});

document.getElementById('runSelector').addEventListener('change', (e) => {
    window.ACTIVE_RUN_ID = e.target.value;
    addLog(`Switched view to Run: ${window.ACTIVE_RUN_ID}`, "SYSTEM");
    
    // Reset the timer so it doesn't double-fire
    clearInterval(liveRefreshTimer);
    
    // Immediately fetch the new run data
    updateAllDashboards();
    
    // Restart the 5-second loop for the new run
    liveRefreshTimer = setInterval(updateAllDashboards, 5000);
});