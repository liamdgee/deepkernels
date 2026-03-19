// script.js
// 1. The Money Graph (Your code, fixed and polished)
async function drawMoneyGraph(runId) {
    try {
        // NOTE: Make sure this URL matches your actual FastAPI endpoint!
        const response = await fetch(`/api/v1/metrics/money-stats/${runId}`);
        const data = await response.json(); // Fixed: .json() instead of .data

        if (!data.points || data.points.length === 0) return;

        const x = data.points.map((_, i) => i);
        const y_pred = data.points.map(p => p.pred);
        const y_std = data.points.map(p => p.std);
        
        const y_upper = y_pred.map((val, i) => val + (1.96 * y_std[i]));
        const y_lower = y_pred.map((val, i) => val - (1.96 * y_std[i]));

        const trace_mean = {
            x: x, y: y_pred,
            type: 'scatter', name: 'NKN-GP Mean',
            line: {color: 'rgb(31, 119, 180)', width: 2}
        };

        const trace_ribbon = {
            x: [...x, ...x.reverse()],
            y: [...y_upper, ...y_lower.reverse()],
            fill: 'toself',
            fillcolor: 'rgba(31, 119, 180, 0.2)',
            line: {color: 'transparent'},
            name: '95% Confidence',
            type: 'scatter'
        };

        const layout = {
            title: 'Frontier GP Uncertainty Calibration',
            paper_bgcolor: '#111', 
            plot_bgcolor: '#111',
            font: { color: '#fff' },
            margin: { t: 50, r: 20, b: 40, l: 40 }
        };

        // Ensure you have a <div id="money-graph"> in your HTML
        Plotly.newPlot('money-graph', [trace_ribbon, trace_mean], layout);
    } catch (error) {
        console.error("Error drawing Money Graph:", error);
    }
}

// 2. The KL Graph (Converted to Async/Await)
async function drawKLGraph(runId) {
    try {
        const response = await fetch(`/api/v1/metrics/kl-evolution/${runId}`);
        const data = await response.json();

        if (data.detail) throw new Error(data.detail);

        const traces = [];
        for (const [key, points] of Object.entries(data.metrics)) {
            traces.push({
                x: points.map(p => p.step),
                y: points.map(p => p.val),
                mode: 'lines',
                name: key.split('.').pop() // Cleans up long names for the legend
            });
        }

        const layout = {
            title: 'KL Divergence Evolution',
            paper_bgcolor: '#111', // Matched your dark mode
            plot_bgcolor: '#111',
            font: { color: '#fff' },
            margin: { t: 50, r: 20, b: 40, l: 40 }
        };

        // Ensure you have a <div id="kl-graph"> in your HTML
        Plotly.newPlot('kl-graph', traces, layout);
    } catch (error) {
        console.error("Error drawing KL Graph:", error);
    }
}

// 3. The Master Controller
async function updateAllDashboards(runId) {
    if (!runId || runId === "NO_RUNS_FOUND") return;
    
    console.log(`Fetching all metrics for run: ${runId}`);

    // PRO-TIP: Promise.all runs both fetches at the exact same time
    // instead of waiting for the Money graph to finish before starting the KL graph.
    await Promise.all([
        drawMoneyGraph(runId),
        drawKLGraph(runId)
    ]);
}

// --- Event Listeners ---

// On initial page load
document.addEventListener("DOMContentLoaded", () => {
    updateAllDashboards(window.ACTIVE_RUN_ID);
});

// When the dropdown changes
document.getElementById('runSelector').addEventListener('change', (e) => {
    const selectedRun = e.target.value;
    window.ACTIVE_RUN_ID = selectedRun;
    updateAllDashboards(selectedRun);
});