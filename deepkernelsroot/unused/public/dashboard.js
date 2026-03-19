// 1. Point this to your FastAPI address
const API_URL = "http://127.0.0.1:8000/api/v1/metrics/money-stats/e0512de7583146329a010afc984a25a5";

async function updateDashboard() {
    try {
        const response = await fetch(API_URL);
        const data = await response.json();

        // 2. Extracting your GP points
        const steps = data.points.map((p, i) => i);
        const y_true = data.points.map(p => p.true);
        const y_pred = data.points.map(p => p.pred);
        const y_std = data.points.map(p => p.std);

        // 3. Create the 'Money Graph' traces
        const truthTrace = {
            x: steps, y: y_true,
            type: 'scatter', mode: 'markers', name: 'Ground Truth',
            marker: { color: 'white', size: 4 }
        };

        const meanTrace = {
            x: steps, y: y_pred,
            type: 'scatter', name: 'GP Mean Prediction',
            line: { color: '#1f77b4', width: 3 }
        };

        // This creates the 'Shaded Ribbon' for uncertainty
        const uncertaintyTrace = {
            x: [...steps, ...steps.slice().reverse()],
            y: [
                ...y_pred.map((v, i) => v + (1.96 * y_std[i])), // Upper 95%
                ...y_pred.map((v, i) => v - (1.96 * y_std[i])).reverse() // Lower 95%
            ],
            fill: 'toself',
            fillcolor: 'rgba(31, 119, 180, 0.3)',
            line: { color: 'transparent' },
            name: '95% Confidence Interval',
            type: 'scatter'
        };

        const layout = {
            paper_bgcolor: '#111',
            plot_bgcolor: '#222',
            font: { color: '#fff' },
            title: 'NKN-GP Uncertainty Calibration'
        };

        Plotly.newPlot('money-graph', [uncertaintyTrace, meanTrace, truthTrace], layout);

    } catch (err) {
        console.error("FastAPI not responding. Is the server running?", err);
    }
}

// Run it once when the page opens
updateDashboard();

// Refresh every 10 seconds to watch the A100's progress live
setInterval(updateDashboard, 10000);

// Add the second API URL
const HW_API_URL = "http://127.0.0.1:8000/api/v1/hardware/efficiency-stats/e0512de7583146329a010afc984a25a5";

async function updateHardwareStats() {
    try {
        const response = await fetch(HW_API_URL);
        const data = await response.json();

        // Extract raw numbers (removing the % sign if necessary)
        const gpuUtil = parseFloat(data.metrics.gpu_util);
        const vramUsage = parseFloat(data.metrics.vram_usage);

        const trace = [
            {
                type: "indicator",
                mode: "gauge+number",
                value: gpuUtil,
                title: { text: "A100 GPU Utilization", font: { size: 18 } },
                gauge: {
                    axis: { range: [0, 100], tickwidth: 1, tickcolor: "white" },
                    bar: { color: "#1f77b4" }, // Matches your GP ribbon
                    bgcolor: "white",
                    borderwidth: 2,
                    bordercolor: "gray",
                    steps: [
                        { range: [0, 50], color: "rgba(0, 255, 0, 0.1)" },
                        { range: [50, 85], color: "rgba(255, 255, 0, 0.1)" },
                        { range: [85, 100], color: "rgba(255, 0, 0, 0.1)" }
                    ],
                    threshold: {
                        line: { color: "red", width: 4 },
                        thickness: 0.75,
                        value: 99
                    }
                }
            }
        ];

        const layout = {
            width: 300, height: 250,
            margin: { t: 25, r: 25, l: 25, b: 25 },
            paper_bgcolor: "#111",
            font: { color: "white", family: "Arial" }
        };

        Plotly.newPlot('gpu-gauge', trace, layout);

    } catch (err) {
        console.error("Hardware API error:", err);
    }
}

// Update the main loop to include the hardware call
async function masterUpdate() {
    await updateDashboard();     // Your Money Graph
    await updateHardwareStats(); // Your GPU Gauge
}

setInterval(masterUpdate, 5000); // Check hardware every 5 seconds
masterUpdate();

function addLog(message, type = "INFO") {
    const console = document.getElementById('log-console');
    const time = new Date().toLocaleTimeString();
    
    // Choose color based on log level
    let color = "#00ff00"; // Default Green
    if (type === "WARN") color = "#ffaa00";
    if (type === "ERROR") color = "#ff0000";
    if (type === "SYSTEM") color = "#58a6ff";

    const logEntry = document.createElement('div');
    logEntry.style.color = color;
    logEntry.innerHTML = `[${time}] [${type}] ${message}`;
    
    console.appendChild(logEntry);
    
    // Auto-scroll to bottom
    console.scrollTop = console.scrollHeight;

    // Keep only the last 50 logs to prevent lag
    if (console.childNodes.length > 50) {
        console.removeChild(console.firstChild);
    }
}