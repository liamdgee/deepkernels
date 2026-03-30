
from collections import defaultdict
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import httpx
import asyncio

# ==========================================
# 1. SETUP & THEMES
# ==========================================
app = dash.Dash(__name__, title="DeepKernels -- An algorithmic auditing tool for ethical lending standards", suppress_callback_exceptions=True)

TEST_RUN_ID = "latest_run_001" 
API_BASE_URL = "/api/v1"

BESPOKE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#adbac7', family="Inter, sans-serif", size=12),
    margin=dict(t=60, b=40, l=40, r=60),
    hovermode='x unified',
    xaxis=dict(
        showgrid=True, gridcolor='#30363d', zeroline=False,
        linecolor='#444c56', tickfont=dict(color='#768390')
    ),
    yaxis=dict(
        showgrid=True, gridcolor='#30363d', zeroline=False,
        linecolor='#444c56', tickfont=dict(color='#768390')
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor='rgba(0,0,0,0)', bordercolor='#444c56', borderwidth=0
    )
)

async def fetch_api_data_async(client: httpx.AsyncClient, endpoint: str) -> dict | None:
    try:
        response = await client.get(f"{API_BASE_URL}/{endpoint}", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return None

async def trigger_simulation_async(payload: dict) -> dict | None:
    """New courier specifically for the Simulator POST request."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/inference/simulate", json=payload, timeout=15.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

# ==========================================
# -- GRAPHING FUNCTIONS (The Painters)
# ==========================================
def build_predictive_posterior_graph(data: dict | None) -> go.Figure:
    """FRONTEND PAINTER: Builds the GP Uncertainty graph."""
    fig_money = go.Figure()
    
    if data and data.get("money_data") and data["money_data"].get("points"):
        pts = data["money_data"]["points"]
        steps = list(range(len(pts)))
        y_true = [p["true"] for p in pts]
        y_pred = [p["pred"] for p in pts]
        y_std = [p["std"] for p in pts]
        
        fig_money.add_trace(go.Scatter(
            x=steps + steps[::-1], 
            y=[p + (1.96 * s) for p, s in zip(y_pred, y_std)] + 
              [p - (1.96 * s) for p, s in zip(y_pred, y_std)][::-1], 
            fill='toself', 
            fillcolor='rgba(88, 166, 255, 0.15)', 
            line=dict(color='rgba(255,255,255,0)'), 
            name='95% Predictive Interval',
            hoverinfo='skip'
        ))
        
        fig_money.add_trace(go.Scatter(
            x=steps, y=y_pred, mode='lines', name='GP Mean',
            line=dict(color='#58a6ff', width=2.5, dash='solid')
        ))
        
        fig_money.add_trace(go.Scatter(
            x=steps, y=y_true, mode='markers', name='Observed',
            marker=dict(color='#f6f8fa', size=5, opacity=0.8, 
                        line=dict(color='#444c56', width=1))
        ))
        
    fig_money.update_layout(
        **BESPOKE_LAYOUT, 
        title="<b>Predictive Posterior</b> <span style='color:#768390; font-weight:normal'>(Uncertainty Evaluation)</span>"
    )
    return fig_money

def build_ece_graph(data: dict | None) -> go.Figure:
    fig = go.Figure()
    color_ece = '#d2a8ff'
    if data and data.get("ece"):
        steps = [p["step"] for p in data["ece"]]
        vals = [p["val"] for p in data["ece"]]
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode='lines+markers', name='ECE',
            line=dict(color=color_ece, width=3, shape='spline'),
            marker=dict(size=5, color=color_ece, line=dict(color='rgba(255,255,255,0.7)', width=1)),
            hovertemplate="<b>ECE</b><br>Value: %{y:.4f}<extra></extra>"
        ))
        
        fig.add_annotation(
            x=steps[-1], y=vals[-1],
            text=f"{vals[-1]:.4f}",
            showarrow=False, xanchor='left', xshift=10,
            font=dict(family="Arial", size=11, color=color_ece),
            bgcolor="rgba(15, 15, 15, 0.8)", bordercolor=color_ece, borderwidth=1, borderpad=3
        )

    fig.update_layout(title="Expected Calibration Error (↓ is better)", **BESPOKE_LAYOUT)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.05)", title="Step", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.05)", title="ECE (Lower = Better)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
    return fig


def build_nlpd_graph(data: dict | None) -> go.Figure:
    fig = go.Figure()
    color_nlpd = '#58a6ff'
    if data and data.get("nlpd"):
        steps = [p["step"] for p in data["nlpd"]]
        vals = [p["val"] for p in data["nlpd"]]
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode='lines+markers', name='NLPD',
            line=dict(color=color_nlpd, width=3, shape='spline'),
            marker=dict(size=5, color=color_nlpd, line=dict(color='rgba(255,255,255,0.7)', width=1)),
            hovertemplate="<b>NLPD</b><br>Value: %{y:.4f}<extra></extra>"
        ))
        
        fig.add_annotation(
            x=steps[-1], y=vals[-1],
            text=f"{vals[-1]:.4f}",
            showarrow=False, xanchor='left', xshift=10,
            font=dict(family="Arial", size=11, color=color_nlpd),
            bgcolor="rgba(15, 15, 15, 0.8)", bordercolor=color_nlpd, borderwidth=1, borderpad=3
        )
        fig.add_trace(go.Scatter(
            x=steps, y=vals, name="NLPD",
            line=dict(color='#ff7b72', width=2) # Coral Red
        ))
        
    fig.update_layout(title="Negative Log Predictive Density (↓ is better)", **BESPOKE_LAYOUT)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.05)", title="Step", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.05)", title="NLPD (Lower = Better)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
    return fig

def build_money_graph(data: dict | None) -> go.Figure:
    fig = go.Figure()
    if data and data.get("points"):
        points = data['points']
        steps = [p["step"] for p in data['points']] if "step" in data['points'][0] else list(range(len(data['points'])))
        y_true = [p["true"] for p in data['points']]
        y_pred = [p["pred"] for p in data['points']]
    
        fig.add_trace(go.Scatter(x=steps, y=y_true, name="Ground Truth", line=dict(color="#7ee787", width=2)))
        fig.add_trace(go.Scatter(x=steps, y=y_pred, name="Prediction", line=dict(color="#58a6ff", width=2)))
    
    fig.update_layout(title="Predictive Trajectory vs Ground Truth", **BESPOKE_LAYOUT)

    return fig


def build_pulse_gauge(data: dict | None) -> go.Figure:
    fig = go.Figure()
    
    if data and data.get('summary'):
        score = data["summary"]["health_score"]
        grade = data["summary"]["model_grade"]
        bar_color = '#7ee787' if score > 85 else ('#d2a8ff' if score > 70 else '#f85149')
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': f"<b>System Health</b><br><span style='color: #768390; font-size:0.8em'>Grade: {grade}</span>"},
            number={'suffix': "/100", 'font': {'color': bar_color, 'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#30363d"},
                'bar': {'color': bar_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#30363d",
                'steps': [
                    {'range': [0, 70], 'color': "rgba(248, 81, 73, 0.1)"},
                    {'range': [70, 85], 'color': "rgba(210, 168, 255, 0.1)"},
                    {'range': [85, 100], 'color': "rgba(126, 231, 135, 0.1)"}
                ],
            }
        ))

    fig.update_layout(title="Model Pulse / Health Score", **BESPOKE_LAYOUT)
    return fig

def build_gradient_flow_graph(data: dict | None) -> go.Figure:
    """Paints a horizontal bar chart of kernel gradient magnitudes."""
    fig = go.Figure()
    if not data or not data.get('gradients'):
        fig.update_layout(title="Kernel Gradient Flow (Awaiting Data)", **BESPOKE_LAYOUT)
        fig.update_xaxes(type="log", title_text="L2 Norm of Gradients")
        fig.update_yaxes(title_text="Kernel Primitive")
        return fig
    gradients = data['gradients']
    layers = list(gradients.keys())
    magnitudes = list(gradients.values())
    
    # 3. Calculate health metrics
    is_healthy = all(v > 1e-5 for v in magnitudes)
    bar_colors = ['#58a6ff' if v > 1e-5 else '#f85149' for v in magnitudes]
    
    health_text = "HEALTHY" if is_healthy else "VANISHING"
    health_color = "#7ee787" if is_healthy else "#ff7b72"

    # 4. Paint the bars
    fig.add_trace(go.Bar(
        x=magnitudes,
        y=layers,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='#444c56', width=1) # Keeps the borders crisp
        ),
        hovertemplate="<b>%{y}</b><br>Magnitude: %{x:.6f}<extra></extra>"
    ))

    # 5. Apply the master theme and dynamic title
    fig.update_layout(
        title=f"Kernel Gradient Flow (<span style='color:{health_color}'>{health_text}</span>)",
        **BESPOKE_LAYOUT
    )
    
    # 6. Safely inject axis overrides
    fig.update_xaxes(type="log", title_text="L2 Norm of Gradients")
    fig.update_yaxes(title_text="Kernel Primitive")

    return fig

def generate_gp_paths(data: dict | None, num_scribbles=10) -> go.Figure:
    fig = go.Figure()
    
    if not data or not data.get('mean'):
        fig.update_layout(title="GP Posterior Manifold (Awaiting Data)", **BESPOKE_LAYOUT)
        return fig

    # Safely extract data
    steps = data['steps']
    mean = data['mean']
    std = data['std']
    samples = data['samples']

    # Trace 0: 95% Confidence Interval
    fig.add_trace(go.Scatter(
        x=list(steps) + list(steps)[::-1],
        y=list(mean + 1.96*std) + list(mean - 1.96*std)[::-1],
        fill='toself', 
        fillcolor='rgba(88, 166, 255, 0.15)',
        line=dict(color='rgba(255,255,255,0)'), 
        name='95% CI Uncertainty',
        hoverinfo='skip'
    ))

    for i in range(num_scribbles):
        fig.add_trace(go.Scatter(
            x=steps, 
            y=samples[i].flatten(), 
            mode='lines',
            line=dict(width=1, color='rgba(210, 168, 255, 0.3)', shape='spline'),
            showlegend=False,
            hoverinfo='skip' 
        ))

    # Trace N+1: The Mean Prediction
    fig.add_trace(go.Scatter(
        x=steps, 
        y=mean, 
        mode='lines+markers', 
        name='GP Mean Prediction',
        line=dict(color='#58a6ff', width=3, shape='spline'),
        marker=dict(size=4, color='#58a6ff'),
        hovertemplate="<b>Step %{x}</b><br>Mean: %{y:.4f}<extra></extra>"
    ))

    show_all = [True] * len(fig.data) 
    hide_scribbles = [True] + [False] * num_scribbles + [True]
    fig.update_layout(
        title="GP Posterior Manifold",
        
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.0,
            y=1.15,
            buttons=[
                dict(label="Show Samples", method="update", args=[{"visible": show_all}]),
                dict(label="Hide Samples", method="update", args=[{"visible": hide_scribbles}])
            ],
            pad={"r": 10, "t": -10},
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(color="#adbac7")
        )],
        
        **BESPOKE_LAYOUT
    )
    
    fig.update_xaxes(title_text="Time Step (dt)")
    fig.update_yaxes(title_text="Predicted Value")

    return fig

def build_kl_evolution_graph(data: dict | None) -> go.Figure:
    """FRONTEND PAINTER: Builds the Latent Regularization graph."""
    fig_kl = go.Figure()
    
    if data and data.get("kl_data") and data["kl_data"].get("metrics"):
        colors = ['#79c0ff', '#d2a8ff', '#ffa657', '#7ee787']
        for i, (key, points) in enumerate(data["kl_data"]["metrics"].items()):
            fig_kl.add_trace(go.Scatter(
                x=[p["step"] for p in points], 
                y=[p["val"] for p in points], 
                mode='lines', 
                name=key.split('.')[-1],
                line=dict(color=colors[i % len(colors)], width=2)
            ))
            
    fig_kl.update_layout(
        **BESPOKE_LAYOUT, 
        title="<b>Latent Regularization</b> <span style='color:#768390; font-weight:normal'>(KL Divergence)</span>"
    )
    return fig_kl

def build_simulation_trajectory(data: dict | None) -> go.Figure:
    fig = go.Figure()
    
    if not data or not data.get("trajectory"):
        fig.update_layout(title="Awaiting Simulation...", **BESPOKE_LAYOUT)
        fig.update_yaxes(range=[0, 1])
        return fig
        
    # Extract Data
    traj = data["trajectory"]
    steps = list(range(len(traj)))
    
    final_step = steps[-1]
    final_val = traj[-1] 
    
    final_mean_text = f"{final_val:.4f}"
    std_val = data.get('final_std', 0) 
    final_sd_text = f"± {std_val * 1.96:.4f}"
    
    fig.add_trace(go.Scatter(
        x=steps, y=traj, mode='lines+markers', name="Rejection Probability",
        line=dict(color='#ff7b72', width=3, shape='spline'), 
        fill='tozeroy', fillcolor='rgba(255, 123, 114, 0.1)'
    ))
    
    combo_text = (
        f"<b>Final Prediction</b><br>"
        f"<span style='color:#ff7b72; font-size:18px;'>{final_mean_text}</span><br>"
        f"<span style='color:#8b949e; font-size:12px;'>Uncertainty: {final_sd_text}</span>"
    )
    
    fig.add_annotation(
        x=final_step,
        y=final_val,
        text=combo_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(255,255,255,0.4)",
        ax=-80, # Pulled back slightly more to fit the extra text
        ay=-50, 
        font=dict(family="Inter, sans-serif", color="white"),
        align="left",
        bgcolor="rgba(15, 15, 15, 0.9)", 
        bordercolor="#ff7b72", 
        borderwidth=1,
        borderpad=8
    )

    # Apply Master Layout
    fig.update_layout(title="Autoregressive Rejection Trajectory", **BESPOKE_LAYOUT)
    fig.update_yaxes(title_text="Probability of Rejection", range=[0, 1])
    
    return fig

def get_telemetry_layout():
    """Returns your exact Telemetry CSS Grid."""
    return html.Div(children=[
        
        dcc.Interval(id='metrics-interval', interval=5000, n_intervals=0),
        
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 3fr', # Top Row: Gauge (25%), Money (75%)
                'gap': '24px',
                'marginBottom': '24px'
            },
            children=[
                html.Div(
                    style={'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'borderRadius': '6px', 'padding': '15px'}, 
                    children=[dcc.Graph(id='live-pulse-gauge')]
                ),
                html.Div(
                    style={'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'borderRadius': '6px', 'padding': '15px'}, 
                    children=[dcc.Graph(id='live-money-graph', style={'height': '350px'})]
                )
            ]
        ),
        
        # --- SUB-METRICS ROW (3 Columns) ---
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(3, 1fr)', # 3 equal columns
                'gap': '24px',
                'marginBottom': '24px'
            },
            children=[
                html.Div(
                    style={'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'borderRadius': '6px', 'padding': '15px'}, 
                    children=[dcc.Graph(id='live-gradients-graph', style={'height': '300px'})]
                ),
                html.Div(
                    style={'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'borderRadius': '6px', 'padding': '15px'}, 
                    children=[dcc.Graph(id='live-ece-graph', style={'height': '300px'})]
                ),
                html.Div(
                    style={'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'borderRadius': '6px', 'padding': '15px'}, 
                    children=[dcc.Graph(id='live-nlpd-graph', style={'height': '300px'})]
                )
            ]
        ),
        
        # --- FULL WIDTH BOTTOM ROW ---
        html.Div(
            style={'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'borderRadius': '6px', 'padding': '15px'},
            children=[dcc.Graph(id='live-gp-paths', style={'height': '500px'})]
        )
        
    ])

def get_simulator_layout():
    """Returns the completed Fairness Simulator with all required Proxies."""
    return html.Div(style={'display': 'flex', 'gap': '24px'}, children=[
        # --- INPUT PANEL (Left) ---
        html.Div(style={'flex': '1', 'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'padding': '20px', 'borderRadius': '6px'}, children=[
            html.H3("Sociocorrelation Parameters", style={'marginTop': '0', 'borderBottom': '1px solid #30363d', 'paddingBottom': '10px'}),
            
            html.Label("Business Tenure (Months)"), 
            dcc.Slider(id='sim-tenure', min=1, max=120, value=24, step=1, marks={1: '1m', 60: '5y', 120: '10y'}),
            
            html.Label("Amount Sought ($)", style={'marginTop': '20px', 'display': 'block'}), 
            dcc.Slider(id='sim-amount', min=5000, max=500000, value=25000, step=5000, marks={5000: '5k', 250000: '250k', 500000: '500k'}),
            
            html.Label("Lending Institution Type", style={'marginTop': '20px', 'display': 'block'}),
            dcc.Dropdown(
                id='sim-lender', 
                options=[
                    {'label': 'Traditional Bank', 'value': 'bank'}, 
                    {'label': 'Fintech', 'value': 'fintech'},
                    {'label': 'Credit Union', 'value': 'creditunion'},
                    {'label': 'CDFI', 'value': 'cdfi'}
                ], 
                value='fintech', 
                style={'color': '#000'}
            ),

            html.Hr(style={'margin': '30px 0', 'borderColor': '#30363d'}),
            html.H4("Algorithmic Bias Proxies", style={'color': '#8b949e'}),

            html.Label("Animosity Proxy (Inter-group friction)"), 
            dcc.Slider(id='sim-animus', min=0.1, max=6.9, value=3.5, step=0.1),

            # --- NEW PROXIES ADDED HERE ---
            html.Label("Isolation Proxy (Network Density)", style={'marginTop': '20px', 'display': 'block'}), 
            dcc.Slider(id='sim-isolation', min=0.1, max=6.9, value=3.5, step=0.1),

            html.Label("IAT Score (Implicit Association)", style={'marginTop': '20px', 'display': 'block'}), 
            dcc.Slider(id='sim-iat', min=0.1, max=6.9, value=3.5, step=0.1),
            html.Div(style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#0d1117', 'borderRadius': '4px', 'border': '1px solid #30363d'}, children=[
                dcc.Checklist(
                    id='sim-compare-all',
                    options=[{'label': ' Compare All Lenders (Competitive Manifold)', 'value': 'all'}],
                    value=['all'],
                    style={'color': '#7ee787', 'fontSize': '14px'}
                )
            ]),
            html.Button(
                "🚀 Simulate Rejection Trajectory", 
                id='sim-submit-btn', 
                n_clicks=0, 
                style={
                    'width': '100%', 'padding': '15px', 'marginTop': '30px', 
                    'backgroundColor': '#238636', 'color': 'white', 'border': 'none', 
                    'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 'bold'
                }
            )
        ]),

        # --- OUTPUT PANEL (Right) ---
        html.Div(style={'flex': '2', 'backgroundColor': '#161b22', 'border': '1px solid #30363d', 'padding': '20px', 'borderRadius': '6px'}, children=[
            dcc.Graph(id='sim-output-graph', style={'height': '550px'}),
            html.Div(id='sim-output-text', style={
                'marginTop': '20px', 'fontSize': '18px', 'color': '#7ee787', 
                'fontWeight': 'bold', 'textAlign': 'center', 'fontFamily': 'monospace'
            })
        ])
    ])


# ==========================================
# 5. THE MASTER APP SHELL
# ==========================================
app.layout = html.Div(style={'backgroundColor': '#0d1117', 'minHeight': '100vh', 'padding': '20px 40px', 'color': '#adbac7', 'fontFamily': 'Inter, sans-serif'}, children=[
    
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'borderBottom': '1px solid #444c56', 'paddingBottom': '15px', 'marginBottom': '30px'}, children=[
        html.H1("DeepKernels Control Panel", style={'fontSize': '24px', 'fontWeight': '400', 'margin': '0'}),
        html.Div(style={'width': '250px'}, children=[
            dcc.Dropdown(
                id='view-selector', 
                options=[{'label': '📊 System Telemetry', 'value': 'telemetry'}, {'label': '⚖️ Fairness Simulator', 'value': 'simulator'}], 
                value='telemetry', 
                clearable=False, 
                style={'color': '#000'}
            )
        ])
    ]),
    
    html.Div(id='page-content')
])


@app.callback(
    [Output('sim-output-graph', 'figure'), Output('sim-output-text', 'children')],
    [Input('sim-submit-btn', 'n_clicks')],
    [
        State('sim-tenure', 'value'), 
        State('sim-amount', 'value'), 
        State('sim-lender', 'value'), 
        State('sim-animus', 'value'),
        State('sim-isolation', 'value'),
        State('sim-iat', 'value'),
        State('sim-compare-all', 'value')
    ],
    prevent_initial_call=True
)
async def run_simulation(n_clicks, tenure, amount, lender, animus, isolation, iat, compare_val):
    is_compare = 'all' in (compare_val or [])
    payload = {
        "tenure_months": float(tenure),
        "amount_sought": float(amount),
        "lender_type": str(lender).lower(),
        "animus_proxy": float(animus),
        "isolation_proxy": float(isolation), # Now dynamic!
        "iat_score": float(iat),             # Now dynamic!
        "horizon_steps": 64,
        "compare_all_lenders": is_compare    # Now dynamic!
    }
    
    data = await trigger_simulation_async(payload)
    
    if not data:
        return go.Figure().update_layout(title="❌ API CONNECTION FAILED", **BESPOKE_LAYOUT), "Check if Uvicorn is running on Port 8000."
    
    fig = go.Figure()
    
    if is_compare and isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
        for lender_name, metrics in data.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(metrics['trajectory']))),
                y=metrics['trajectory'],
                name=lender_name.upper(),
                mode='lines',
                line=dict(width=2, shape='spline')
            ))
        status_msg = "Comparative manifold generated across lender archetypes."
    else:
        fig = build_simulation_trajectory(data)
        mean = data.get('final_mean', 0)
        status_msg = f"Target Projection: {mean:.4f} for {lender.upper()}"

    fig.update_layout(**BESPOKE_LAYOUT)
    return fig, status_msg
    
@app.callback(Output('page-content', 'children'), [Input('view-selector', 'value')])
def render_page(view_mode):
    if view_mode == 'simulator':
        return get_simulator_layout()
    return get_telemetry_layout()


@app.callback(
    [Output('live-money-graph', 'figure'), Output('live-pulse-gauge', 'figure'),
     Output('live-gradients-graph', 'figure'), Output('live-ece-graph', 'figure'),
     Output('live-nlpd-graph', 'figure'), Output('live-gp-paths', 'figure')],
    [Input('metrics-interval', 'n_intervals')]
)
async def update_telemetry(n_intervals):
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_api_data_async(client, f"metrics/money-stats/{TEST_RUN_ID}"),
            fetch_api_data_async(client, f"metrics/pulse-check/{TEST_RUN_ID}"),
            fetch_api_data_async(client, f"metrics/gradient-flow/{TEST_RUN_ID}"),
            fetch_api_data_async(client, f"metrics/calibration-stats/{TEST_RUN_ID}"), 
            fetch_api_data_async(client, f"metrics/gp-paths/{TEST_RUN_ID}")
        ]
        results = await asyncio.gather(*tasks)
    
    money, pulse, grad, calib, paths = results
    
    return (
        build_money_graph(money), build_pulse_gauge(pulse), build_gradient_flow_graph(grad),
        build_ece_graph(calib), build_nlpd_graph(calib), generate_gp_paths(paths)
    )


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8050)