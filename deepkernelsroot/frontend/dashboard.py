import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import httpx
import asyncio
from pathlib import Path
import numpy as np

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
# ==========================================
# 1. SETUP & THEMES
# ==========================================
app = dash.Dash(__name__, title="DeepKernels -- An algorithmic auditing tool for ethical lending standards", suppress_callback_exceptions=True)


import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000") + "/v1"
GRID_COLOR = '#e8e8ed'  # Soft light-gray grid lines
AXIS_LABEL_COLOR = '#86868b'  # Muted secondary text
ZERO_LINE_COLOR = '#d2d2d7'   # Visible but subtle baseline
TEXT_COLOR = '#1d1d1f'        # Deep charcoal/black text


BESPOKE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',  
    font=dict(
        color='#1d1d1f', 
        family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif", 
        size=12
    ),
    hovermode='x unified',
    
    margin=dict(t=80, b=80, l=50, r=20), 
    
    xaxis=dict(
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.04)', zeroline=False,
        linecolor='rgba(0, 0, 0, 0.1)', tickfont=dict(color='#86868b'), automargin=True, fixedrange=True
    ),
    yaxis=dict(
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.04)', zeroline=True,
        zerolinecolor='rgba(0, 0, 0, 0.1)', zerolinewidth=1, linecolor='rgba(0, 0, 0, 0.1)', 
        tickfont=dict(color='#86868b'), automargin=True, fixedrange=True
    ),
    
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor='rgba(255, 255, 255, 0.65)',
        bordercolor='rgba(0, 0, 0, 0.05)',
        borderwidth=1,
        font=dict(size=10, color='#86868b')
    )
)

X_TEXT ="Simulated Application Processes"
Y_TEXT = "Δ Rejection Rate (%)"


#=============
#-css / html -#
#==============

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <meta name="apple-mobile-web-app-capable" content="yes">

        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">

        <meta name="apple-mobile-web-app-title" content="deepkernels">

        <link rel="apple-touch-icon" href="/assets/apple-icon.png">

        <style>
            /* 1. FOUNDATION: APPLE OFF-WHITE */
            body {
                background-color: #f5f5f7 !important; 
                color: #1d1d1f !important;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
                margin: 0;
            }

            #react-entry-point {
                background-color: #f5f5f7 !important;
            }
            /* By default, the layout container is a flex row. We need a class for it. */
            .main-dashboard-container {
                display: flex;
                flex-direction: row; /* Side-by-side on desktop */
                gap: 20px;
                align-items: stretch;
            }

            /* When the screen is narrow (like a phone), stack them! */
            @media (max-width: 850px) {
                .main-dashboard-container {
                    flex-direction: column; /* Stack vertically on mobile */
                }
                
                /* Ensure the left column doesn't stay fixed at 360px width on mobile */
                .mobile-fluid-col {
                    flex: none !important;
                    width: 100% !important;
                }
            }/* By default, the layout container is a flex row. We need a class for it. */
            .main-dashboard-container {
                display: flex;
                flex-direction: row; /* Side-by-side on desktop */
                gap: 20px;
                align-items: stretch;
            }

            /* When the screen is narrow (like a phone), stack them! */
            @media (max-width: 850px) {
                .main-dashboard-container {
                    flex-direction: column; /* Stack vertically on mobile */
                }
                
                /* Ensure the left column doesn't stay fixed at 360px width on mobile */
                .mobile-fluid-col {
                    flex: none !important;
                    width: 100% !important;
                }
            }

            /* --- THE HOVER STATE --- */
            .lender-toggle-btn:hover {
                background-color: rgba(255, 255, 255, 0.4); /* Subtle ghosting effect */
                color: #1d1d1f !important; /* Darken text slightly */
                transform: translateY(-1px); /* Subtle lift */
            }

            /* Ensure the hover doesn't override the ACTIVE (checked) state */
            .lender-toggle-container input[type="radio"]:checked + label:hover {
                background-color: #ffffff !important; /* Keep solid white if already selected */
                transform: scale(1.02); /* Keep the active scale */
                cursor: default;
            }

            /* 2. CARD NEUMORPHISM */
            .control-card, .graph-card {
                background-color: rgba(255, 255, 255, 0.8) !important;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                border-radius: 18px !important;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05) !important;
                padding: 24px;
            }

            /* 3. LOGO & HEADER */
            .logo-box {
                width: 35px;
                height: 35px;
                background: linear-gradient(135deg, #0071e3, #5e5ce6) !important;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(94, 92, 230, 0.3);
                display: block !important;
            }

            label, .section-title {
                color: #1d1d1f !important;
                font-weight: 600 !important;
                letter-spacing: -0.02em;
                text-transform: none !important; /* Apple style uses sentence case */
            }

            /* 4. THE SIGMA SEGMENTED CONTROL */
            .custom-tabs-container {
                background-color: #e8e8ed !important; 
                border-radius: 10px !important;
                padding: 2px !important;
                display: flex !important;
                height: 32px !important;
                border: none !important;
                width: 100% !important;
            }

            .custom-tabs-container > div {
                display: flex !important;
                width: 100% !important;
            }

            .custom-tab {
                flex: 1 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                background-color: transparent !important;
                color: #86868b !important;
                border: none !important;
                font-size: 11px !important;
                font-weight: 500 !important;
                cursor: pointer !important;
                transition: all 0.2s ease !important;
                border-radius: 8px !important;
            }

            .custom-tab--selected {
                background-color: #ffffff !important;
                color: #0071e3 !important;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
                font-weight: 700 !important;
            }

            /* 5. SUBMIT BUTTON (Bold Vibrant Purple) */
            #sim-submit-btn {
                background: linear-gradient(135deg, #5e5ce6, #af52de) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                font-weight: 700 !important;
                letter-spacing: 0.5px;
                transition: all 0.3s cubic-bezier(0, 0, 0.5, 1);
            }

            #sim-submit-btn:hover {
                transform: scale(1.02);
                box-shadow: 0 10px 20px rgba(175, 82, 222, 0.3) !important;
            }

            /* 6. TRAJECTORY ANIMATION */
            .animate-trajectory .js-line {
                stroke-dasharray: 2000;
                stroke-dashoffset: 2000;
                animation: dash 3s linear forwards;
            }

            @keyframes dash { to { stroke-dashoffset: 0; } }
            
            /* 7. LENDER TOGGLES */
            .lender-toggle-container {
                display: flex;
                gap: 6px;
                background-color: #e8e8ed;
                padding: 4px;
                border-radius: 10px;
            }

            .lender-toggle-btn {
                flex: 1;
                text-align: center;
                padding: 8px;
                font-size: 10px;
                font-weight: 600;
                color: #86868b;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s;
            }

            .lender-toggle-container input[type="radio"]:checked + label {
                background-color: #ffffff !important;
                color: #0071e3 !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }

            /* --- SEGMENTED CONTROL (The Sigma Bounds) --- */
            .ios-tabs-container {
                display: flex !important;
                background-color: #f1f5f9 !important; /* Smooth, modern light gray */
                border-radius: 8px !important;
                padding: 4px !important;
                height: 38px !important;
                box-sizing: border-box !important;
                border: none !important;
            }

            .ios-tab {
                flex: 1 !important;
                background-color: transparent !important;
                border: none !important;
                color: #64748b !important;
                font-weight: 600 !important;
                font-size: 13px !important;
                padding: 0 !important;
                margin: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                border-radius: 6px !important;
                cursor: pointer !important;
                transition: all 0.2s ease !important;
            }

            .ios-tab--active {
                background-color: #ffffff !important;
                color: #3B82F6 !important; /* Matches your submit button's blue */
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08) !important;
                border: none !important;
            }

            /* --- THE CHECKBOX --- */
            .clean-checkbox input[type="checkbox"] {
                accent-color: #8B5CF6; /* Matches your submit button's purple */
                width: 16px;
                height: 16px;
                cursor: pointer;
                margin-right: 8px;
                position: relative;
                top: 2px;
            }

            .clean-checkbox label {
                color: #475569;
                font-weight: 600;
                font-size: 14px;
                cursor: pointer;
            }
            
            .rc-slider-tooltip,
            .dash-tooltip,
            .custom-slider input[type="number"],
            .rc-slider-handle-tooltip {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
            }

            .lender-toggle-container input[type="radio"] { display: none; }

            /* --- FORCE HIDE SLIDER TEXT INPUTS --- */
            .rc-slider-tooltip,
            .dash-tooltip,
            .custom-slider input[type="number"],
            .rc-slider-handle-tooltip {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
            }

            /* ========================================== */
            /* NEW: MOBILE RESPONSIVENESS (The Quick Fix) */
            /* ========================================== */
            .main-dashboard-container {
                display: flex;
                flex-direction: row; 
                gap: 20px;
                align-items: stretch;
            }

            @media (max-width: 850px) {
                .main-dashboard-container {
                    flex-direction: column !important; 
                }
                
                .mobile-fluid-col {
                    flex: none !important;
                    width: auto !important; /* Lets it fill the screen horizontally */
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def get_ghost_figure():
    steps = list(range(64))
    fig = go.Figure()
    
    for _ in range(5):
        ghost_jitter = np.random.normal(0, 0.000125, 64) 
        fig.add_trace(go.Scatter(
            x=steps, y=ghost_jitter, 
            mode='lines',
            line=dict(width=1, color='rgba(88, 166, 255, 0.2)', shape='spline'),
            showlegend=False, hoverinfo='skip'
        ))
    
    fig.update_layout(
        **BESPOKE_LAYOUT,
        title={'text': "<b>Initialise Risk Simulation Engine</b>", 'y': 0.95, 'x': 0.5}
    )
    

    fig.update_xaxes(
        title_text=X_TEXT,
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(size=11, color=AXIS_LABEL_COLOR),
        tickfont=dict(color=AXIS_LABEL_COLOR),
        linecolor=ZERO_LINE_COLOR
    )

    fig.update_yaxes(
        title_text=Y_TEXT,
        tickformat='.1%',
        zeroline=True,
        zerolinecolor=ZERO_LINE_COLOR,
        zerolinewidth=1,
        gridcolor=GRID_COLOR,
        title_font=dict(size=11, color=AXIS_LABEL_COLOR),
        tickfont=dict(color=AXIS_LABEL_COLOR)
    )
    
    return fig

async def fetch_api_data_async(client: httpx.AsyncClient, endpoint: str) -> dict | None:
    try:
        response = await client.get(f"{API_BASE_URL}/{endpoint}", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return None

def trigger_simulation(payload: dict) -> dict | None:
    """Synchronous courier for the Simulator POST request."""
    with httpx.Client() as client:
        try:
            # Note: We use client.post, not await client.post
            response = client.post(f"{API_BASE_URL}/inference/simulate", json=payload, timeout=45.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return None

def generate_gp_paths(data: dict | None, num_scribbles=12) -> go.Figure:
    """Generates the Single-Lender GP Manifold with Apple-style high-clarity styling."""
    fig = go.Figure()
    
    if data is None or 'relative_trajectory' not in data:
        return get_ghost_figure()

    # 1. DATA EXTRACTION
    mean = np.array(data['relative_trajectory'])
    std = np.array(data['std']) 
    steps = list(range(len(mean)))
    samples = data.get('samples', [])
    k_sigma = float(data.get('sigma_scale', 0.001))
    sigma_label = data.get('sigma_label', '1') 
    
    display_std = std * k_sigma

    # 2. THE CONFIDENCE WASH (Subtle Blue)
    fig.add_trace(go.Scatter(
        x=list(steps) + list(steps)[::-1],
        y=list(mean + display_std) + list(mean - display_std)[::-1],
        fill='toself', 
        fillcolor='rgba(0, 113, 227, 0.06)', 
        line=dict(color='rgba(255,255,255,0)'), 
        name=f'{sigma_label}σ Confidence Interval',
        showlegend=False, 
        hoverinfo='skip'
    ))

    # 3. THE SCRIBBLES (Latent Sample Paths)
    if len(samples) > 0:
        actual_scribbles = min(len(samples), num_scribbles)
        for i in range(actual_scribbles):
            fig.add_trace(go.Scatter(
                x=steps, 
                y=np.array(samples[i]).flatten() / 100, 
                mode='lines',
                line=dict(width=1, color='rgba(175, 82, 222, 0.2)', shape='spline'),
                showlegend=False,
                visible=False, # <--- THE FIX: Hidden by default so they don't blow out the Y-axis
                hoverinfo='skip' 
            ))

    # 4. THE POSTERIOR MEAN
    fig.add_trace(go.Scatter(
        x=steps, 
        y=mean, 
        mode='lines+markers', 
        name='Expected Rejection Trajectory (μ)',
        line=dict(color='#0071e3', width=3.5, shape='spline'),
        marker=dict(size=5, color='#0071e3', line=dict(color='white', width=1)),
        hovertemplate="<b>Step %{x}</b><br>Mean Δ: %{y:.4f}<extra></extra>"
    ))

    # 5. BUTTONS & LAYOUT
    total_traces = len(fig.data)
    show_all = [True] * total_traces 
    # Create the boolean mask: Trace 0 is wash (True), middle traces are scribbles (False), last trace is mean (True)
    hide_scribbles = [True] + [False] * (total_traces - 2) + [True]

    fig.update_layout(
        **BESPOKE_LAYOUT,
        title={
            'text': f"<b>GP Posterior Manifold</b> <span style='color:#86868b; font-weight:normal'>({sigma_label}σ Scaling)</span>",
            'y': 0.95, 
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': '#1d1d1f'}
        },
        updatemenus=[dict(
            type="buttons",
            direction="right", # <--- THE FIX: Side-by-side layout
            active=1,               # <--- THE FIX: Defaults the active state to the "Hide" button
            x=0.5,                  # Center horizontally
            y=-0.15,                # Push neatly BELOW the x-axis
            xanchor='center',
            yanchor='top',
            buttons=[
                dict(label="Show Sample Paths", method="update", args=[{"visible": show_all}]),
                dict(label="Hide Sample Paths", method="update", args=[{"visible": hide_scribbles}])
            ],
            bgcolor="rgba(255, 255, 255, 0.7)", 
            bordercolor="rgba(0, 0, 0, 0.05)",  
            font=dict(color="#1d1d1f", size=10)
        )]
    )
    
    fig.update_xaxes(title_text=X_TEXT)
    fig.update_yaxes(title_text=Y_TEXT)

    return fig


def build_simulation_trajectory(data: dict | None) -> go.Figure:
    fig = go.Figure()
    
    if not data or not data.get("relative_trajectory"):
        fig.update_layout(title="Awaiting Simulation...", **BESPOKE_LAYOUT)
        return fig
        
    traj = data["relative_trajectory"]
    steps = list(range(len(traj)))
    
    final_step = steps[-1]
    final_val = traj[-1] 
    
    # 3. Format the text dynamically as percentages (e.g., +15.20%)
    final_mean_text = f"{final_val:+.2%}" 
    std_val = data.get('final_std', 0) 
    final_sd_text = f"± {std_val:.2%}" 
    
    fig.add_trace(go.Scatter(
        x=steps, y=traj, mode='lines+markers', name="Relative Divergence",
        line=dict(color='#ff7b72', width=3, shape='spline'), 
        fill='tozeroy', fillcolor='rgba(255, 123, 114, 0.1)'
    ))
    
    combo_text = (
        f"<b>Final Divergence</b><br>"
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
        ax=-80, 
        ay=-50, 
        font=dict(family="Inter, sans-serif", color="white"),
        align="left",
        bgcolor="rgba(15, 15, 15, 0.9)", 
        bordercolor="#ff7b72", 
        borderwidth=1,
        borderpad=8
    )
    fig.update_layout(title="Relative Rejection Divergence", **BESPOKE_LAYOUT)
    fig.update_yaxes(
        title_text="Divergence from Day 0", 
        tickformat='.1%',
        zeroline=True,
        zerolinecolor='#8b949e',
        zerolinewidth=2
    )
    
    return fig


def get_simulator_layout():
    """Returns the heavily optimized, elegant Fairness Simulator."""
    
    LABEL_STYLE = {
        'color': '#8b949e', 'fontSize': '11px', 'textTransform': 'uppercase', 
        'letterSpacing': '1px', 'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'
    }

    CARD_STYLE = {
        'backgroundColor': 'rgba(255, 255, 255, 0.8)', 
        'backdropFilter': 'blur(20px)',
        'border': '1px solid rgba(0, 0, 0, 0.05)', 
        'padding': '30px', 
        'borderRadius': '18px',
        'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.05)'
    }

    SECTION_TITLE = {
        'color': '#1d1d1f', 'fontSize': '16px', 'fontWeight': '600', 
        'marginBottom': '20px', 'letterSpacing': '-0.02em'
    }

    SLIDER_KWARGS = {
        'tooltip': {"placement": "bottom", "always_visible": False},
        'className': 'custom-slider' 
    }
    
    return html.Div(className='main-dashboard-container', style={'display': 'flex', 'gap': '20px', 'alignItems': 'stretch'}, children=[
        
        # ==========================================
        # LEFT COLUMN: INPUTS & EXECUTION
        # ==========================================
        html.Div(className='mobile-fluid-col', style={**CARD_STYLE, 'flex': '0 0 360px', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px', 'padding': '20px'}, children=[
            html.Div([
                html.Div("Borrower Default Risk Parameters", style=SECTION_TITLE),
                html.Label("Business Tenure (Months)", style=LABEL_STYLE), 
                dcc.Slider(id='sim-tenure', min=MINTEN, max=MAXTEN, value=MEAN_TENURE, step=6, marks=None, **SLIDER_KWARGS),
                
                html.Label("Number of Simulated Credit Applications", style=LABEL_STYLE), 
                dcc.Slider(id='sim-horizon', min=16, max=128, value=32, step=16, marks=None, allow_direct_input=False, **SLIDER_KWARGS),
                html.Div(style={'height': '15px'}),
                
                html.Label("Amount Sought ($)", style=LABEL_STYLE), 
                dcc.Slider(id='sim-amount', min=MINASO, max=MAXASO, value=SOUGHT_MEAN, step=50000, marks=None, allow_direct_input=False, **SLIDER_KWARGS),
            ]),

            html.Hr(style={'borderColor': '#21262d', 'margin': '5px 0'}),
            
            html.Div([
                html.Div("Lender Proxy Bias Parameters", style=SECTION_TITLE),
                html.Label("Geographic Disadvantage Proxy", style=LABEL_STYLE), 
                dcc.Slider(id='sim-animus', min=MINAN+EPS, max=MAXAN-EPS, value=MEAN_ANIMUS, step=0.1, marks=None,  allow_direct_input=False, **SLIDER_KWARGS),
                
                html.Label("Community Disadvantage Proxy", style=LABEL_STYLE), 
                dcc.Slider(id='sim-isolation', min=MINISO+EPS, max=MAXISO-EPS, value=MEAN_ISO, step=0.1, marks=None, allow_direct_input=False, **SLIDER_KWARGS),
                
                html.Label("Systemic Bias Proxy", style=LABEL_STYLE), 
                dcc.Slider(id='sim-iat', min=MINIAT+EPS, max=MAXIAT-EPS, value=MEAN_IAT, step=0.1, marks=None, allow_direct_input=False, **SLIDER_KWARGS),
            ]),

            # 3. Lender Selection & Toggle
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '12px'}, children=[
                dcc.Checklist(
                    id='sim-view-selected-only',
                    options=[{'label': 'View Single Lender Simulations', 'value': 'selected'}],
                    value=[], #--Empty list means unchecked by default (showing multilender view)
                    className='clean-checkbox'
                ),
                html.Div([
                    html.Label("Lender Type", style=LABEL_STYLE),
                    dcc.RadioItems(
                        id='sim-lender',
                        options=[
                            {'label': 'BANK', 'value': 'bank'}, 
                            {'label': 'FINTECH', 'value': 'fintech'},
                            {'label': 'CREDIT UNION', 'value': 'creditunion'},
                            {'label': 'CDFI', 'value': 'cdfi'},
                            {'label': 'MDI', 'value': 'mdi'},
                            {'label': 'MCA', 'value': 'factoringccmca'}
                        ],
                        value='fintech',
                        labelClassName="lender-toggle-btn",
                        className="lender-toggle-container"
                    ),
                ])
            ]),
            
            # 4. PRIMARY TRIGGER (Now cleanly anchored at the bottom of the input list)
            html.Div(style={'marginTop': 'auto', 'paddingTop': '20px'}, children=[ 
                html.Button(
                    "Simulate Approval Trajectories", 
                    id='sim-submit-btn', 
                    n_clicks=0, 
                    style={
                        'width': '100%', 
                        'height': '60px', 
                        'fontSize': '16px', 
                        'background': 'linear-gradient(to right, #8B5CF6, #3B82F6)', 
                        'color': 'white',
                        'border': 'none', 
                        'borderRadius': '8px', 
                        'cursor': 'pointer', 
                        'fontWeight': '800', 
                        'letterSpacing': '1.5px', 
                        'textTransform': 'uppercase',
                        'boxShadow': '0 8px 25px rgba(139, 92, 246, 0.4)', 
                        'transition': 'transform 0.2s ease, box-shadow 0.2s ease'
                    }
                )
            ]),
                html.Button(
                    "ⓘ About the Engine", 
                    id="open-about-modal", 
                    n_clicks=0,
                    style={
                        'background': 'transparent', 'border': 'none', 'color': '#8b949e', 
                        'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600',
                        'display': 'flex', 'alignItems': 'center', 'gap': '5px'
                    }
                )
        ]),

        # ==========================================
        # RIGHT COLUMN: VISUALIZATION & CHART CONTROLS
        # ==========================================
        html.Div(style={**CARD_STYLE, 'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'padding': '20px'}, children=[
            html.Div(style={
                'display': 'flex', 
                'justifyContent': 'flex-end', 
                'alignItems': 'center', 
                'gap': '30px',
                'marginBottom': '15px',
                'paddingRight': '20px' 
            }, children=[
                
                html.Div(style={'width': '180px'}, children=[ 
                    dcc.Tabs(
                        id='sim-sigma-level', 
                        value='0.00196', 
                        parent_className='ios-tabs-container',
                        children=[
                            dcc.Tab(label='1σ', value='0.001', className='ios-tab', selected_className='ios-tab--active'),
                            dcc.Tab(label='2σ', value='0.00196', className='ios-tab', selected_className='ios-tab--active'),
                            dcc.Tab(label='3σ', value='0.00258', className='ios-tab', selected_className='ios-tab--active'),
                        ]
                    )
                ])
            ]),
            dcc.Graph(
                id='sim-output-graph', 
                figure=get_ghost_figure(), 
                style={'height': '550px'}, 
                config={'displayModeBar': False}, 
                className="animate-trajectory"
            ),
            
            html.Div(id='sim-output-text', style={
                'textAlign': 'center', 'color': '#8b949e', 'fontSize': '13px', 'paddingTop': '10px'
            })
        ])
    ])

#============
#-applayout-#
#============

LABEL_STYLE = {
    'color': '#8b949e', 'fontSize': '11px', 'textTransform': 'uppercase', 
    'letterSpacing': '1px', 'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'
}
CARD_STYLE = {
    'backgroundColor': 'rgba(255, 255, 255, 0.8)', 
    'backdropFilter': 'blur(20px)',
    'border': '1px solid rgba(0, 0, 0, 0.05)', 
    'padding': '30px', 
    'borderRadius': '18px',
    'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.05)'
}
SECTION_TITLE = {
    'color': '#1d1d1f', 'fontSize': '16px', 'fontWeight': '600', 
    'marginBottom': '20px', 'letterSpacing': '-0.02em'
}
SLIDER_KWARGS = {
    'tooltip': {"placement": "bottom", "always_visible": False},
    'className': 'custom-slider'
}
app.layout = html.Div(style={
    'backgroundColor': '#f5f5f7', 
    'minHeight': '100vh', 
    'padding': '30px 60px 150px 60px',
    'color': '#1d1d1f',
    'fontFamily': '-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif'
}, children=[
    
    # --- HEADER BLOCK ---
    html.Div(style={
        'display': 'flex', 
        'alignItems': 'center', 
        'gap': '20px', 
        'marginBottom': '40px', 
        'borderBottom': '1px solid #d2d2d7', # Light gray border
        'paddingBottom': '20px'
    }, children=[
        
        # 1. PRIMARY LOGO (Solid Gradient - Glow Removed)
        html.Div(className="logo-box", style={
            'width': '35px', 
            'height': '35px', 
            'borderRadius': '8px'
        }),
        
        # --- LOGO & TITLE GROUP ---
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '20px'}, children=[
            
            # 2. SECONDARY LOGO (Procedural Manifold)
            html.Div(style={
                'width': '45px', 
                'height': '45px', 
                'position': 'relative', 
                'borderRadius': '10px', 
                'background': 'linear-gradient(135deg, #1f6feb, #8957e5)',
                'boxShadow': '0 0 15px rgba(31, 111, 235, 0.4)', 
                'display': 'flex', 
                'alignItems': 'center', 
                'justifyContent': 'center', 
                'overflow': 'hidden'
            }, children=[
                html.Div(style={'width': '120%', 'height': '60%', 'borderTop': '3px solid white', 'borderRadius': '50%', 'position': 'absolute', 'top': '30%', 'opacity': '0.8'}),
                html.Div(style={'width': '120%', 'height': '60%', 'borderTop': '2px solid rgba(255,255,255,0.4)', 'borderRadius': '50%', 'position': 'absolute', 'top': '45%'})
            ]),

            # 3. TYPOGRAPHY (Title & Subtitle)
            html.Div(children=[
                html.H1(children=[
                    html.Span("deep", style={'fontWeight': '700', 'color': '#1d1d1f', 'letterSpacing': '-1px'}), 
                    html.Span("kernels", style={'fontWeight': '300', 'color': '#0071e3'})
                ], style={'fontSize': '32px', 'margin': '0', 'fontFamily': 'monospace'}),
                
                html.Div("RISK SIMULATION ENGINE", 
                    style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '2px', 'marginTop': '4px', 'fontWeight': '600'})
            ]),
            
            # 4. ABOUT MODAL OVERLAY (Hidden by default)
            html.Div(
                id="about-modal",
                style={
                    'display': 'none', 
                    'position': 'fixed', 
                    'top': '0', 
                    'left': '0', 
                    'width': '100%', 
                    'height': '100%',
                    'backgroundColor': 'rgba(13, 17, 23, 0.7)', 
                    'backdropFilter': 'blur(5px)',
                    'zIndex': '9999', 
                    'justifyContent': 'center', 
                    'alignItems': 'center'
                },
                children=[
                    html.Div(style={
                        **CARD_STYLE, 
                        'width': '600px', 
                        'maxWidth': '90%', 
                        'maxHeight': '80vh', 
                        'overflowY': 'auto',
                        'position': 'relative'
                    }, children=[
                        # Close Button
                        html.Button(
                            "✕", id="close-about-modal", n_clicks=0,
                            style={
                                'position': 'absolute', 'top': '20px', 'right': '20px', 
                                'background': 'transparent', 'border': 'none', 'color': '#8b949e',
                                'fontSize': '18px', 'cursor': 'pointer'
                            }
                        ),
                        
                        html.H2("About deepkernels", style={'color': '#1d1d1f', 'marginTop': '0'}),

                        html.P(
                            "This is the interface for a custom quantitative engine designed to scale Bayesian inference using stochastic simulation. "
                            "It serves as an algorithmic auditing simulation tool to quantify and expose systemic lending biases in credit markets",
                            style={'color': '#475569', 'lineHeight': '1.6', 'fontSize': '14px', 'marginBottom': '12px'}
                        ),

                        html.P(
                            "The underlying architecture leverages GPU-accelerated Multitask Gaussian Process Regression to model both volatile local dynamics and deterministic global trends. "
                            "By incorporating proxy variables derived from demographic data, it mathematically maps the socioeconomic disparities embedded in US credit markets.",
                            style={'color': '#475569', 'lineHeight': '1.6', 'fontSize': '14px', 'marginBottom': '12px'}
                        ),

                        html.P(
                            "By generating distributional trajectories across unsupervised latent groups of borrowers, "
                            "the deep kernel captures exactly how institutional credit risk models and their inherent biases evolve over repeated application attempts (simulated time steps).",
                            style={'color': '#475569', 'lineHeight': '1.6', 'fontSize': '14px', 'marginBottom': '12px'}
                        ),

                        html.P([
                            "The ", html.Strong("deepkernels v1.0", style={'color': '#1d1d1f'}), " engine is currently powering the Algorithmic Auditing Simulator on the main dashboard. "
                            "It measures dynamic changes in the topology of the probability space with respect to each selectable lender type—hence ",
                            html.A("topologicaldisparity.com", href="https://topologicaldisparity.com", target="_blank", style={'color': '#0071e3', 'textDecoration': 'none'}),
                            ". See the GitHub repository for core model architecture."
                        ], style={'color': '#475569', 'lineHeight': '1.6', 'fontSize': '14px'}),
                        
                        
                        html.Hr(style={'borderColor': 'rgba(0,0,0,0.1)', 'margin': '25px 0'}),
                        
                        html.H3("Developer Contact", style={'color': '#1d1d1f', 'fontSize': '15px'}),
                        html.P("Liam Douglas Giles", style={'fontWeight': '600', 'color': '#1d1d1f', 'margin': '5px 0'}),
                        html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px', 'marginTop': '10px'}, children=[
                            html.A("linkedin.com/in/liamdouglasgiles", href="https://www.linkedin.com/in/liamdouglasgiles", target="_blank", style={'color': '#3B82F6', 'textDecoration': 'none', 'fontSize': '14px'}),
                            html.A("github.com/liamdgee/deepkernels", href="https://github.com/liamdgee/deepkernels", target="_blank", style={'color': '#3B82F6', 'textDecoration': 'none', 'fontSize': '14px'})
                        ])
                    ])
                ]
            )
        ])
    ]),

    html.Div(get_simulator_layout()) 
])


# ==========================================
# 5. CALLBACKS
# ==========================================

@app.callback(
    Output("about-modal", "style"),
    [Input("open-about-modal", "n_clicks"),
     Input("close-about-modal", "n_clicks")],
    [State("about-modal", "style")]
)
def toggle_modal(open_clicks, close_clicks, current_style):
    ctx = callback_context
    if not ctx.triggered:
        return current_style
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "open-about-modal":
        current_style["display"] = "flex"
    elif button_id == "close-about-modal":
        current_style["display"] = "none"
        
    return current_style

@app.callback(
    [Output('sim-output-graph', 'figure'), Output('sim-output-text', 'children')],
    [Input('sim-submit-btn', 'n_clicks')],
    [
        State('sim-sigma-level', 'value'),
        State('sim-horizon', 'value'),
        State('sim-tenure', 'value'),
        State('sim-amount', 'value'), 
        State('sim-lender', 'value'), 
        State('sim-animus', 'value'),
        State('sim-isolation', 'value'),
        State('sim-iat', 'value'),
        State('sim-view-selected-only', 'value') # <-- FIX 1: Updated to the new ID
    ],
    prevent_initial_call=True
)
def run_simulation(n_clicks, sigma_val, horizon_steps, tenure, amount, lender, animus, isolation, iat, view_selected_val):
    is_compare = 'selected' not in (view_selected_val or [])
    status_msg = "Running simulation..."
    payload = {
        "tenure_months": float(tenure),
        "amount_sought": float(amount),
        "lender_type": str(lender).lower(),
        "animus_proxy": float(np.clip(animus, 1.5, 5.5)),
        "isolation_proxy": float(np.clip(isolation, 1.5, 5.5)), 
        "iat_score": float(np.clip(iat, 1.5, 5.5)), 
        "horizon_steps": int(horizon_steps),
        "compare_all_lenders": is_compare    
    }
    # --- THE FIX: Clean Map for the UI Labels ---
    k_sigma = float(sigma_val) if sigma_val is not None else 0.001
    sigma_map = {
        '0.001': '1',
        '0.00196': '2',
        '0.00258': '3'
    }
    clean_sigma_label = sigma_map.get(str(sigma_val), '1') 

    data = trigger_simulation(payload)
    # --- DEFINE APPLE WHITE PALETTE ---
    PRIMARY_BLUE = '#0071e3'
    LILAC_ALT = 'rgba(175, 82, 222, 0.25)' # Vibrant but translucent purple
    WASH_COLOR = 'rgba(0, 113, 227, 0.08)' # Subtlest blue for CI
    GRID_COLOR = '#e8e8ed'  
    AXIS_LABEL_COLOR = '#86868b'  
    ZERO_LINE_COLOR = '#d2d2d7'   
    TEXT_COLOR = '#1d1d1f'
    
    if not data:
        return go.Figure().update_layout(title="❌ API CONNECTION FAILED", **BESPOKE_LAYOUT), "Check Port 8000."
    fig = go.Figure()
    
    if is_compare and isinstance(data, dict):
        

        for lender_name, metrics in data.items():
            current_lender_key = str(lender_name).strip().lower()
            selected_lender_val = str(lender).strip().lower()
            is_selected = (current_lender_key == selected_lender_val)
            
            rel_traj = metrics.get('relative_trajectory', [])
            std_hist = metrics.get('std_history', [])
            if not rel_traj:
                continue
                
            steps = list(range(len(rel_traj)))
            mu = np.array(rel_traj)
            std = np.array(std_hist)
            
            if is_selected and len(std) > 0:
                display_std = std * k_sigma
                fig.add_trace(go.Scatter(
                    x=steps + steps[::-1],
                    y=list(mu + display_std) + list(mu - display_std)[::-1],
                    fill='toself',
                    fillcolor=WASH_COLOR, 
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{clean_sigma_label}σ Confidence', # The name won't matter anymore
                    showlegend=False,
                    legendgroup="uncertainty",
                    hoverinfo='skip'
                ))

            # 2. THE TRAJECTORY LINE
            fig.add_trace(go.Scatter(
                x=steps, y=mu,
                name=lender_name.replace('factoringccmca', 'MCA').upper(),
                mode='lines',
                line=dict(
                    width=4 if is_selected else 1.2,
                    color=PRIMARY_BLUE if is_selected else LILAC_ALT,
                    shape='spline'
                ),
                legendrank=1 if is_selected else 100 
            ))
            
            display_name = lender.replace('factoringccmca', 'MCA').upper()
            if is_selected:
                final_val = mu[-1]
                fig.add_annotation(
                    x=steps[-1],
                    y=final_val,
                    text=f"<b>{display_name}</b><br>{final_val:+.2%}",
                    showarrow=True,            
                    arrowhead=0,
                    ax=45,
                    ay=0,
                    font=dict(size=10, color='#1d1d1f'),
                    bgcolor='#e8e8ed',
                    bordercolor='#d2d2d7',
                    borderwidth=1,
                    borderpad=5,
                    opacity=0.95
                )
            

            status_msg = f"Real-time Probabilistic Inference: Epistemic Uncertainty Estimates for {display_name}"
        
        # --- CORRECTED UPDATE BLOCK ---
        fig.update_layout(
            **BESPOKE_LAYOUT,
            title={
                'text': f"<b>Projected Loan Rejection Rates</b>" if is_compare else f"<b>Projected Exposure: {lender.upper()}</b>",
                'y': 0.95, 
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20, 'color': '#1d1d1f'}
            }
        )
   
    else:
        single_lender_data = list(data.values())[0] if isinstance(data, dict) else data
        lender_key = list(data.keys())[0]
        single_lender_data = data[lender_key]
        raw_samples = single_lender_data.get('samples', [])
        
        gp_manifold_data = {
            "relative_trajectory": single_lender_data.get('relative_trajectory', []),
            "std": single_lender_data.get('absolute_std', []), # Use absolute_std for the manifold wash
            "samples": np.array(raw_samples),
            "sigma_scale": k_sigma,
            "sigma_label": clean_sigma_label 
        }
        
        fig = generate_gp_paths(gp_manifold_data)
        
        
        rel_traj = single_lender_data.get('relative_trajectory', [0])
        rel_change = rel_traj[-1] if len(rel_traj) > 0 else 0
        status_msg = f"Final Marginal Exposure: {rel_change:+.2%} for {lender.upper()}"
        
        fig.update_layout(
            **BESPOKE_LAYOUT,
            title={
                'text': f"<b>Projected Loan Rejection Rates</b>",
                'y': 0.95, 
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20, 'color': '#1d1d1f'}
            }
        )
        
    fig.update_xaxes(
        title_text=X_TEXT,
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(size=11, color=AXIS_LABEL_COLOR),
        tickfont=dict(color=AXIS_LABEL_COLOR),
        linecolor=ZERO_LINE_COLOR
    )

    fig.update_yaxes(
        title_text=Y_TEXT, 
        tickformat='.1%',
        zeroline=True,
        zerolinecolor=ZERO_LINE_COLOR,
        zerolinewidth=1,
        gridcolor=GRID_COLOR,
        title_font=dict(size=11, color=AXIS_LABEL_COLOR),
        tickfont=dict(color=AXIS_LABEL_COLOR)
    )

    return fig, status_msg

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8050)