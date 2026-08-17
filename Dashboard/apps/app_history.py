import json
from pathlib import Path

import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

from app import app, theme_layout

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

HOUR_RANGES = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]


def get_history_index(city: str) -> list[dict]:
    """Load historical index from database with JSON fallback."""
    try:
        import sys

        root_path = str(ROOT_DIR)
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
        from core import db

        records = db.get_all_weekly_history(city)
        if records:
            return records
    except Exception:
        pass

    index_file = ROOT_DIR / city / "Data" / "History" / "history_index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def get_week_data(city: str, week_id: str) -> dict:
    """Load full historical record for a specific week from database."""
    try:
        import sys

        root_path = str(ROOT_DIR)
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
        from core import db

        doc = db.get_single_week_data(city, week_id)
        if doc:
            return doc
    except Exception:
        pass

    week_file = ROOT_DIR / city / "Data" / "History" / f"weekly_{week_id}.json"
    if week_file.exists():
        try:
            with open(week_file) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# Layout Definition
layout = html.Div(
    className="",
    children=[
        # ---- Executive Toolbar ----
        html.Div(
            className="modern-card no-hover",
            style={"padding": "1.25rem 1.5rem", "marginBottom": "1.5rem"},
            children=[
                html.Div(
                    className="flex-between flex-wrap",
                    children=[
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                html.H1(
                                    "Historical Analytics & Parameter Rotation",
                                    style={"fontSize": "1.6rem", "margin": 0},
                                ),
                                html.Span(
                                    [
                                        html.I(className="fa-solid fa-clock-rotate-left"),
                                        " WEEKLY ARCHIVE",
                                    ],
                                    className="badge-pill primary",
                                    style={"fontSize": "0.72rem"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                # City Selector Radio
                                dcc.RadioItems(
                                    id="history-city-toggle",
                                    options=[
                                        {"label": " Madrid EMT", "value": "Madrid"},
                                        {"label": " London TfL", "value": "London"},
                                    ],
                                    value="Madrid",
                                    className="custom-tab",
                                    style={
                                        "display": "flex",
                                        "gap": "1rem",
                                        "alignItems": "center",
                                        "fontSize": "0.9rem",
                                        "fontWeight": "600",
                                    },
                                ),
                                # Week Selector Dropdown
                                html.Div(
                                    style={"minWidth": "220px"},
                                    children=[
                                        dcc.Dropdown(
                                            id="history-week-dropdown",
                                            clearable=False,
                                            style={"fontSize": "0.88rem"},
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.P(
                    "Autonomous weekly model retraining runs every Monday at 00:00. Inspect nominal headway parameter drift (μ, Σ), QoS service regularity indices, telemetry collection health, and storage savings over time.",
                    style={
                        "color": "var(--text-muted)",
                        "fontSize": "0.92rem",
                        "margin": "0.75rem 0 0 0",
                    },
                ),
            ],
        ),
        # ---- Weekly Summary KPIs Row ----
        html.Div(
            className="grid-4",
            style={"marginBottom": "1.5rem"},
            children=[
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Weekly Telemetry Rows", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-database",
                                    style={"color": "var(--primary-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="hist-kpi-records", className="kpi-val", children="—"),
                        html.Span("Positions ingested", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Active Route Fleet", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-bus-simple",
                                    style={"color": "var(--accent-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="hist-kpi-fleet", className="kpi-val", children="—"),
                        html.Span("Unique buses monitored", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Service Regularity (QoS)", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-gauge-high", style={"color": "#10B981"}
                                ),
                            ],
                        ),
                        html.Div(id="hist-kpi-qos", className="kpi-val", children="—"),
                        html.Span("Weekly regularity score", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Disk Storage Saved", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-hard-drive",
                                    style={"color": "var(--warning-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="hist-kpi-storage", className="kpi-val", children="—"),
                        html.Span("Reclaimed via parameter distillation", className="kpi-sub"),
                    ],
                ),
            ],
        ),
        # ---- Multi-Week Trends & Analytics Grid ----
        html.Div(
            className="workspace-grid",
            style={"marginBottom": "1.5rem"},
            children=[
                # Chart 1: Multi-Week Headway Parameter Drift
                html.Div(
                    className="modern-card no-hover",
                    style={"padding": "1.25rem"},
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.6rem"},
                            children=[
                                html.H3(
                                    "Headway Parameter Drift (Multi-Week)",
                                    style={"fontSize": "1.1rem", "margin": 0},
                                ),
                                html.Span(
                                    "Nominal μ ± σ",
                                    className="badge-pill primary",
                                    style={"fontSize": "0.65rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="hist-drift-graph",
                            style={"height": "42vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                # Chart 2: Hourly Headway Profile
                html.Div(
                    className="modern-card no-hover",
                    style={"padding": "1.25rem"},
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.6rem"},
                            children=[
                                html.H3(
                                    "Time-of-Day Hourly Profile",
                                    style={"fontSize": "1.1rem", "margin": 0},
                                ),
                                html.Span(
                                    "Week Profile",
                                    className="badge-pill warning",
                                    style={"fontSize": "0.65rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="hist-hourly-graph",
                            style={"height": "42vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
        # ---- Anomaly & Collector Health Grid ----
        html.Div(
            className="workspace-grid",
            style={"marginBottom": "1.5rem"},
            children=[
                # Chart 3: Anomaly Distribution
                html.Div(
                    className="modern-card no-hover",
                    style={"padding": "1.25rem"},
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.6rem"},
                            children=[
                                html.H3(
                                    "Weekly Anomaly Incident Trends",
                                    style={"fontSize": "1.1rem", "margin": 0},
                                ),
                                html.Span(
                                    "Incident Volume",
                                    className="badge-pill danger",
                                    style={"fontSize": "0.65rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="hist-anom-graph",
                            style={"height": "38vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                # Chart 4: Collector Uptime & API Health
                html.Div(
                    className="modern-card no-hover",
                    style={"padding": "1.25rem"},
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.6rem"},
                            children=[
                                html.H3(
                                    "Data Collector Health & Ingestion Volume",
                                    style={"fontSize": "1.1rem", "margin": 0},
                                ),
                                html.Span(
                                    "API Health",
                                    className="badge-pill success",
                                    style={"fontSize": "0.65rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="hist-collector-graph",
                            style={"height": "38vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
        # ---- Fitted Gaussian Baseline Models Table ----
        html.Div(
            className="modern-card no-hover",
            style={"padding": "1.25rem"},
            children=[
                html.Div(
                    className="flex-between",
                    style={"marginBottom": "1rem"},
                    children=[
                        html.H3(
                            "Fitted Baseline Gaussian Parameters (Current Model)",
                            style={"fontSize": "1.15rem", "margin": 0},
                        ),
                        html.Span("Active Rotation", className="badge-pill primary"),
                    ],
                ),
                html.Div(id="hist-models-table-container"),
            ],
        ),
    ],
)


# CALLBACK: Populate week dropdown based on city selection
@app.callback(
    [Output("history-week-dropdown", "options"), Output("history-week-dropdown", "value")],
    [Input("history-city-toggle", "value")],
)
def update_week_options(city: str):
    history_index = get_history_index(city)
    if not history_index:
        return [[{"label": "No history records found", "value": "none"}], "none"]

    options = []
    for entry in history_index:
        wid = entry.get("week_id", "")
        ts = entry.get("timestamp", "")[:10]
        records = entry.get("total_records", 0)
        label = f"{wid} (Started {ts}) — {records:,} records"
        options.append({"label": label, "value": wid})

    default_value = options[0]["value"] if options else "none"
    return [options, default_value]


# CALLBACK: Update KPIs and Visualizations
@app.callback(
    [
        Output("hist-kpi-records", "children"),
        Output("hist-kpi-fleet", "children"),
        Output("hist-kpi-qos", "children"),
        Output("hist-kpi-storage", "children"),
        Output("hist-drift-graph", "figure"),
        Output("hist-hourly-graph", "figure"),
        Output("hist-anom-graph", "figure"),
        Output("hist-collector-graph", "figure"),
        Output("hist-models-table-container", "children"),
    ],
    [
        Input("history-city-toggle", "value"),
        Input("history-week-dropdown", "value"),
        Input("theme-store", "data"),
    ],
)
def update_history_dashboard(city: str, week_id: str, theme: str = "dark"):
    history_index = get_history_index(city)
    week_record = get_week_data(city, week_id) if week_id != "none" else {}
    stats_data = week_record.get("stats", {})
    models_data = week_record.get("models", {})

    # 1. KPIs
    records_str = f"{stats_data.get('total_records', 0):,}"
    fleet_str = f"{stats_data.get('fleet_size', 0)} buses"
    qos_val = stats_data.get("overall_qos", 92.0)
    qos_str = f"{qos_val}% Regular"
    storage_saved = stats_data.get("disk_space_saved_mb", 0.32)
    storage_str = f"{storage_saved:.2f} MB"

    # 2-6. Build all visualizations and table via helpers
    drift_fig = build_drift_figure(history_index, theme)
    hourly_fig = build_hourly_figure(stats_data, theme)
    anom_fig = build_anomaly_figure(history_index, theme)
    collector_fig = build_collector_figure(history_index, theme)
    models_table = build_models_table(models_data)

    return [
        records_str,
        fleet_str,
        qos_str,
        storage_str,
        drift_fig,
        hourly_fig,
        anom_fig,
        collector_fig,
        models_table,
    ]


def build_drift_figure(history_index, theme):
    """Multi-week headway parameter drift chart with error bands."""
    fig = go.Figure()
    if not history_index:
        return fig
    weeks = [e.get("week_id", "") for e in reversed(history_index)]
    all_lines = list(history_index[0].get("lines", {}).keys())[:4]
    line_colors = ["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"]

    for idx, line_id in enumerate(all_lines):
        means, stds = [], []
        for entry in reversed(history_index):
            ldata = entry.get("lines", {}).get(line_id, {})
            means.append(ldata.get("mean_headway", 360.0))
            stds.append(ldata.get("std_headway", 60.0))
        color = line_colors[idx % len(line_colors)]
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=means,
                mode="lines+markers",
                name=f"Line {line_id}",
                line={"width": 3, "color": color},
                error_y={
                    "type": "data",
                    "array": stds,
                    "visible": True,
                    "thickness": 1.5,
                    "width": 4,
                },
                text=[
                    f"Line {line_id}<br>Mean: {m:.0f}s<br>Std: ±{s:.0f}s"
                    for m, s in zip(means, stds)
                ],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        xaxis_title="ISO Calendar Week",
        yaxis_title="Headway (seconds)",
        **{k: v for k, v in theme_layout(theme).items() if k != "title"},
        title={"text": "Nominal Headway μ ± σ Evolution Across Weeks"},
    )
    return fig


def build_hourly_figure(stats_data, theme):
    """Time-of-day working-day headway profile."""
    fig = go.Figure()
    line_baselines = stats_data.get("lines", {})
    if not line_baselines:
        return fig
    hours = [f"{h[0]}-{h[1]}" for h in HOUR_RANGES]
    colors = ["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"]
    for idx, (line_id, ldata) in enumerate(list(line_baselines.items())[:4]):
        h_means = [ldata.get("hourly_baselines", {}).get(h, {}).get("mean", 360.0) for h in hours]
        fig.add_trace(
            go.Scatter(
                x=[f"{h.split('-')[0]}:00" for h in hours],
                y=h_means,
                mode="lines+markers",
                name=f"Line {line_id}",
                line={"width": 3, "color": colors[idx % 4], "shape": "spline"},
                text=[f"Line {line_id} @ {h}: {m:.0f}s" for h, m in zip(hours, h_means)],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Nominal Headway (s)",
        **{k: v for k, v in theme_layout(theme).items() if k != "title"},
        title={"text": "Time-of-Day Headway Baseline (Working Days)"},
    )
    return fig


def build_anomaly_figure(history_index, theme):
    """Weekly anomaly incidents + QoS regularity dual-axis chart."""
    fig = go.Figure()
    if not history_index:
        return fig
    weeks = [e.get("week_id", "") for e in reversed(history_index)]
    anom_counts = [e.get("anomalies_detected", 15) for e in reversed(history_index)]
    qos_scores = [e.get("overall_qos", 92.0) for e in reversed(history_index)]
    fig.add_trace(
        go.Bar(
            x=weeks, y=anom_counts, name="Flagged Anomalies", marker_color="#EF4444", opacity=0.85
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=qos_scores,
            name="QoS Regularity (%)",
            yaxis="y2",
            mode="lines+markers",
            line={"width": 3, "color": "#10B981"},
        )
    )
    fig.update_layout(
        xaxis_title="Week",
        yaxis={"title": "Anomalies Count"},
        yaxis2={
            "title": "QoS Regularity %",
            "overlaying": "y",
            "side": "right",
            "range": [70, 100],
        },
        legend={"orientation": "h", "y": -0.2},
        **{k: v for k, v in theme_layout(theme).items() if k not in ("title", "xaxis", "yaxis")},
        title={"text": "Weekly Anomaly Incidents & Regularity (QoS)"},
    )
    return fig


def build_collector_figure(history_index, theme):
    """Weekly ingestion volume + API health dual-axis chart."""
    fig = go.Figure()
    if not history_index:
        return fig
    weeks = [e.get("week_id", "") for e in reversed(history_index)]
    records_volume = [e.get("total_records", 0) for e in reversed(history_index)]
    api_rates = [e.get("api_success_rate", 99.4) for e in reversed(history_index)]
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=records_volume,
            name="Ingested Positions",
            marker_color="#8B5CF6",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=api_rates,
            name="API Success Rate (%)",
            yaxis="y2",
            mode="lines+markers",
            line={"width": 3, "color": "#06B6D4"},
        )
    )
    fig.update_layout(
        xaxis_title="Week",
        yaxis={"title": "Records Ingested"},
        yaxis2={"title": "API Health %", "overlaying": "y", "side": "right", "range": [95, 100]},
        legend={"orientation": "h", "y": -0.2},
        **{k: v for k, v in theme_layout(theme).items() if k not in ("title", "xaxis", "yaxis")},
        title={"text": "Weekly Ingestion Volume & API Health"},
    )
    return fig


def build_models_table(models_data):
    """Compact table of fitted Gaussian parameters per line/day/hour."""
    table_rows = []
    if models_data:
        for line_id, ldict in models_data.items():
            for day_type, ddict in ldict.items():
                for hr_key, hdict in ddict.items():
                    max_d = hdict.get("max_dim", 1)
                    m1 = hdict.get("1", {})
                    mu = m1.get("mean", "—")
                    std = m1.get("cov_matrix", "—")
                    mu = f"{mu:.1f}s" if isinstance(mu, float) else str(mu)
                    std = f"{std:.1f}s" if isinstance(std, float) else str(std)
                    table_rows.append(
                        {
                            "Line": str(line_id),
                            "Day Type": day_type,
                            "Hour Window": hr_key,
                            "Max Dim": max_d,
                            "1D Mean (μ)": mu,
                            "1D Std (σ)": std,
                        }
                    )

    if table_rows:
        return dash_table.DataTable(
            columns=[{"name": k, "id": k} for k in table_rows[0].keys()],
            data=table_rows[:40],
            page_action="native",
            page_size=8,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#1E293B", "color": "white", "fontWeight": "bold"},
            style_cell={"textAlign": "center", "padding": "8px"},
        )
    return html.P(
        "No model parameters found for this selection.", style={"color": "var(--text-muted)"}
    )
