import json

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

pd.options.mode.chained_assignment = None  # suppress chained assignment warnings

location = "Madrid"

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(rel_path):
    if rel_path.startswith("../"):
        rel_path = rel_path[3:]
    return os.path.join(ROOT_DIR, rel_path)


# 16-Color High-Contrast Deterministic Vehicle & Group Palette
BUS_PALETTE = [
    "#8B5CF6",  # Purple (Primary)
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#3B82F6",  # Bright Blue
    "#EF4444",  # Rose Red
    "#84CC16",  # Lime Green
    "#6366F1",  # Indigo
    "#14B8A6",  # Teal
    "#F97316",  # Tangerine
    "#A855F7",  # Violet
    "#0EA5E9",  # Sky Blue
    "#E11D48",  # Crimson
    "#22C55E",  # Green
    "#D946EF",  # Fuchsia
]

colors = BUS_PALETTE
colors2 = BUS_PALETTE


def get_bus_color(bus_id):
    """Deterministic, high-contrast color for an individual bus plate ID."""
    if not bus_id or str(bus_id).strip() in ("0", "nan", "None", ""):
        return "#64748B"
    s = str(bus_id).strip()
    val = sum(ord(c) * (31**i) for i, c in enumerate(s))
    return BUS_PALETTE[val % len(BUS_PALETTE)]


def get_group_color(bus_a, bus_b):
    """Deterministic color for a consecutive bus pair (headway group)."""
    sa, sb = str(bus_a).strip(), str(bus_b).strip()
    val = sum(ord(c) * 17 for c in sa) + sum(ord(c) * 31 for c in sb)
    return BUS_PALETTE[val % len(BUS_PALETTE)]


def str_to_int(s):
    """Deterministic string-to-int mapping."""
    return sum(ord(c) for c in str(s))


zooms = {
    "1": 12.6,
    "44": 11.9,
    "82": 11.5,
    "F": 12.9,
    "G": 12.9,
    "U": 12.9,
    "132": 11.7,
    "133": 11.1,
    "N2": 11.4,
    "N6": 11.4,
}

max_ttls = {
    "1": 2800,
    "44": 2500,
    "82": 2500,
    "F": 2500,
    "G": 2500,
    "U": 2500,
    "132": 2800,
    "133": 2200,
    "N2": 2500,
    "N6": 2500,
}


# Timestamp-based dataframe cache (avoids recomputing unchanged bursts)
_DF_CACHE: dict = {}
_LAST_SERIES_TS: dict = {}  # per-(graph,line) last seen timestamp

box_height = "33.3vh"

# WE LOAD THE DATA
stops = pd.read_csv(resolve_path(location + "/Data/Static/stops.csv"))
line_shapes = pd.read_csv(resolve_path(location + "/Data/Static/line_shapes.csv"))
with open(resolve_path(location + "/Data/Static/lines_dict.json")) as f:
    lines_dict = json.load(f)

# Models parameters dictionary
with open(resolve_path(location + "/Data/Anomalies/models_params.json")) as f:
    models_params_dict = json.load(f)

layout = html.Div(
    className="cockpit-view",
    children=[
        # ---- Executive Control & Parameters Toolbar ----
        html.Div(
            className="modern-card no-hover toolbar-card",
            style={
                "padding": "0.5rem 1rem 0.85rem 1rem",
                "marginBottom": "0.5rem",
                "overflow": "visible",
            },
            children=[
                # Top Row: Title, Route Pills, Action Buttons
                html.Div(
                    className="flex-between flex-wrap",
                    style={"gap": "0.6rem", "marginBottom": "0.3rem"},
                    children=[
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                html.H1(
                                    "Madrid Transit Monitor",
                                    style={"fontSize": "1.2rem", "margin": 0, "fontWeight": "700"},
                                ),
                                html.Span(
                                    [html.Span(className="pulse-indicator"), " LIVE"],
                                    className="badge-pill success",
                                    style={"fontSize": "0.68rem"},
                                ),
                                html.Span(
                                    id="tab-title" + location,
                                    style={
                                        "fontSize": "0.85rem",
                                        "color": "var(--text-muted)",
                                        "fontFamily": "var(--font-mono)",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                html.Div(
                                    id="route-pills-container" + location,
                                    className="flex-gap flex-wrap",
                                ),
                                html.Button(
                                    [html.I(className="fa-solid fa-rotate"), " Refresh"],
                                    className="btn-primary-gradient",
                                    id="update-button" + location,
                                    n_clicks=0,
                                    style={"padding": "0.3rem 0.75rem", "fontSize": "0.8rem"},
                                ),
                            ],
                        ),
                    ],
                ),
                # Bottom Row: Horizontal Sliders (Title/Label on the LEFT of the slider)
                html.Div(
                    style={
                        "borderTop": "1px solid var(--border-color)",
                        "paddingTop": "0.3rem",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "gap": "1.25rem",
                        "flexWrap": "wrap",
                    },
                    children=[
                        # Slider 1: Confidence (horizontal flex)
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "0.6rem",
                                "flex": "1 1 300px",
                                "maxWidth": "420px",
                            },
                            children=[
                                html.Label(
                                    [
                                        html.I(
                                            className="fa-solid fa-sliders",
                                            style={
                                                "marginRight": "0.3rem",
                                                "color": "var(--primary-color)",
                                            },
                                        ),
                                        "Confidence (1 - α):",
                                    ],
                                    style={
                                        "fontSize": "0.75rem",
                                        "fontWeight": "600",
                                        "whiteSpace": "nowrap",
                                        "minWidth": "125px",
                                    },
                                ),
                                html.Div(
                                    style={"flex": "1 1 0", "minWidth": "140px"},
                                    children=[
                                        dcc.Slider(
                                            id="conf-slider" + location,
                                            min=90,
                                            max=99.9,
                                            step=0.1,
                                            value=98,
                                            marks={90: "90%", 95: "95%", 98: "98%", 99.9: "99.9%"},
                                            className="compact-slider",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Slider 2: Size Threshold (horizontal flex)
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "0.6rem",
                                "flex": "1 1 280px",
                                "maxWidth": "380px",
                            },
                            children=[
                                html.Label(
                                    [
                                        html.I(
                                            className="fa-solid fa-filter",
                                            style={
                                                "marginRight": "0.3rem",
                                                "color": "var(--accent-color)",
                                            },
                                        ),
                                        "Filter Window (k):",
                                    ],
                                    style={
                                        "fontSize": "0.75rem",
                                        "fontWeight": "600",
                                        "whiteSpace": "nowrap",
                                        "minWidth": "115px",
                                    },
                                ),
                                html.Div(
                                    style={"flex": "1 1 0", "minWidth": "140px"},
                                    children=[
                                        dcc.Slider(
                                            id="size-th-slider" + location,
                                            min=1,
                                            max=10,
                                            step=1,
                                            value=3,
                                            marks={1: "1", 3: "3", 5: "5", 10: "10"},
                                            className="compact-slider",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Model Badge
                        html.Span(
                            [
                                html.I(
                                    className="fa-solid fa-chart-pie",
                                    style={"marginRight": "0.3rem"},
                                ),
                                "Gaussian d≤3",
                            ],
                            className="badge-pill primary",
                            style={"fontSize": "0.7rem", "whiteSpace": "nowrap"},
                        ),
                    ],
                ),
            ],
        ),
        # ---- Hidden State & Polling ----
        html.Div(id="hidden-div" + location, style={"display": "none"}),
        dcc.Interval(id="interval-component" + location, interval=5000, n_intervals=0),
        # ---- KPI Cards (5-Column Sleek Strip matching Grid) ----
        html.Div(
            className="grid-5",
            style={"marginBottom": "0.5rem"},
            children=[
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Active Fleet", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-bus",
                                    style={"color": "var(--primary-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="kpi-fleet" + location, className="kpi-val", children="—"),
                        html.Span("In active sequence", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Mean Headway", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-clock",
                                    style={"color": "var(--accent-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="kpi-headway" + location, className="kpi-val", children="—"),
                        html.Span("Avg gap between buses", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("QoS Regularity", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-gauge-high",
                                    style={"color": "#10B981"},
                                ),
                            ],
                        ),
                        html.Div(id="kpi-qos" + location, className="kpi-val", children="—"),
                        html.Span("Service regularity index", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Idle / Terminus", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-ban",
                                    style={"color": "var(--warning-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="kpi-filtered" + location, className="kpi-val", children="—"),
                        html.Span("Edge buses filtered", className="kpi-sub"),
                    ],
                ),
                html.Div(
                    className="kpi-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span("Anomalies", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-triangle-exclamation",
                                    style={"color": "var(--danger-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="kpi-anoms" + location, className="kpi-val", children="—"),
                        html.Span("Detected in window", className="kpi-sub"),
                    ],
                ),
            ],
        ),
        # ---- Top Workspace Grid: Map (50%) + Headway Corridor (50%) ----
        html.Div(
            className="workspace-grid",
            children=[
                html.Div(
                    className="modern-card no-hover",
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.25rem"},
                            children=[
                                html.H3(
                                    "Fleet Spatial Map",
                                    style={"fontSize": "0.95rem", "margin": 0},
                                ),
                                html.Span(
                                    "Live positions",
                                    className="badge-pill primary",
                                    style={"fontSize": "0.62rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="map" + location,
                            style={"height": "100%", "width": "100%"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    className="modern-card no-hover",
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.25rem"},
                            children=[
                                html.H3(
                                    "Headway Corridor",
                                    style={"fontSize": "0.95rem", "margin": 0},
                                ),
                                html.Span(
                                    "Bunching risk view",
                                    className="badge-pill warning",
                                    style={"fontSize": "0.62rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="flat-hws" + location,
                            style={"height": "100%", "width": "100%"},
                            figure=go.Figure(),
                            clear_on_unhover=True,
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
        # ---- Bottom Workspace Grid (2 Columns: 1D/2D Series (50%) + Mahalanobis/Anomalies (50%)) ----
        html.Div(
            className="workspace-grid",
            children=[
                # Bottom-Left Card: 1D Headway Series / 2D Dynamics
                html.Div(
                    className="modern-card no-hover",
                    children=[
                        dcc.Tabs(
                            id="tabs-series" + location,
                            value="ts1" + location,
                            className="minimal-tabs-container",
                            children=[
                                dcc.Tab(
                                    label="1D Headway Series (s)",
                                    value="ts1" + location,
                                    className="minimal-tab",
                                    selected_className="minimal-tab--selected",
                                ),
                                dcc.Tab(
                                    label="2D Dynamics (s)",
                                    value="ts2" + location,
                                    className="minimal-tab",
                                    selected_className="minimal-tab--selected",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1 1 0", "minHeight": "0", "width": "100%"},
                            children=[
                                dcc.Graph(
                                    id="time-series-hws" + location,
                                    style={"display": "block", "height": "100%", "width": "100%"},
                                    figure=go.Figure(),
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    id="2d-time-series-hws" + location,
                                    style={"display": "none"},
                                    figure=go.Figure(),
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                ),
                # Bottom-Right Card: Mahalanobis Distance / Anomaly Events
                html.Div(
                    className="modern-card no-hover",
                    children=[
                        dcc.Tabs(
                            id="tabs-anoms" + location,
                            value="md" + location,
                            className="minimal-tabs-container",
                            children=[
                                dcc.Tab(
                                    label="Mahalanobis Distance (σ)",
                                    value="md" + location,
                                    className="minimal-tab",
                                    selected_className="minimal-tab--selected",
                                ),
                                dcc.Tab(
                                    label="Anomaly Events",
                                    value="an" + location,
                                    className="minimal-tab",
                                    selected_className="minimal-tab--selected",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1 1 0", "minHeight": "0", "width": "100%"},
                            children=[
                                dcc.Graph(
                                    id="mdist-hws" + location,
                                    style={"display": "block", "height": "100%", "width": "100%"},
                                    figure=go.Figure(),
                                    config={"displayModeBar": False},
                                ),
                                html.Div(
                                    id="anom-hws-div" + location,
                                    style={"display": "none"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
