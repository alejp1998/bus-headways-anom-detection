import json
import math
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html
from numpy import cos, pi, sin
from scipy.stats.distributions import chi2

from app import app, theme_layout

pd.options.mode.chained_assignment = None  # suppress chained assignment warnings

location = "London"

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


max_ttls = {"18": 4800, "24": 7200, "25": 5500, "73": 7200}


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
                                    "London Transit Monitor",
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


def str_to_int(s):
    """Deterministically map a bus plate string to an integer for color assignment."""
    return sum(ord(c) for c in str(s))


def _read_db(name, line):
    """Fast database read for live telemetry (returns None when unavailable)."""
    import sys

    root_path = resolve_path(".")
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from core import db

    if name == "burst":
        df = db.get_latest_burst_df("London", str(line) if line else None)
    elif name == "hws_burst":
        df = db.get_latest_headways_df("London", str(line) if line else None)
    elif name == "series":
        df = db.get_series_df("London", str(line) if line else "25", dim=None, limit=500)
    elif name == "anomalies":
        df = db.get_anomalies_df("London", str(line) if line else "25", limit=100)
    else:
        return None
    return df if not df.empty else None


def _read_csv_fallback(name, line):
    """Legacy CSV fallback when the database is empty."""
    file_map = {
        "burst": "/Data/RealTime/buses_data_burst_cleaned.csv",
        "hws_burst": "/Data/RealTime/headways_burst.csv",
        "series": "/Data/RealTime/series.csv",
        "anomalies": "/Data/Anomalies/anomalies.csv",
    }
    rel = file_map.get(name)
    if rel is None:
        return pd.DataFrame()
    p = resolve_path(location + rel)
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, dtype={"line": "str"})
        if line and not df.empty:
            df = df[df["line"] == str(line)]
        return df
    except Exception:
        return pd.DataFrame()


_DB_TABLES = {
    "burst": "buses_burst",
    "hws_burst": "headways_burst",
    "series": "headways_series",
    "anomalies": "anomaly_events",
}


def _parse_hover_buses(hoverData, line):
    """Extract the bus pair highlighted in the corridor graph (None on failure)."""
    try:
        if "text" in hoverData["points"][0].keys():
            return [hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0]]
        hws_burst = read_df("hws_burst", line=line)
        dest = hoverData["points"][0]["y"][3:-1]
        x = hoverData["points"][0]["x"]
        direction = 1 if dest == lines_dict[line]["destinations"][1] else 2
        buses = hws_burst[
            (hws_burst.line == line)
            & (hws_burst.direction == direction)
            & (hws_burst.busB_ttls >= x)
        ].sort_values("busB_ttls")
        return [buses.busA.iloc[0], buses.busB.iloc[0]]
    except Exception:
        return None


def _get_hour_range_and_model(line, dim):
    """Resolve the current day-type/hour-window model baseline (always active 24/7 with nearest fallback)."""
    now = dt.now()
    day_type = "LA" if now.weekday() <= 4 else ("SA" if now.weekday() == 5 else "FE")
    hour = now.hour
    if hour < 7:
        hour_range = "7-9"
    elif hour >= 23:
        hour_range = "21-23"
    else:
        hour_ranges = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]
        for h_range in hour_ranges:
            if h_range[0] <= hour < h_range[1]:
                hour_range = f"{h_range[0]}-{h_range[1]}"
                break
        else:
            hour_range = "21-23"

    try:
        models = models_params_dict.get(str(line), {}).get(day_type, {})
        if hour_range in models and str(dim) in models[hour_range]:
            return models[hour_range][str(dim)]
        for hr in ["21-23", "19-21", "17-19", "7-9"]:
            if hr in models and str(dim) in models[hr]:
                return models[hr][str(dim)]
    except Exception:
        pass
    return None


def _cached_figure(graph_key: str, line: str):
    """Return the cached figure for (graph, line) or a sentinel when it needs rebuilding.

    Returns (cached_figure, is_current):
      - (fig, True)  -> reuse the cached figure, no rebuild needed
      - (None, False)-> rebuild required
    """
    try:
        from core import db

        ts = db.get_latest_timestamp("London", "headways_series")
        if ts is None:
            return None, False
        # Ignore future timestamps (clock skew / stale test data) - never freeze on them
        try:
            ts_dt = dt.fromisoformat(str(ts))
            if ts_dt > dt.now():
                return None, False
        except Exception:
            pass
        cached = _LAST_SERIES_TS.get((graph_key, line))
        if cached and cached[0] == ts and cached[1] is not None:
            return cached[1], True
        return None, False
    except Exception:
        return None, False


def _store_figure(graph_key: str, line: str, figure):
    """Persist the freshly built figure together with its source timestamp."""
    try:
        from core import db

        ts = db.get_latest_timestamp("London", "headways_series")
        _LAST_SERIES_TS[(graph_key, line)] = (ts, figure)
    except Exception:
        pass


def read_df(name, line=None):
    """Read latest telemetry with timestamp-based memoization."""
    cache_key = f"{name}:{line}"
    try:
        from core import db

        table = _DB_TABLES.get(name)
        if table is not None:
            ts = db.get_latest_timestamp("London", table)
            cached = _DF_CACHE.get(cache_key)
            if cached and cached[0] == ts:
                return cached[1]
    except Exception:
        ts = None

    try:
        df = _read_db(name, line)
        if df is not None:
            if ts is not None:
                _DF_CACHE[cache_key] = (ts, df)
            return df
    except Exception:
        pass
    return _read_csv_fallback(name, line)


def ellipse(mus, cov_matrix, conf):
    a = cov_matrix[0][0]
    b = cov_matrix[0][1]
    c = cov_matrix[1][1]

    lambda1 = (a + c) / 2 + math.sqrt(((a - c) / 2) ** 2 + b**2)
    lambda2 = (a + c) / 2 - math.sqrt(((a - c) / 2) ** 2 + b**2)

    # Rotation angle
    if (b == 0) and (a >= c):
        theta = 0
    elif (b == 0) and (a < c):
        theta = math.pi / 2
    else:
        theta = math.atan2(lambda1 - a, b)

    # Eigenvectors
    ei_vecs = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]

    # Chi-Value for desired confidence
    chi_val = chi2.ppf(conf, df=2)

    # Eigenvalues
    r1 = math.sqrt(chi_val * lambda1)
    r2 = math.sqrt(chi_val * lambda2)
    ei_vals = [r1, r2]

    # CALCULATE ELLIPSE POINTS
    N = 100
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    t = np.linspace(0, 2 * pi, N)
    # ellipse parameterization with respect to a system of axes of directions a1, a2
    xs = ei_vals[0] * cos(t)
    ys = ei_vals[1] * sin(t)
    # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
    xp, yp = np.dot(ei_vecs, [xs, ys])
    x = xp + mus[0]
    y = yp + mus[1]
    return x, y


mapbox_access_token = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
mapbox_style = "streets" if mapbox_access_token else "carto-darkmatter"
mapbox_light_style = "light" if mapbox_access_token else "carto-positron"

zooms = {"18": 12.0, "24": 12.6, "25": 11.8, "73": 12.2}


def calc_map_params(line="25"):
    """Compute stable map center coordinates and zoom level for the route."""
    shapes = line_shapes.loc[line_shapes.line_sn.astype(str) == str(line)]
    if not shapes.empty:
        center_x = float(shapes.lon.mean())
        center_y = float(shapes.lat.mean())
    else:
        center_x, center_y = -0.1278, 51.5074
    zoom = float(zooms.get(str(line), 12.2))
    return center_x, center_y, zoom


def build_map(line_df, theme="dark", line="25"):
    """Build the interactive MapLibre/Scattermap route diagram with live buses."""
    dark = theme == "dark"
    map_style = mapbox_style if dark else mapbox_light_style
    center_x, center_y, zoom = calc_map_params(line)

    new_map = go.Figure()
    new_map.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"r": 0, "t": 0, "b": 0, "l": 0},
        showlegend=True,
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(15,23,42,0.75)" if dark else "rgba(255,255,255,0.85)",
        },
        uirevision=str(line),
        map={
            "bearing": 0,
            "center": {"lat": center_y, "lon": center_x},
            "pitch": 0,
            "zoom": zoom,
            "style": map_style,
            "uirevision": str(line),
        },
    )

    # Add stops
    st_line = (
        stops.loc[stops["line"].astype(str) == str(line)] if "line" in stops.columns else stops
    )
    if not st_line.empty:
        stop_color = "#64748B" if dark else "#94A3B8"
        new_map.add_trace(
            go.Scattermap(
                lat=st_line.lat,
                lon=st_line.lon,
                mode="markers",
                marker=go.scattermap.Marker(size=8, color=stop_color, opacity=0.55),
                text=st_line.name if "name" in st_line.columns else st_line.id,
                hoverinfo="text",
                name="Stops",
            )
        )

    # Add route shape lines (direction 1 = #6366F1, direction 2 = #F59E0B)
    for dir_val, color in [(1, "#6366F1"), (2, "#F59E0B")]:
        l_shape = line_shapes.loc[
            (line_shapes.line_sn.astype(str) == str(line)) & (line_shapes.direction == dir_val)
        ]
        if not l_shape.empty:
            l_shape = l_shape.sort_values("sequence")
            new_map.add_trace(
                go.Scattermap(
                    lat=l_shape.lat,
                    lon=l_shape.lon,
                    mode="lines",
                    line={"width": 3, "color": color},
                    text=f"Route {line} (Dir {dir_val})",
                    hoverinfo="skip",
                    opacity=0.9,
                    name=f"Dir {dir_val}",
                )
            )

    # Add live bus markers
    if not line_df.empty:
        for bus in line_df.itertuples():
            color = get_bus_color(bus.bus)
            new_map.add_trace(
                go.Scattermap(
                    lat=[bus.lat],
                    lon=[bus.lon],
                    mode="markers",
                    marker=go.scattermap.Marker(size=16, color=color, opacity=0.95),
                    text=[
                        f"<b>Bus {bus.bus}</b><br>ETA: {bus.estimateArrive}s<br>Stop: {bus.stop}"
                    ],
                    hoverinfo="text",
                    name=f"Bus {bus.bus}",
                    showlegend=False,
                )
            )

    return new_map


def _draw_corridor_rails(graph, dest1, dest2, max_x, track_color):
    """Draw background route rails for both directions."""
    for dest_label in [dest1, dest2]:
        graph.add_trace(
            go.Scatter(
                x=[0, max_x],
                y=[dest_label, dest_label],
                mode="lines",
                line={"width": 6, "color": track_color},
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _draw_headway_bridges(graph, dir_hws, dest_label):
    """Draw headway spacing bridges between consecutive buses on the corridor."""
    for i in range(len(dir_hws)):
        row = dir_hws.iloc[i]
        bus_a, bus_b = str(row.busA), str(row.busB)
        hw = float(row.headway)
        ttls_b = float(row.busB_ttls)
        ttls_a = (
            float(row.busA_ttls)
            if ("busA_ttls" in row and row.busA_ttls > 0)
            else max(0.0, ttls_b - hw)
        )

        if bus_a and bus_a != "0" and hw > 0:
            if hw < 120:
                bridge_color, status = "#EF4444", "⚠️ BUNCHING RISK"
            elif hw > 720:
                bridge_color, status = "#F59E0B", "⏳ SERVICE GAP"
            else:
                bridge_color, status = "#8B5CF6", "✅ REGULAR"

            hw_min = round(hw / 60.0, 1)
            mid_x = (ttls_a + ttls_b) / 2.0

            graph.add_trace(
                go.Scatter(
                    x=[ttls_a, ttls_b],
                    y=[dest_label, dest_label],
                    mode="lines",
                    line={"width": 5, "color": bridge_color},
                    hoverinfo="text",
                    text=f"<b>Headway: {hw:.0f}s ({hw_min} min)</b><br>Between: Bus {bus_a} → Bus {bus_b}<br>Status: {status}",
                    showlegend=False,
                )
            )
            graph.add_trace(
                go.Scatter(
                    x=[mid_x],
                    y=[dest_label],
                    mode="text",
                    text=[f"<b>{hw_min}m</b>"],
                    textposition="top center",
                    textfont={
                        "size": 10,
                        "color": bridge_color,
                        "family": "JetBrains Mono, monospace",
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )


def _draw_bus_nodes(graph, dir_hws, dest_label, dark, text_color):
    """Draw vehicle marker nodes with unique colors and tooltips."""
    for i in range(len(dir_hws)):
        row = dir_hws.iloc[i]
        bus_id = str(row.busB)
        if not bus_id or bus_id == "0":
            continue
        ttls = float(row.busB_ttls)
        hw = float(row.headway)
        bus_color = get_bus_color(bus_id)
        ttls_min = round(ttls / 60.0, 1)

        graph.add_trace(
            go.Scatter(
                x=[ttls],
                y=[dest_label],
                mode="markers+text",
                name=f"Bus {bus_id}",
                marker={
                    "size": 22,
                    "color": bus_color,
                    "line": {"color": "#FFFFFF" if dark else "#0F172A", "width": 2},
                    "symbol": "circle",
                },
                text=[f"<b>{bus_id}</b>"],
                textposition="bottom center",
                textfont={"size": 9.5, "color": text_color, "family": "Space Grotesk, sans-serif"},
                hoverinfo="text",
                hovertext=f"<b>Bus {bus_id}</b><br>Headway: {hw:.0f}s ({round(hw / 60, 1)} min)<br>TTLS: {ttls:.0f}s ({ttls_min} min to terminus)<br>Track: {dest_label}",
                showlegend=False,
            )
        )


def build_graph(line_hws, theme="dark"):
    """Build modern linear route stringline corridor with color-coded headway bridges and bus nodes."""
    headways = line_hws
    graph = go.Figure()
    dark = theme == "dark"

    if headways is None or headways.empty or "line" not in headways.columns:
        graph.update_layout(
            template="plotly_dark" if dark else "plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"r": 10, "l": 10, "t": 20, "b": 20},
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Waiting for active bus corridor telemetry...",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 13, "color": "#94A3B8"},
                }
            ],
        )
        return graph

    line = str(headways.line.iloc[0])
    destinations = lines_dict.get(line, {}).get("destinations", ["Inbound", "Outbound"])
    d1_full = destinations[0] if len(destinations) > 0 else "Outbound"
    d2_full = destinations[1] if len(destinations) > 1 else "Inbound"
    dest1 = f"Dir 1 ({d1_full[:14]})"
    dest2 = f"Dir 2 ({d2_full[:14]})"

    max_x = max_ttls.get(line, 4500)
    track_color = "rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.12)"
    grid_color = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.06)"
    text_color = "#F8FAFC" if dark else "#0F172A"

    graph.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"r": 20, "l": 80, "t": 15, "b": 35},
        uirevision=str(line),
        showlegend=False,
        hovermode="closest",
        xaxis={
            "title": {
                "text": "<b>Time to Terminal (TTLS)</b> → (Minutes from Arrival)",
                "font": {"size": 10, "color": "#94A3B8"},
            },
            "range": [-100, max_x + 200],
            "tickmode": "array",
            "tickvals": [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000],
            "ticktext": [
                "0m",
                "10m",
                "20m",
                "30m",
                "40m",
                "50m",
                "60m",
                "70m",
                "80m",
                "90m",
                "100m",
            ],
            "gridcolor": grid_color,
            "zeroline": True,
            "zerolinecolor": track_color,
            "tickfont": {"size": 9.5, "color": "#94A3B8", "family": "JetBrains Mono, monospace"},
        },
        yaxis={
            "type": "category",
            "categoryorder": "array",
            "categoryarray": [dest2, dest1],
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"size": 10.5, "color": text_color, "family": "Space Grotesk, sans-serif"},
        },
    )

    _draw_corridor_rails(graph, dest1, dest2, max_x, track_color)

    for dir_val, dest_label in [(1, dest1), (2, dest2)]:
        dir_hws = headways.loc[headways.direction == dir_val]
        if dir_hws.empty:
            continue
        dir_hws = dir_hws.sort_values("busB_ttls")
        _draw_headway_bridges(graph, dir_hws, dest_label)
        _draw_bus_nodes(graph, dir_hws, dest_label, dark, text_color)

    return graph


def build_time_series_graph(series_df, model, conf):
    graph = go.Figure()

    # Set title and clean layout (no cluttered 25-pair text wall)
    graph.update_layout(
        title=None,
        uirevision="ts1",
        showlegend=False,
        hovermode="closest",
        margin={"r": 15, "l": 45, "t": 10, "b": 22},
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "nticks": 8,
            "tickfont": {"size": 9.5, "family": "JetBrains Mono, monospace"},
        },
        yaxis={
            "title": {"text": "Headway", "font": {"size": 10, "color": "#94A3B8"}},
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "zeroline": True,
            "zerolinecolor": "darkgrey",
            "ticksuffix": "s",
            "tickfont": {"size": 9.5, "family": "JetBrains Mono, monospace"},
        },
    )

    series_df = series_df.loc[series_df.dim == 1]
    if series_df.shape[0] < 1:
        return graph

    min_time = series_df.datetime.min()
    max_time = series_df.datetime.max()

    dim = 1
    std = model["cov_matrix"]
    mean = model["mean"]
    m_th = math.sqrt(chi2.ppf(conf, df=dim))

    # Add statistical threshold lines (dashed red bounds)
    thresholds = [(mean - std * m_th), (mean + std * m_th)]
    for th in thresholds:
        graph.add_shape(
            name=f"Threshold ({conf * 100:.0f}%)",
            type="line",
            x0=min_time,
            y0=th,
            x1=max_time,
            y1=th,
            line={"color": "#EF4444", "width": 1.5, "dash": "dashdot"},
        )

    # Add group lines with matching colors
    for (b1, b2), group_df in series_df.groupby(["bus1", "bus2"], sort=False):
        group_df = group_df.sort_values("datetime")
        name = f"{b1}-{b2}"
        color = get_group_color(b1, b2)

        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.datetime,
                y=group_df.hw12,
                mode="lines+markers",
                line={"width": 2.5, "color": color},
                marker={"size": 5, "color": color},
                showlegend=False,
                text=[
                    f"<b>Group: {name}</b><br>Headway: {row.hw12:.0f}s ({round(row.hw12 / 60, 1)} min)<br>Time: {str(row.datetime)[:19]}"
                    for row in group_df.itertuples()
                ],
                hoverinfo="text",
            )
        )

    return graph


def build_2d_time_series_graph(series_df, model, conf):
    graph = go.Figure()

    graph.update_layout(
        title=None,
        uirevision="ts2",
        showlegend=False,
        hovermode="closest",
        margin={"r": 15, "l": 45, "t": 10, "b": 25},
        xaxis={
            "title": {"text": "Headway HW12", "font": {"size": 10, "color": "#94A3B8"}},
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "ticksuffix": "s",
            "tickfont": {"size": 9.5, "family": "JetBrains Mono, monospace"},
        },
        yaxis={
            "title": {"text": "Headway HW23", "font": {"size": 10, "color": "#94A3B8"}},
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "ticksuffix": "s",
            "tickfont": {"size": 9.5, "family": "JetBrains Mono, monospace"},
        },
    )

    series_df = series_df.loc[series_df.dim == 2]
    if series_df.shape[0] < 1:
        return graph

    # Add Gaussian confidence ellipse
    mus = [model["mean"][0], model["mean"][1]]
    cov_matrix = model["cov_matrix"]
    x, y = ellipse(mus, cov_matrix, conf)

    graph.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": "#EF4444", "width": 1.5, "dash": "dashdot"},
            name=f"{conf * 100:.1f}% Confidence Ellipse",
            hoverinfo="name",
            showlegend=False,
        )
    )

    for (b1, b2, b3), group_df in series_df.groupby(["bus1", "bus2", "bus3"], sort=False):
        group_df = group_df.sort_values("datetime")
        name = f"{b1}-{b2}-{b3}"
        color = get_group_color(b1, b2)

        # 1. Continuous dynamic trajectory line over time
        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.hw12,
                y=group_df.hw23,
                mode="lines+markers",
                line={"width": 2, "color": color},
                marker={"size": 4.5, "color": color},
                showlegend=False,
                text=[
                    f"<b>Triplet: {name}</b><br>HW12: {row.hw12:.0f}s | HW23: {row.hw23:.0f}s<br>Time: {str(row.datetime)[:19]}"
                    for row in group_df.itertuples()
                ],
                hoverinfo="text",
            )
        )

        # 2. Current head marker (latest point in trajectory)
        graph.add_trace(
            go.Scatter(
                name=name,
                x=[group_df.hw12.iloc[-1]],
                y=[group_df.hw23.iloc[-1]],
                mode="markers",
                marker={"size": 8.5, "color": color, "line": {"color": "#FFFFFF", "width": 1.5}},
                showlegend=False,
                text=[
                    f"<b>Current Head: {name}</b><br>Latest HW: [{group_df.hw12.iloc[-1]:.0f}s, {group_df.hw23.iloc[-1]:.0f}s]<br>Time: {str(group_df.datetime.iloc[-1])[:19]}"
                ],
                hoverinfo="text",
            )
        )

    return graph


def build_m_dist_graph(series_df, line):
    graph = go.Figure()

    graph.update_layout(
        title=None,
        uirevision="md",
        showlegend=False,
        hovermode="closest",
        margin={"r": 15, "l": 45, "t": 10, "b": 22},
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "nticks": 8,
            "tickfont": {"size": 9.5, "family": "JetBrains Mono, monospace"},
        },
        yaxis={
            "title": {"text": "M-Distance (σ)", "font": {"size": 10, "color": "#94A3B8"}},
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "tickfont": {"size": 9.5, "family": "JetBrains Mono, monospace"},
        },
    )

    series_df = series_df.loc[series_df.dim == 1]
    if series_df.shape[0] < 1:
        return graph

    min_time = series_df.datetime.min()
    max_time = series_df.datetime.max()

    # Read threshold
    try:
        with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
            hyperparams = json.load(f)
        conf = hyperparams.get(str(line), {}).get("conf", 0.98)
    except Exception:
        conf = 0.98
    m_th = math.sqrt(chi2.ppf(conf, df=1))

    # Add red anomaly threshold line
    graph.add_shape(
        name=f"Anomaly Threshold ({conf * 100:.0f}%)",
        type="line",
        x0=min_time,
        y0=m_th,
        x1=max_time,
        y1=m_th,
        line={"color": "#EF4444", "width": 1.5, "dash": "dashdot"},
    )

    for (b1, b2), group_df in series_df.groupby(["bus1", "bus2"], sort=False):
        group_df = group_df.sort_values("datetime")
        name = f"{b1}-{b2}"
        color = get_group_color(b1, b2)

        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.datetime,
                y=group_df.m_dist,
                mode="lines+markers",
                line={"width": 2.5, "color": color},
                marker={"size": 5, "color": color},
                showlegend=False,
                text=[
                    f"<b>Group: {name}</b><br>M-Dist: {row.m_dist:.2f}σ (Th: {m_th:.2f}σ)<br>Anomaly: {'🔴 YES' if row.anom else '🟢 NO'}<br>Time: {str(row.datetime)[:19]}"
                    for row in group_df.itertuples()
                ],
                hoverinfo="text",
            )
        )

    return graph


def build_anoms_table(anomalies_df):
    # All bus names (adaptive to available columns)
    bus_names_all = [c for c in anomalies_df.columns if c.startswith("bus")]
    if not bus_names_all:
        bus_names_all = ["bus1", "bus2"]

    if anomalies_df.shape[0] < 1:
        return "No anomalies were detected yet."

    # Build group names
    names = []
    for i in range(anomalies_df.shape[0]):
        group = [anomalies_df.iloc[i][bus_names_all[k]] for k in range(len(bus_names_all))]
        name = str(group[0])
        for bus in group[1:]:
            if bus != "0":
                name += "-" + str(bus)
            else:
                break

        names.append(name)

    anomalies_df["group"] = names

    anomalies_df = anomalies_df[["dim", "group", "anom_size", "m_dist", "datetime"]]

    groups_dfs, n_groups = [], 0
    for group in anomalies_df.group.unique():
        group_df = anomalies_df[anomalies_df.group == group]
        group_df["m_dist"] = round(group_df.m_dist.mean(), 4)
        groups_dfs.append(group_df)
        n_groups += 1
        if n_groups >= 21:
            break

    # Final data for the table
    anomalies_df = pd.concat(groups_dfs)
    anomalies_df = anomalies_df.sort_values("datetime", ascending=False).drop_duplicates(
        "group", keep="first"
    )

    table = dash_table.DataTable(
        id="table" + location,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current=0,
        page_size=5,
        style_header={"backgroundColor": "rgb(50, 50, 50)", "color": "white", "fontWeight": "bold"},
        style_cell={
            "padding": "2px",
            "width": "auto",
            "textAlign": "center",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_table={"overflowX": "auto"},
        columns=[{"name": i, "id": i} for i in anomalies_df.columns],
        data=anomalies_df.to_dict("records"),
    )

    return table


def _empty_figure(message):
    """Return a minimal themed figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"family": "DM Sans, sans-serif", "size": 15, "color": "#94A3B8"},
    )
    fig.update_layout(theme_layout("dark"))
    return fig


# CALLBACKS


# CALLBACK 0a - Live Map Positions
@app.callback(
    [Output("map" + location, "figure")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
    ],
)
def update_buses_position(n_intervals, n_clicks, pathname, theme="dark"):
    line = pathname.split("/")[-1] if pathname else "25"
    burst = read_df("burst", line=line)
    line_burst = (
        burst.loc[burst.line.astype(str) == str(line)] if not burst.empty else pd.DataFrame()
    )
    if not line_burst.empty:
        # One position per bus (closest upcoming stop)
        line_burst = line_burst.sort_values("estimateArrive").drop_duplicates(["bus"], keep="first")
    return [build_map(line_burst, theme, line=line)]


# CALLBACK 0b - Title
@app.callback(
    [Output("tab-title" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
    ],
)
def update_title_sliders(n_intervals, n_clicks, pathname):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")

    now = dt.now()
    now = now.replace(microsecond=0)

    return [f"Line {line} — updated {now.time()}"]


# CALLBACK 0c - Sliders update
@app.callback(
    [Output("hidden-div" + location, "children")],
    [
        Input("conf-slider" + location, "value"),
        Input("size-th-slider" + location, "value"),
        Input("url", "pathname"),
    ],
)
def update_hyperparams(conf, size_th, pathname):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")
    try:
        if (conf == 0) | (size_th == 0):
            return [html.H1("", className="box subtitle is-6")]

        conf = round(conf / 100, 3)

        # Read dict
        with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
            hyperparams = json.load(f)

        # Update hyperparams
        hyperparams[line]["conf"] = conf
        hyperparams[line]["size_th"] = size_th

        # Write dict
        with open(resolve_path(location + "/Data/Anomalies/hyperparams.json"), "w") as fp:
            json.dump(hyperparams, fp)

    except:  # noqa: E722
        pass

    return [""]


# CALLBACK 2 - Buses headways representation
@app.callback(
    [Output("flat-hws" + location, "figure")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
    ],
)
def update_flat_hws(n_intervals, n_clicks, pathname, theme="dark"):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")

    hws_burst = read_df("hws_burst", line=line)

    line_hws = hws_burst.loc[hws_burst.line == line]

    if line_hws.shape[0] < 1:
        return [_empty_figure("No headway data available for this line right now.")]

    # Create graph
    flat_hws_graph = build_graph(line_hws)

    # Apply theme styling and return the figure directly (keeps graph mounted, no flicker)
    flat_hws_graph.update_layout(**theme_layout(theme, uirevision=str(line)))
    return [flat_hws_graph]


# CALLBACK 3 - 1D Headways Time Series
@app.callback(
    [Output("time-series-hws" + location, "figure")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_time_series_hws(n_intervals, n_clicks, pathname, theme="dark", hoverData=None):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")
    cached = _cached_figure("ts1", line)
    if cached[1]:
        return [cached[0]]

    hover_buses = _parse_hover_buses(hoverData, line)

    series = read_df("series", line=line)

    line_series = series.loc[(series.line == line) & (series.dim == 1)]

    if hover_buses:
        if len(hover_buses) == 1:
            line_series = line_series.loc[
                (line_series.bus1 == hover_buses[0]) | (line_series.bus2 == hover_buses[0])
            ]
        elif len(hover_buses) == 2:
            line_series = line_series.loc[(line_series.bus1 == hover_buses[0])]

    if line_series.shape[0] < 1:
        return [
            _empty_figure(
                "No headways to analyse. There are less than 2 buses inside each line direction."
            )
        ]

    model = _get_hour_range_and_model(line, 1)
    if model is None:
        return [_empty_figure("Hour range for current time not defined. Waiting till 7am.")]

    # Read dict (bounded retries - never spin forever)
    conf = 0.98
    for _ in range(5):
        try:
            with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
                hyperparams = json.load(f)
            conf = hyperparams.get(line, {}).get("conf", 0.98)
            break
        except Exception:
            time.sleep(0.2)

    time_series_graph = build_time_series_graph(line_series, model, conf)

    _store_figure("ts1", line, time_series_graph)
    time_series_graph.update_layout(**theme_layout(theme, uirevision=str(line)))
    return [time_series_graph]


# CALLBACK 4 - 2D Headways Time Series
@app.callback(
    [Output("2d-time-series-hws" + location, "figure")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_2d_time_series_hws(n_intervals, n_clicks, pathname, theme="dark", hoverData=None):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")
    cached = _cached_figure("ts2", line)
    if cached[1]:
        return [cached[0]]

    hover_buses = _parse_hover_buses(hoverData, line)

    series = read_df("series", line=line)

    line_series = series.loc[(series.line == line) & (series.dim == 2)]

    if hover_buses:
        if len(hover_buses) == 1:
            line_series = line_series.loc[(line_series.bus2 == hover_buses[0])]
        elif len(hover_buses) == 2:
            return [html.H1("Click a bus, links not supported.", className="title is-5")]

    if line_series.shape[0] < 1:
        return [_empty_figure("No 2d headways to analyse. Click a bus between two buses.")]

    model = _get_hour_range_and_model(line, 2)
    if model is None:
        return [_empty_figure("2D Model for this hour range not available.")]

    # Read dict (bounded retries - never spin forever)
    conf = 0.98
    for _ in range(5):
        try:
            with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
                hyperparams = json.load(f)
            conf = hyperparams.get(line, {}).get("conf", 0.98)
            break
        except Exception:
            time.sleep(0.2)

    time_series_graph = build_2d_time_series_graph(line_series, model, conf)

    _store_figure("ts2", line, time_series_graph)
    time_series_graph.update_layout(**theme_layout(theme, uirevision=str(line)))
    return [time_series_graph]


# CALLBACK 5 - Mahalanobis Distance series
@app.callback(
    [Output("mdist-hws" + location, "figure")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_mdist_series(n_intervals, n_clicks, pathname, theme="dark", hoverData=None):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")
    cached = _cached_figure("md", line)
    if cached[1]:
        return [cached[0]]

    hover_buses = _parse_hover_buses(hoverData, line)

    series = read_df("series", line=line)

    line_series = series.loc[series.line == line]

    if hover_buses:
        if len(hover_buses) == 1:
            line_series = line_series.loc[
                (line_series.bus1 == hover_buses[0])
                | (line_series.bus2 == hover_buses[0])
                | (line_series.bus3 == hover_buses[0])
                | (line_series.bus4 == hover_buses[0])
                | (line_series.bus5 == hover_buses[0])
                | (line_series.bus6 == hover_buses[0])
                | (line_series.bus7 == hover_buses[0])
                | (line_series.bus8 == hover_buses[0])
                | (line_series.bus9 == hover_buses[0])
            ]

        elif len(hover_buses) == 2:
            line_series = line_series.loc[
                ((line_series.bus1 == hover_buses[0]) & (line_series.bus2 == hover_buses[1]))
                | ((line_series.bus2 == hover_buses[0]) & (line_series.bus3 == hover_buses[1]))
                | ((line_series.bus3 == hover_buses[0]) & (line_series.bus4 == hover_buses[1]))
                | ((line_series.bus4 == hover_buses[0]) & (line_series.bus5 == hover_buses[1]))
                | ((line_series.bus5 == hover_buses[0]) & (line_series.bus6 == hover_buses[1]))
                | ((line_series.bus6 == hover_buses[0]) & (line_series.bus7 == hover_buses[1]))
                | ((line_series.bus7 == hover_buses[0]) & (line_series.bus8 == hover_buses[1]))
                | ((line_series.bus8 == hover_buses[0]) & (line_series.bus9 == hover_buses[1]))
            ]

    if line_series.shape[0] < 1:
        return [
            _empty_figure(
                "No headways to analyse. There are less than 2 buses inside each line direction."
            )
        ]

    # Create mh dist graph
    m_dist_graph = build_m_dist_graph(line_series, line)

    _store_figure("md", line, m_dist_graph)
    m_dist_graph.update_layout(**theme_layout(theme, uirevision=str(line)))
    return [m_dist_graph]


# CALLBACK 6 - Anomalies series
@app.callback(
    [Output("anom-hws-div" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
    ],
)
def update_anomalies_table(n_intervals, n_clicks, pathname):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")

    anomalies = read_df("anomalies", line=line)

    if anomalies.shape[0] < 1:
        return [
            html.Div(
                className="box",
                style={"height": box_height},
                children=[html.H2("No anomalies detected yet.", className="title is-5")],
            )
        ]

    line_anoms = anomalies.loc[anomalies.line == line]

    # Create anomalies table
    anoms_table = build_anoms_table(line_anoms)

    # And return all of them
    return [
        html.Div(
            className="box",
            style={"height": box_height},
            children=[html.H2("DETECTED ANOMALIES", className="title is-5"), anoms_table],
        )
    ]


# CALLBACK 7 - KPI cards
# CALLBACK 7 - KPI Strip Calculations
@app.callback(
    [
        Output("kpi-fleet" + location, "children"),
        Output("kpi-headway" + location, "children"),
        Output("kpi-qos" + location, "children"),
        Output("kpi-filtered" + location, "children"),
        Output("kpi-anoms" + location, "children"),
    ],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
    ],
)
def update_kpis(n_intervals, n_clicks, pathname):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")
    try:
        hws = read_df("hws_burst", line=line)
        burst = read_df("burst", line=line)

        hws_line = (
            hws.loc[hws.line == line] if (hws is not None and not hws.empty) else pd.DataFrame()
        )
        line_hws = hws_line.loc[hws_line.hw_pos > 0] if not hws_line.empty else pd.DataFrame()

        fleet = int(hws_line["busB"].nunique()) if not hws_line.empty else 0
        total_buses = (
            int(burst.loc[burst.line.astype(str) == str(line)]["bus"].nunique())
            if (burst is not None and not burst.empty)
            else fleet
        )
        filtered = max(0, total_buses - fleet)

        mean_hw = int(line_hws.headway.mean()) if not line_hws.empty else 0

        # QoS regularity
        model = _get_hour_range_and_model(line, 1)
        if model and not line_hws.empty:
            mean = float(model.get("mean", 360))
            std = float(model.get("cov_matrix", 120))
            within_bounds = line_hws.headway.apply(lambda hw: abs(hw - mean) <= 2 * std)
            qos = int(round(100 * within_bounds.sum() / len(within_bounds)))
        else:
            qos = 100

        anoms = read_df("anomalies", line=line)
        anoms_line = (
            anoms.loc[anoms.line == line]
            if (anoms is not None and not anoms.empty)
            else pd.DataFrame()
        )
        n_anoms = len(anoms_line.drop_duplicates(["bus1", "bus2"])) if not anoms_line.empty else 0

        return [str(fleet), f"{mean_hw}s", f"{qos}%", str(filtered), str(n_anoms)]
    except Exception:
        return ["—", "—", "—", "—", "—"]


# CALLBACK 8a - Series tab switching (1D / 2D)
@app.callback(
    [
        Output("time-series-hws" + location, "style"),
        Output("2d-time-series-hws" + location, "style"),
    ],
    [Input("tabs-series" + location, "value")],
)
def switch_series_tab(tab):
    visible = {"display": "block", "height": "100%", "width": "100%"}
    hidden = {"display": "none"}
    return [
        visible if tab == "ts1" + location else hidden,
        visible if tab == "ts2" + location else hidden,
    ]


# CALLBACK 8b - Anomalies tab switching (M-Dist / Table)
@app.callback(
    [
        Output("mdist-hws" + location, "style"),
        Output("anom-hws-div" + location, "style"),
    ],
    [Input("tabs-anoms" + location, "value")],
)
def switch_anoms_tab(tab):
    visible = {"display": "block", "height": "100%", "width": "100%"}
    hidden = {"display": "none"}
    return [
        visible if tab == "md" + location else hidden,
        {"display": "block", "height": "100%", "overflowY": "auto"}
        if tab == "an" + location
        else hidden,
    ]


# CALLBACK 9 - Dynamic Route Pills Active State
@app.callback(
    Output("route-pills-container" + location, "children"),
    [Input("url", "pathname")],
)
def update_active_route_pills(pathname):
    current_line = pathname.split("/")[-1] if pathname else "25"
    all_lines = [
        ("18", "Line 18"),
        ("24", "Line 24"),
        ("25", "Line 25"),
        ("73", "Line 73"),
    ]
    pills = []
    for line, label in all_lines:
        is_active = line == current_line
        cls = "route-btn active" if is_active else "route-btn"
        pills.append(
            dcc.Link(
                label,
                href=f"/realtime/london/{line}",
                className=cls,
                style={"padding": "0.45rem 0.9rem", "minWidth": "0", "fontSize": "0.85rem"},
            )
        )
    return pills
