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


# Available colors
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

colors2 = [
    "#023fa5",
    "#7d87b9",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#f79cd4",
]

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
    className="",
    children=[
        # ---- Executive Control & Parameters Toolbar ----
        html.Div(
            className="modern-card no-hover",
            style={"padding": "1.25rem 1.5rem", "marginBottom": "1.25rem"},
            children=[
                # Top Row: Title, Route Pills, Action Buttons
                html.Div(
                    className="flex-between flex-wrap",
                    style={"gap": "1rem", "marginBottom": "1rem"},
                    children=[
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                html.H1(
                                    "London Transit Monitor",
                                    style={"fontSize": "1.55rem", "margin": 0, "fontWeight": "700"},
                                ),
                                html.Span(
                                    [html.Span(className="pulse-indicator"), " LIVE"],
                                    className="badge-pill success",
                                    style={"fontSize": "0.7rem"},
                                ),
                                html.Span(
                                    id="tab-title" + location,
                                    style={
                                        "fontSize": "0.92rem",
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
                                ),
                            ],
                        ),
                    ],
                ),
                # Bottom Row: Responsive Sliders & Model Baseline Bar
                html.Div(
                    style={
                        "borderTop": "1px solid var(--border-color)",
                        "paddingTop": "0.85rem",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "flexWrap": "wrap",
                        "gap": "1.25rem",
                    },
                    children=[
                        # Slider 1: Confidence
                        html.Div(
                            style={"flex": "1 1 240px", "minWidth": "220px", "maxWidth": "380px"},
                            children=[
                                html.Div(
                                    className="flex-between",
                                    style={"marginBottom": "0.2rem"},
                                    children=[
                                        html.Label(
                                            [
                                                html.I(
                                                    className="fa-solid fa-sliders",
                                                    style={
                                                        "marginRight": "0.4rem",
                                                        "color": "var(--primary-color)",
                                                    },
                                                ),
                                                "Confidence (1 - α)",
                                            ],
                                            style={"fontSize": "0.82rem", "fontWeight": "600"},
                                        ),
                                        html.Span(
                                            "Anomaly Threshold",
                                            style={
                                                "fontSize": "0.75rem",
                                                "color": "var(--text-muted)",
                                            },
                                        ),
                                    ],
                                ),
                                dcc.Slider(
                                    id="conf-slider" + location,
                                    min=90,
                                    max=100,
                                    step=0.05,
                                    value=99,
                                    marks={i: f"{i}%" for i in range(90, 101, 2)},
                                ),
                            ],
                        ),
                        # Slider 2: Size Threshold
                        html.Div(
                            style={"flex": "1 1 240px", "minWidth": "220px", "maxWidth": "380px"},
                            children=[
                                html.Div(
                                    className="flex-between",
                                    style={"marginBottom": "0.2rem"},
                                    children=[
                                        html.Label(
                                            [
                                                html.I(
                                                    className="fa-solid fa-filter",
                                                    style={
                                                        "marginRight": "0.4rem",
                                                        "color": "var(--accent-color)",
                                                    },
                                                ),
                                                "Filter Window (k)",
                                            ],
                                            style={"fontSize": "0.82rem", "fontWeight": "600"},
                                        ),
                                        html.Span(
                                            "Consecutive Ticks",
                                            style={
                                                "fontSize": "0.75rem",
                                                "color": "var(--text-muted)",
                                            },
                                        ),
                                    ],
                                ),
                                dcc.Slider(
                                    id="size-th-slider" + location,
                                    min=1,
                                    max=15,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(1, 16, 2)},
                                ),
                            ],
                        ),
                        # Model Badge & Info
                        html.Div(
                            className="flex-gap",
                            style={"flex": "0 1 auto"},
                            children=[
                                html.Span(
                                    [
                                        html.I(
                                            className="fa-solid fa-chart-pie",
                                            style={"marginRight": "0.35rem"},
                                        ),
                                        "Gaussian d≤3",
                                    ],
                                    className="badge-pill primary",
                                    style={"fontSize": "0.72rem"},
                                    title="Multivariate Gaussian distribution with Mahalanobis distance metric",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # ---- Hidden State & Polling ----
        html.Div(id="hidden-div" + location, style={"display": "none"}),
        dcc.Interval(id="interval-component" + location, interval=15000, n_intervals=0),
        # ---- KPI Cards ----
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
                                html.Span("Active Fleet", className="kpi-label"),
                                html.I(
                                    className="fa-solid fa-bus",
                                    style={"color": "var(--primary-color)"},
                                ),
                            ],
                        ),
                        html.Div(id="kpi-fleetLondon", className="kpi-val", children="—"),
                        html.Span("Vehicles reporting", className="kpi-sub"),
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
                        html.Div(id="kpi-headwayLondon", className="kpi-val", children="—"),
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
                                    className="fa-solid fa-gauge-high", style={"color": "#10B981"}
                                ),
                            ],
                        ),
                        html.Div(id="kpi-qosLondon", className="kpi-val", children="—"),
                        html.Span("Service regularity index", className="kpi-sub"),
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
                        html.Div(id="kpi-anomsLondon", className="kpi-val", children="—"),
                        html.Span("Detected in current window", className="kpi-sub"),
                    ],
                ),
            ],
        ),
        # ---- Main Workspace: Map + Headway Corridor ----
        html.Div(
            className="workspace-grid",
            children=[
                html.Div(
                    className="modern-card no-hover",
                    style={"padding": "1rem"},
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.6rem"},
                            children=[
                                html.H3(
                                    "Fleet Spatial Map", style={"fontSize": "1.1rem", "margin": 0}
                                ),
                                html.Span(
                                    "Live positions",
                                    className="badge-pill primary",
                                    style={"fontSize": "0.65rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="map" + location,
                            style={"height": "52vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False, "scrollZoom": True},
                        ),
                    ],
                ),
                html.Div(
                    className="modern-card no-hover",
                    style={"padding": "1rem"},
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "0.6rem"},
                            children=[
                                html.H3(
                                    "Headway Corridor", style={"fontSize": "1.1rem", "margin": 0}
                                ),
                                html.Span(
                                    "Bunching risk view",
                                    className="badge-pill warning",
                                    style={"fontSize": "0.65rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="flat-hws" + location,
                            style={"height": "52vh"},
                            figure=go.Figure(),
                            clear_on_unhover=True,
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
        # ---- Analytics Tabs ----
        html.Div(
            className="modern-card no-hover",
            style={"padding": "1rem", "marginTop": "1.5rem"},
            children=[
                dcc.Tabs(
                    id="analytics-tabs" + location,
                    value="ts1" + location,
                    className="custom-tabs-container",
                    children=[
                        dcc.Tab(
                            label="Headway Time Series (1D)",
                            value="ts1" + location,
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            label="Headway Dynamics (2D)",
                            value="ts2" + location,
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            label="Mahalanobis Distance",
                            value="md" + location,
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            label="Anomaly Events",
                            value="an" + location,
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                    ],
                ),
                html.Div(
                    style={"marginTop": "0.5rem"},
                    children=[
                        dcc.Graph(
                            id="time-series-hws" + location,
                            style={"height": "52vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                        dcc.Graph(
                            id="2d-time-series-hws" + location,
                            style={"height": "52vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                        dcc.Graph(
                            id="mdist-hws" + location,
                            style={"height": "52vh"},
                            figure=go.Figure(),
                            config={"displayModeBar": False},
                        ),
                        html.Div(
                            id="anom-hws-div" + location,
                            style={"height": "52vh", "overflowY": "auto"},
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
        df = db.get_series_df("London", str(line) if line else "25", dim=1, limit=300)
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
            color = colors2[str_to_int(bus.bus) % len(colors2)]
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
                )
            )

    return new_map


def build_graph(line_hws):
    """
    Returns a figure with the graph of headways between buses
    """

    # Process headways
    headways = line_hws

    # Create figure object
    graph = go.Figure()

    # Set title and layout
    graph.update_layout(
        xaxis={"nticks": 30},
        yaxis={"type": "category", "showgrid": False, "zeroline": False},
        showlegend=False,
        margin={"r": 0, "l": 0, "t": 0, "b": 0},
        hovermode="closest",
    )

    if headways.shape[0] < 1:
        return graph

    # Destinations
    line = headways.line.iloc[0]
    dest2, dest1 = lines_dict[line]["destinations"]

    # Max dists
    hw1 = headways.loc[headways.direction == 1]
    hw2 = headways.loc[headways.direction == 2]
    if hw1.shape[0] == 0:
        pass
    else:
        hw1.busB_ttls.max()
        # Add trace
        for i in range(hw1.shape[0] - 1):
            N, X = 50, [hw1.iloc[i].busB_ttls, hw1.iloc[i].busB_ttls + hw1.iloc[i + 1].headway]
            X_new = []
            for k in range(N + 1):
                X_new.append(X[0] + (X[1] - X[0]) * k / N)

            graph.add_trace(
                go.Scatter(
                    x=X_new,
                    y=[("<b>" + dest1 + " ") for i in range(len(X_new))],
                    mode="lines",
                    line={
                        "width": 3,
                        "color": colors2[
                            (str_to_int(hw1.iloc[i + 1].busA) + str_to_int(hw1.iloc[i + 1].busB))
                            % len(colors2)
                        ],
                    },
                    showlegend=False,
                    hoverinfo="text",
                    text="<b>Bus group: "
                    + str(hw1.iloc[i + 1].busA)
                    + "-"
                    + str(hw1.iloc[i + 1].busB)
                    + "</b> <br>"
                    + "Headway: "
                    + str(hw1.iloc[i + 1].headway)
                    + "s",
                )
            )

    if hw2.shape[0] == 0:
        pass
    else:
        hw2.busB_ttls.max()
        # Add trace
        for i in range(hw2.shape[0] - 1):
            N, X = 50, [hw2.iloc[i].busB_ttls, hw2.iloc[i].busB_ttls + hw2.iloc[i + 1].headway]
            X_new = []
            for k in range(N + 1):
                X_new.append(X[0] + (X[1] - X[0]) * k / N)

            graph.add_trace(
                go.Scatter(
                    x=X_new,
                    y=[("<b>" + dest2 + " ") for i in range(len(X_new))],
                    mode="lines",
                    line={
                        "width": 3,
                        "color": colors2[
                            (str_to_int(hw2.iloc[i + 1].busA) + str_to_int(hw2.iloc[i + 1].busB))
                            % len(colors2)
                        ],
                    },
                    showlegend=False,
                    hoverinfo="text",
                    text="<b>Bus group: "
                    + str(hw2.iloc[i + 1].busA)
                    + "-"
                    + str(hw2.iloc[i + 1].busB)
                    + "</b> <br>"
                    + "Headway: "
                    + str(hw2.iloc[i + 1].headway)
                    + "s",
                )
            )

    # Add buses to graph
    for bus in headways.itertuples():
        # Assign color based on bus id
        color = colors[str_to_int(bus.busB) % len(colors)]

        if bus.direction == 1:
            dest = dest1
        else:
            dest = dest2

        # Add marker
        graph.add_trace(
            go.Scatter(
                mode="markers",
                name=bus.busB,
                x=[bus.busB_ttls],
                y=["<b>" + dest + " "],
                marker={"size": 30, "color": color, "line": {"color": "black", "width": 1.5}},
                text=[
                    "<b>Bus: "
                    + str(bus.busB)
                    + "</b> <br>"
                    + str(bus.headway)
                    + "s to next bus <br>"
                    + str(bus.busB_ttls)
                    + "s to last stop"
                ],
                hoverinfo="text",
            )
        )
        graph.add_trace(
            go.Scatter(
                mode="text",
                text="<b>" + str(bus.busB),
                x=[bus.busB_ttls],
                y=["<b>" + dest + " "],
                hoverinfo="none",
            )
        )

    graph.update_layout(xaxis_range=(0, max_ttls[line]), uirevision=str(line))

    # Finally we return the graph
    return graph


def build_time_series_graph(series_df, model, conf):
    graph = go.Figure()

    # Set title and layout
    graph.update_layout(
        title="<b>1D HEADWAYS TIME SERIES</b> - (In seconds)",
        uirevision="ts1",
        legend_title="<b>Group ids</b>",
        yaxis={
            "nticks": 20,
            "range": (series_df.hw12.min() - 50, series_df.hw12.max() + 50),
            "zerolinecolor": "darkgrey",
        },
        legend={"x": -0.02, "y": -0.05, "orientation": "h"},
        margin={"r": 0, "l": 0, "t": 40, "b": 0},
        hovermode="closest",
    )

    series_df = series_df.loc[series_df.dim == 1]
    if series_df.shape[0] < 1:
        return graph

    # All bus names
    bus_names_all = ["bus" + str(i) for i in range(1, 3)]
    ["hw" + str(i) + str(i + 1) for i in range(1, 2)]

    # Min and max datetimes
    min_time = series_df.datetime.min()
    max_time = series_df.datetime.max()

    # Dim threshold
    dim = 1
    std = model["cov_matrix"]
    mean = model["mean"]
    m_th = math.sqrt(chi2.ppf(conf, df=dim))
    # Add thresholds
    thresholds = [(mean - std * m_th), (mean + std * m_th)]
    for th in thresholds:
        graph.add_shape(
            name=str(th),
            type="line",
            x0=min_time,
            y0=th,
            x1=max_time,
            y1=th,
            line={
                "color": "red",
                "width": 2,
                "dash": "dashdot",
            },
        )

    # Locate unique groups
    unique_groups = []
    unique_groups_df = series_df.drop_duplicates(bus_names_all)
    for i in range(unique_groups_df.shape[0]):
        group = [unique_groups_df.iloc[i][bus_names_all[k]] for k in range(2)]
        unique_groups.append(group)

    for group in unique_groups:
        # Build indexing conditions
        conds = [series_df[bus_names_all[k]] == group[k] for k in range(2)]
        final_cond = True
        for cond in conds:
            final_cond &= cond
        group_df = series_df.loc[final_cond]
        group_df = group_df.sort_values("datetime")

        name = str(group[0])
        for bus in group[1:]:
            if bus != 0:
                name += "-" + str(bus)
            else:
                break

        # Build group trace
        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.datetime,
                y=group_df.hw12,
                mode="lines+markers",
                line={
                    "width": 3,
                    "color": colors2[
                        (str_to_int(group_df.bus1.iloc[0]) + str_to_int(group_df.bus2.iloc[0]))
                        % len(colors2)
                    ],
                },
                text=[
                    "<b>Bus group: "
                    + str(name)
                    + "</b> <br>"
                    + "Headway: "
                    + str(row.hw12)
                    + "s<br>"
                    + row.datetime
                    for row in group_df.itertuples()
                ],
                hoverinfo="text",
            )
        )

    return graph


def build_2d_time_series_graph(series_df, model, conf):
    graph = go.Figure()

    # Set title and layout
    graph.update_layout(
        title="<b>2D HEADWAYS TIME SERIES</b> - (In seconds)",
        uirevision="ts2",
        legend_title="<b>Group ids</b>",
        xaxis={"nticks": 20, "zerolinecolor": "darkgrey"},
        yaxis={"nticks": 20, "zerolinecolor": "darkgrey"},
        legend={"x": -0.02, "y": -0.05, "orientation": "h"},
        margin={"r": 0, "l": 0, "t": 40, "b": 0},
        hovermode="closest",
    )

    series_df = series_df.loc[series_df.dim == 2]
    if series_df.shape[0] < 1:
        return graph

    # All bus names
    bus_names_all = ["bus" + str(i) for i in range(1, 4)]
    ["hw" + str(i) + str(i + 1) for i in range(1, 3)]

    # Min and max datetimes
    series_df.datetime.min()
    series_df.datetime.max()

    # Dim threshold
    dim = 2
    cov_matrix = model["cov_matrix"]
    mean = model["mean"]
    math.sqrt(chi2.ppf(conf, df=dim))

    # Confidence ellipse points
    x, y = ellipse(mean, cov_matrix, conf)
    # Confidence ellipse
    graph.add_trace(
        go.Scatter(
            name=f"{conf * 100}% Confidence Ellipse",
            x=x,
            y=y,
            mode="lines",
            line={"color": "red", "dash": "dash"},
            text=f"{conf * 100}% Confidence Ellipse",
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Locate unique groups
    unique_groups = []
    unique_groups_df = series_df.drop_duplicates(bus_names_all)
    for i in range(unique_groups_df.shape[0]):
        group = [unique_groups_df.iloc[i][bus_names_all[k]] for k in range(3)]
        unique_groups.append(group)

    for group in unique_groups:
        # Build indexing conditions
        conds = [series_df[bus_names_all[k]] == group[k] for k in range(3)]
        final_cond = True
        for cond in conds:
            final_cond &= cond
        group_df = series_df.loc[final_cond]
        group_df = group_df.sort_values("datetime")

        name = str(group[0])
        for bus in group[1:]:
            if bus != 0:
                name += "-" + str(bus)
            else:
                break

        # Head point
        graph.add_trace(
            go.Scatter(
                name=name,
                x=[group_df.hw12.iloc[-1]],
                y=[group_df.hw23.iloc[-1]],
                mode="markers",
                marker={"size": 10, "color": "black"},
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Build group trace
        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.hw12,
                y=group_df.hw23,
                mode="lines+markers",
                line={"width": 3, "color": colors[str_to_int(group_df.bus2.iloc[0]) % len(colors)]},
                text=[
                    "<b>Bus group: "
                    + str(name)
                    + "</b> <br>"
                    + "Headways: ["
                    + str(row.hw12)
                    + ","
                    + str(row.hw23)
                    + "]<br>"
                    + row.datetime
                    for row in group_df.itertuples()
                ],
                hoverinfo="text",
            )
        )
    return graph


def build_m_dist_graph(series_df, line):
    graph = go.Figure()

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

    # Set title and layout
    graph.update_layout(
        title="<b>MAHALANOBIS DISTANCE</b>",
        uirevision="md",
        legend_title="<b>Group ids</b>",
        xaxis={"nticks": 20},
        yaxis={"title_text": "Mahalanobis Distance", "nticks": 20},
        legend={"x": -0.02, "y": -0.05, "orientation": "h"},
        margin={"r": 0, "l": 0, "t": 40, "b": 0},
        hovermode="closest",
    )

    if series_df.shape[0] < 1:
        return graph

    # All bus names (adaptive to whatever dimensionality the data carries)
    avail_bus = [c for c in series_df.columns if c.startswith("bus")]
    avail_hw = [c for c in series_df.columns if c.startswith("hw")]
    bus_names_all = avail_bus
    hw_names_all = avail_hw

    # Min and max datetimes
    min_time = series_df.datetime.min()
    max_time = series_df.datetime.max()

    # Locate unique groups
    unique_groups = []
    unique_groups_df = series_df.drop_duplicates(bus_names_all)
    for i in range(unique_groups_df.shape[0]):
        group = [unique_groups_df.iloc[i][bus_names_all[k]] for k in range(len(bus_names_all))]
        unique_groups.append(group)

    last_dim = 0
    for group in unique_groups:
        # Build indexing conditions
        conds = [series_df[bus_names_all[k]] == group[k] for k in range(len(bus_names_all))]
        final_cond = True
        for cond in conds:
            final_cond &= cond
        group_df = series_df.loc[final_cond]
        group_df = group_df.sort_values("datetime")

        # Dimension
        dim = group_df.iloc[0].dim
        color = colors[dim % len(colors)]

        # Dim threshold
        m_th = math.sqrt(chi2.ppf(conf, df=dim))

        if dim != last_dim:
            graph.add_shape(
                name=f"{dim}Dim MD Threshold",
                type="line",
                x0=min_time,
                y0=m_th,
                x1=max_time,
                y1=m_th,
                line={
                    "color": color,
                    "width": 2,
                    "dash": "dashdot",
                },
            )

        last_dim = dim

        name = str(group[0])
        for bus in group[1 : dim + 1]:
            name += "-" + str(bus)

        hw_values = []
        for _index, row in group_df.iterrows():
            hw_value = str(row.hw12)
            for hw_name in hw_names_all[1:dim]:
                if hw_name in group_df.columns:
                    hw_value += "," + str(row[hw_name])
            hw_values.append(hw_value)

        # Build group trace
        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.datetime,
                y=group_df.m_dist,
                mode="lines+markers",
                line={"width": 3, "color": color},
                text=[
                    "<b>Bus group: "
                    + str(name)
                    + "</b> <br>"
                    + "Headways: ["
                    + hw_values[i]
                    + "]<br>"
                    + group_df.iloc[i].datetime
                    for i in range(group_df.shape[0])
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

    return [
        html.H1(
            f"Confidence set to {conf} and size threshold set to {size_th} in the next update",
            className="box subtitle is-6",
        )
    ]


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
def update_time_series_hws(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
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
def update_2d_time_series_hws(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
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

    _store_figure("ts1", line, time_series_graph)
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
def update_mdist_series(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
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
@app.callback(
    [
        Output("kpi-fleetLondon", "children"),
        Output("kpi-headwayLondon", "children"),
        Output("kpi-qosLondon", "children"),
        Output("kpi-anomsLondon", "children"),
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
        hws_line = hws.loc[hws.line == line]
        line_hws = hws_line.loc[hws_line.hw_pos > 0]

        fleet = int(hws_line["busB"].nunique())
        mean_hw = int(line_hws.headway.mean()) if line_hws.shape[0] > 0 else 0

        # QoS regularity: share of observations within 2 sigma of the modeled mean
        now = dt.now()
        day_type = "LA" if now.weekday() <= 4 else ("SA" if now.weekday() == 5 else "FE")
        hour_ranges = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]
        hour_range = None
        for h_range in hour_ranges:
            if h_range[0] <= now.hour < h_range[1]:
                hour_range = str(h_range[0]) + "-" + str(h_range[1])
                break

        if hour_range and line in models_params_dict and day_type in models_params_dict[line]:
            model = models_params_dict[line][day_type].get(hour_range, {})
            m1 = model.get("1", {})
            if m1 and line_hws.shape[0] > 0:
                mu = float(m1.get("mean", 0))
                std = float(m1.get("cov_matrix", 1))
                if std > 0:
                    within = ((line_hws.headway - mu).abs() <= 2 * std).mean()
                    qos = int(round(within * 100))
                else:
                    qos = 100
            else:
                qos = 0
        else:
            qos = 0

        anoms = read_df("anomalies", line=line)
        anoms_line = anoms.loc[anoms.line == line] if anoms.shape[0] > 0 else anoms
        n_anoms = int(anoms_line.shape[0])

        return [str(fleet), f"{mean_hw}s", f"{qos}%", str(n_anoms)]
    except Exception:
        return ["—", "—", "—", "—"]


# CALLBACK 8 - Analytics tab switching (show/hide panels)
@app.callback(
    [
        Output("time-series-hws" + location, "style"),
        Output("2d-time-series-hws" + location, "style"),
        Output("mdist-hws" + location, "style"),
        Output("anom-hws-div" + location, "style"),
    ],
    [Input("analytics-tabs" + location, "value")],
)
def switch_analytics_tab(tab):
    hidden = {"display": "none"}
    visible = {"height": "52vh"}
    return [
        visible if tab == "ts1" + location else hidden,
        visible if tab == "ts2" + location else hidden,
        visible if tab == "md" + location else hidden,
        {"height": "52vh", "overflowY": "auto"} if tab == "an" + location else hidden,
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
