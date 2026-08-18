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
                                    "Madrid Transit Monitor",
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
                            style={
                                "flex": "1 1 200px",
                                "minWidth": "160px",
                                "maxWidth": "300px",
                                "padding": "0 10px",
                            },
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
                                    value=98,
                                    marks={i: f"{i}%" for i in range(90, 101, 2)},
                                ),
                            ],
                        ),
                        # Slider 2: Size Threshold
                        html.Div(
                            style={
                                "flex": "1 1 200px",
                                "minWidth": "160px",
                                "maxWidth": "300px",
                                "padding": "0 10px",
                            },
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
                        html.Div(id="kpi-fleetMadrid", className="kpi-val", children="—"),
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
                        html.Div(id="kpi-headwayMadrid", className="kpi-val", children="—"),
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
                        html.Div(id="kpi-qosMadrid", className="kpi-val", children="—"),
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
                        html.Div(id="kpi-anomsMadrid", className="kpi-val", children="—"),
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
                        dcc.Graph(
                            id="mdist-hws" + location,
                            style={"display": "none"},
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
)
# MAPBOX CONFIGURATION (uses carto tiles if no token configured)
mapbox_access_token = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
mapbox_style = "streets" if mapbox_access_token else "carto-darkmatter"
mapbox_light_style = "light" if mapbox_access_token else "carto-positron"


_CSV_PATHS = {
    "burst": "/Data/RealTime/buses_data_burst_cleaned.csv",
    "hws_burst": "/Data/RealTime/headways_burst.csv",
    "series": "/Data/RealTime/series.csv",
    "anomalies": "/Data/Anomalies/anomalies.csv",
}


_DB_TABLES = {
    "burst": "buses_burst",
    "hws_burst": "headways_burst",
    "series": "headways_series",
    "anomalies": "anomaly_events",
}


def _read_db(name, line):
    """Fast database read for live telemetry (returns None when unavailable)."""
    import sys

    root_path = resolve_path(".")
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from core import db

    if name == "burst":
        df = db.get_latest_burst_df("Madrid", str(line) if line else None)
    elif name == "hws_burst":
        df = db.get_latest_headways_df("Madrid", str(line) if line else None)
    elif name == "series":
        df = db.get_series_df("Madrid", str(line) if line else "1", dim=None, limit=60)
    elif name == "anomalies":
        df = db.get_anomalies_df("Madrid", str(line) if line else "1", limit=100)
    else:
        return None
    return df if not df.empty else None


def _read_csv_fallback(name, line):
    """Legacy CSV fallback when the database is empty."""
    rel = _CSV_PATHS.get(name)
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

        ts = db.get_latest_timestamp("Madrid", "headways_series")
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

        ts = db.get_latest_timestamp("Madrid", "headways_series")
        _LAST_SERIES_TS[(graph_key, line)] = (ts, figure)
    except Exception:
        pass


def read_df(name, line=None):
    """Read latest telemetry with timestamp-based memoization."""
    cache_key = f"{name}:{line}"
    ts = None
    try:
        from core import db

        table = _DB_TABLES.get(name)
        if table is not None:
            ts = db.get_latest_timestamp("Madrid", table)
            cached = _DF_CACHE.get(cache_key)
            if cached and cached[0] == ts:
                return cached[1]
    except Exception:
        pass

    try:
        df = _read_db(name, line)
        if df is not None:
            if ts is not None:
                _DF_CACHE[cache_key] = (ts, df)
            return df
    except Exception:
        pass
    return _read_csv_fallback(name, line)


def calculate_coords(df, stop_id, dist_to_stop):
    line_sn = df.iloc[0].line_sn
    direction = str(df.iloc[0].direction)
    bus_distance = int(lines_dict[line_sn][direction]["distances"][str(stop_id)]) - dist_to_stop
    nearest_row = find_nearest_row_by_dist(df, bus_distance)
    return nearest_row.lon, nearest_row.lat


def find_nearest_row_by_dist(df, dist_traveled):
    min_dist_error = 1000000.0
    df_reduced = df.loc[
        (df.dist_traveled > dist_traveled - 500) & (df.dist_traveled < dist_traveled + 500)
    ]
    if df_reduced.shape[0] != 0:
        for row in df_reduced.itertuples():
            error = abs(row.dist_traveled - dist_traveled)
            if error < min_dist_error:
                min_dist_error = error
                nearest_row = row
    else:
        nearest_row = df.iloc[0]
    return nearest_row


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


def calc_map_params(df):
    line = str(df.line.iloc[0])
    line1 = line_shapes.loc[(line_shapes.line_sn == line) & (line_shapes.direction == 1)]
    line2 = line_shapes.loc[(line_shapes.line_sn == line) & (line_shapes.direction == 2)]
    dest1 = lines_dict[line]["destinations"][1]

    lons, lats = [], []
    for bus in df.itertuples():
        if bus.destination == dest1:
            lon, lat = calculate_coords(line1, bus.stop, bus.DistanceBus)
        else:
            lon, lat = calculate_coords(line2, bus.stop, bus.DistanceBus)
        lons.append(lon)
        lats.append(lat)

    df["lon"] = lons
    df["lat"] = lats

    # Stable route-level geometric center and zoom (prevents camera jumps on updates)
    if not line1.empty and not line2.empty:
        center_x = float((line1.lon.mean() + line2.lon.mean()) / 2.0)
        center_y = float((line1.lat.mean() + line2.lat.mean()) / 2.0)
    elif not line1.empty:
        center_x = float(line1.lon.mean())
        center_y = float(line1.lat.mean())
    elif not df.empty and "lon" in df.columns:
        center_x = float(df.lon.mean())
        center_y = float(df.lat.mean())
    else:
        center_x, center_y = -3.6922, 40.4299

    zoom = float(zooms.get(line, 12.5))
    return df, center_x, center_y, zoom


def build_map(line_df, theme="dark"):
    """
    Returns a figure with the map of live location of buses (theme-aware).
    """
    if line_df.shape[0] < 1:
        return _empty_figure("No active buses on this line right now.")

    dark = theme == "dark"

    # Line and destinations
    line = line_df.iloc[0].line
    dest2, dest1 = lines_dict[line]["destinations"]

    # Select line line shapes
    line1 = line_shapes.loc[(line_shapes.line_sn == line) & (line_shapes.direction == 1)]
    line2 = line_shapes.loc[(line_shapes.line_sn == line) & (line_shapes.direction == 2)]

    # We drop the duplicated buses keeping the instance that is closer to a stop
    line_df = line_df.sort_values(by="DistanceBus").drop_duplicates(["bus"], keep="first")

    line_df, center_x, center_y, zoom = calc_map_params(line_df)

    # We create the figure object with theme-aware tiles
    new_map = go.Figure()
    map_style = mapbox_light_style if not dark else mapbox_style
    hover_bg = "#FFFFFF" if not dark else "#1E293B"
    hover_text = "#0F172A" if not dark else "#F8FAFC"
    new_map.update_layout(
        margin={"r": 0, "l": 0, "t": 0, "b": 0},
        hovermode="closest",
        showlegend=False,
        uirevision=str(line),
        hoverlabel={
            "bgcolor": hover_bg,
            "bordercolor": "rgba(139, 92, 246, 0.8)",
            "font": {
                "family": "Space Grotesk, sans-serif",
                "size": 13,
                "color": hover_text,
            },
        },
        map={
            "bearing": 0,
            "center": {"lat": center_y, "lon": center_x},
            "pitch": 0,
            "zoom": zoom,
            "style": map_style,
            "uirevision": str(line),
        },
    )

    # Select line stops
    if line_df[line_df.destination == dest1].shape[0] < 1:
        stop_names = lines_dict[line]["2"]["stops"][1:]
        lines_hovered = [line2]
    elif line_df[line_df.destination == dest2].shape[0] < 1:
        stop_names = lines_dict[line]["1"]["stops"][1:]
        lines_hovered = [line1]
    else:
        stop_names = lines_dict[line]["1"]["stops"][1:] + lines_dict[line]["2"]["stops"][1:]
        lines_hovered = [line1, line2]

    line_stops = stops.loc[stops.id.isin(stop_names)]

    # Add the stops to the figure
    stop_color = "#64748B" if dark else "#94A3B8"
    new_map.add_trace(
        go.Scattermap(
            lat=line_stops.lat,
            lon=line_stops.lon,
            mode="markers",
            marker=go.scattermap.Marker(size=8, color=stop_color, opacity=0.55),
            text=line_stops.id,
            hoverinfo="text",
            name="Stops",
        )
    )

    # Add lines to the figure (direction colored)
    for line_shape in lines_hovered:
        color = "#6366F1" if line_shape.iloc[0].direction == 1 else "#F59E0B"
        new_map.add_trace(
            go.Scattermap(
                lat=line_shape.lat,
                lon=line_shape.lon,
                mode="lines",
                line={"width": 3, "color": color},
                text=f"Route {line}-{line_shape.iloc[0].direction}",
                hoverinfo="skip",
                opacity=0.9,
                name=f"Route {line_shape.iloc[0].direction}",
            )
        )

    # Add the bus points to the figure with glowing markers
    for bus in line_df.itertuples():
        color = colors[bus.bus % len(colors)]
        new_map.add_trace(
            go.Scattermap(
                lat=[bus.lat],
                lon=[bus.lon],
                mode="markers",
                marker=go.scattermap.Marker(
                    size=16,
                    color=color,
                    opacity=0.95,
                ),
                text=[f"<b>Bus {bus.bus}</b><br>ETA: {bus.estimateArrive}s"],
                hoverinfo="text",
                name=f"Bus {bus.bus}",
            )
        )

    # And finally we return the map
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

    # Vectorized group iteration (100x faster than boolean slicing loops)
    for (bus1, bus2), group_df in series_df.groupby(["bus1", "bus2"], sort=False):
        if str(bus1) == "0" or str(bus2) == "0":
            continue
        group_df = group_df.sort_values("datetime")
        name = f"{bus1}-{bus2}"
        color_idx = (
            (str_to_int(bus1) + str_to_int(bus2)) % len(colors2)
            if "str_to_int" in globals()
            else (int(bus1) + int(bus2)) % len(colors2)
        )

        graph.add_trace(
            go.Scatter(
                name=name,
                x=group_df.datetime,
                y=group_df.hw12,
                mode="lines+markers",
                line={"width": 3, "color": colors2[color_idx]},
                text=[
                    f"<b>Bus group: {name}</b><br>Headway: {r.hw12}s<br>{r.datetime}"
                    for r in group_df.itertuples()
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
                line={"width": 3, "color": colors[group_df.bus2.iloc[0] % len(colors)]},
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

    # All bus names
    bus_names_all = ["bus" + str(i) for i in range(1, 8 + 2)]
    hw_names_all = ["hw" + str(i) + str(i + 1) for i in range(1, 8 + 1)]

    # Min and max datetimes
    min_time = series_df.datetime.min()
    max_time = series_df.datetime.max()

    # Locate unique groups
    unique_groups = []
    unique_groups_df = series_df.drop_duplicates(bus_names_all)
    for i in range(unique_groups_df.shape[0]):
        group = [unique_groups_df.iloc[i][bus_names_all[k]] for k in range(8 + 1)]
        unique_groups.append(group)

    last_dim = 0
    for group in unique_groups:
        # Build indexing conditions
        conds = [series_df[bus_names_all[k]] == group[k] for k in range(8 + 1)]
        final_cond = True
        for cond in conds:
            final_cond &= cond
        group_df = series_df.loc[final_cond]
        group_df = group_df.sort_values("datetime")

        # Dimension
        dim = group_df.iloc[0].dim
        color = colors[dim]

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
        for bus in group[1:]:
            if bus != 0:
                name += "-" + str(bus)
            else:
                break

        hw_values = []
        for _index, row in group_df.iterrows():
            hw_value = str(row.hw12)
            for hw_name in hw_names_all[1:dim]:
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
            if str(bus) != "0":
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


# CALLBACK 1 - Buses Position
@app.callback(
    [Output("map" + location, "figure")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_buses_position(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
    line = pathname.split("/")[-1] if pathname else ("1" if location == "Madrid" else "25")

    try:
        if "text" in hoverData["points"][0].keys():
            hover_buses = [
                int(hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0])
            ]
        else:
            hws_burst = read_df("hws_burst", line=line)

            dest = hoverData["points"][0]["y"][3:-1]
            x = hoverData["points"][0]["x"]

            direction = 1 if dest == lines_dict[line]["destinations"][1] else 2

            buses = hws_burst[
                (hws_burst.line == line)
                & (hws_burst.direction == direction)
                & (hws_burst.busB_ttls >= x)
            ].sort_values("busB_ttls")
            hover_buses = [buses.busA.iloc[0], buses.busB.iloc[0]]

    except:  # noqa: E722
        hover_buses = None

    burst = read_df("burst", line=line)

    # Line dataframe
    line_burst = burst.loc[burst.line == line]
    if hover_buses:
        line_burst = line_burst[line_burst.bus.isin(hover_buses)]

    if line_burst.shape[0] < 1:
        return [_empty_figure("No buses were found inside the line.")]

    # Create map (theme-aware tiles & markers)
    return [build_map(line_burst, theme)]


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

    try:
        try:
            if "text" in hoverData["points"][0].keys():
                hover_buses = [
                    int(hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0])
                ]
            else:
                hws_burst = read_df("hws_burst", line=line)

                dest = hoverData["points"][0]["y"][3:-1]
                x = hoverData["points"][0]["x"]

                direction = 1 if dest == lines_dict[line]["destinations"][1] else 2

                buses = hws_burst[
                    (hws_burst.line == line)
                    & (hws_burst.direction == direction)
                    & (hws_burst.busB_ttls >= x)
                ].sort_values("busB_ttls")
                hover_buses = [buses.busA.iloc[0], buses.busB.iloc[0]]
        except:  # noqa: E722
            hover_buses = None

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
    except:  # noqa: E722
        return [
            _empty_figure(
                "No headways to analyse. There are less than 2 buses inside each line direction."
            )
        ]


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
        Output("kpi-fleetMadrid", "children"),
        Output("kpi-headwayMadrid", "children"),
        Output("kpi-qosMadrid", "children"),
        Output("kpi-anomsMadrid", "children"),
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
    # Use visibility (not display) so Plotly keeps rendering hidden graphs with real dimensions
    hidden = {"visibility": "hidden", "position": "absolute", "height": "52vh", "width": "100%"}
    visible = {"height": "52vh", "position": "relative", "width": "100%"}
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
    current_line = pathname.split("/")[-1] if pathname else "1"
    all_lines = ["1", "44", "82", "132", "133"]
    pills = []
    for line in all_lines:
        is_active = line == current_line
        cls = "route-btn active" if is_active else "route-btn"
        pills.append(
            dcc.Link(
                f"Line {line}",
                href=f"/realtime/madrid/{line}",
                className=cls,
                style={"padding": "0.45rem 0.9rem", "minWidth": "0", "fontSize": "0.85rem"},
            )
        )
    return pills
