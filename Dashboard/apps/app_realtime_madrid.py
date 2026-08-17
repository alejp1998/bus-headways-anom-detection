import json
import math
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
        html.Div(
            className="box",
            children=[
                html.Div(
                    className="columns",
                    children=[
                        html.Div(id="tab-title" + location, className="column"),
                        html.Div(
                            id="conf" + location,
                            className="column",
                            style={"height": "4vh"},
                            children=[
                                dcc.Input(
                                    id="conf-slider" + location,
                                    type="text",
                                    value=0,
                                    style={"display": "none"},
                                )
                            ],
                        ),
                        html.Div(
                            id="size-th" + location,
                            className="column",
                            style={"height": "4vh"},
                            children=[
                                dcc.Input(
                                    id="size-th-slider" + location,
                                    type="text",
                                    value=0,
                                    style={"display": "none"},
                                )
                            ],
                        ),
                        html.Div(
                            className="column is-narrow",
                            style={"height": "4vh", "width": "7vh"},
                            children=[
                                dcc.Loading(
                                    id="new-interval-loading" + location,
                                    type="dot",
                                    style={"height": "4vh", "width": "7vh"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="column is-narrow",
                            style={"height": "0.5vh"},
                            children=[
                                html.Button(
                                    "Force Update",
                                    className="button",
                                    id="update-button" + location,
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="columns",
                    children=[
                        html.Div(
                            id="flat-hws-div" + location,
                            className="column",
                            children=[
                                dcc.Graph(
                                    id="flat-hws" + location,
                                    className="box",
                                    style={"height": "15vh"},
                                    figure=go.Figure(),
                                    clear_on_unhover=True,
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="columns",
                    children=[
                        html.Div(
                            id="buses-pos-div" + location,
                            className="column is-4",
                            children=[
                                dcc.Graph(
                                    id="map" + location,
                                    className="box",
                                    style={"height": box_height},
                                    figure=go.Figure(),
                                )
                            ],
                        ),
                        html.Div(
                            id="time-series-hws-div" + location,
                            className="column is-4",
                            children=[
                                dcc.Graph(
                                    id="time-series-hws" + location,
                                    className="box",
                                    style={"height": box_height},
                                    figure=go.Figure(),
                                )
                            ],
                        ),
                        html.Div(
                            id="2d-time-series-hws-div" + location,
                            className="column is-4",
                            children=[
                                dcc.Graph(
                                    id="2d-time-series-hws" + location,
                                    className="box",
                                    style={"height": box_height},
                                    figure=go.Figure(),
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="columns",
                    children=[
                        html.Div(id="mdist-hws-div" + location, className="column is-half"),
                        html.Div(id="anom-hws-div" + location, className="column is-half"),
                    ],
                ),
            ],
        ),
        html.Div(id="hidden-div" + location, style={"display": "none"}),
        dcc.Interval(
            id="interval-component" + location,
            interval=30 * 1000,  # in milliseconds
            n_intervals=0,
        ),
    ],
)

# MAPBOX CONFIGURATION (uses open carto-darkmatter tiles if no token configured)
mapbox_access_token = os.environ.get("MAPBOX_ACCESS_TOKEN", "")
style = (
    "carto-darkmatter"
    if not mapbox_access_token
    else "mapbox://styles/alejp1998/ck9voa0bb002y1ipcx8j00oeu"
)
# Plotly template is set globally in app.py (theme-aware); remove local override.


# FUNCTIONS
def read_df(name):
    if name == "burst":
        # Read last burst of data
        df = pd.read_csv(
            resolve_path(location + "/Data/RealTime/buses_data_burst_cleaned.csv"),
            dtype={
                "line": "str",
                "destination": "str",
                "stop": "uint16",
                "bus": "uint16",
                "given_coords": "bool",
                "pos_in_burst": "uint16",
                "estimateArrive": "int32",
                "DistanceBus": "int32",
                "request_time": "int32",
                "lat": "float32",
                "lon": "float32",
            },
        )[["line", "destination", "stop", "bus", "datetime", "estimateArrive", "DistanceBus"]]
        # Parse the dates
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S.%f")
    elif name == "hws_burst":
        # Read last processed headways
        df = pd.read_csv(
            resolve_path(location + "/Data/RealTime/headways_burst.csv"),
            dtype={
                "line": "str",
                "direction": "uint16",
                "busA": "uint16",
                "busB": "uint16",
                "headway": "int16",
                "busB_ttls": "uint16",
            },
        )[["line", "direction", "datetime", "hw_pos", "busA", "busB", "headway", "busB_ttls"]]
    elif name == "series":
        # Read last series data
        df = pd.read_csv(
            resolve_path(location + "/Data/RealTime/series.csv"), dtype={"line": "str"}
        )
    elif name == "anomalies":
        # Read last anomalies data
        df = pd.read_csv(
            resolve_path(location + "/Data/Anomalies/anomalies.csv"), dtype={"line": "str"}
        )
    return df


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
    # Select line line shapes
    line = df.line.iloc[0]
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
    center_x = df.lon.mean()
    center_y = df.lat.mean()

    if df.shape[0] >= 3:
        center_x = line1.lon.mean()
        center_y = line2.lat.mean()
        zoom = zooms[line]
    elif (df.shape[0] > 1) & (df.shape[0] < 3):
        zoom = min(
            max(
                3
                * math.log(
                    1
                    / max(
                        (
                            min(
                                math.sqrt(
                                    (max(lons) - min(lons)) ** 2 + (max(lats) - min(lats)) ** 2
                                ),
                                1,
                            )
                        ),
                        0.0001,
                    )
                ),
                zooms[line],
            ),
            13.5,
        )
    else:
        zoom = 14
    return df, center_x, center_y, zoom


def build_map(line_df):
    """
    Returns a figure with the map of live location of buses
    """
    if line_df.shape[0] < 1:
        return "EMPTY"

    # Line and destinations
    line = line_df.iloc[0].line
    dest2, dest1 = lines_dict[line]["destinations"]

    # Select line line shapes
    line1 = line_shapes.loc[(line_shapes.line_sn == line) & (line_shapes.direction == 1)]
    line2 = line_shapes.loc[(line_shapes.line_sn == line) & (line_shapes.direction == 2)]

    # We drop the duplicated buses keeping the instance that is closer to a stop
    line_df = line_df.sort_values(by="DistanceBus").drop_duplicates(["bus"], keep="first")

    line_df, center_x, center_y, zoom = calc_map_params(line_df)

    # We create the figure object
    new_map = go.Figure()

    # And set the figure layout
    new_map.update_layout(
        title="<b>BUSES POSITION</b>",
        margin={"r": 0, "l": 0, "t": 40, "b": 0},
        hovermode="closest",
        showlegend=False,
        mapbox={
            "accesstoken": mapbox_access_token if mapbox_access_token else "pk.placeholder",
            "bearing": 0,
            "center": {"lat": center_y, "lon": center_x},
            "pitch": 0,
            "zoom": zoom,
            "style": "carto-darkmatter" if not mapbox_access_token else "streets",
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
    new_map.add_trace(
        go.Scattermapbox(
            lat=line_stops.lat,
            lon=line_stops.lon,
            mode="markers",
            marker=go.scattermapbox.Marker(size=10, color="#2F4F4F", opacity=0.5),
            text=line_stops.id,
            hoverinfo="text",
        )
    )

    # Add lines to the figure
    for line_shape in lines_hovered:
        color = "#1E90FF" if line_shape.iloc[0].direction == 1 else "#B22222"
        new_map.add_trace(
            go.Scattermapbox(
                lat=line_shape.lat,
                lon=line_shape.lon,
                mode="lines",
                line={"width": 2, "color": color},
                text=f"Línea : {line}-{line_shape.iloc[0].direction}",
                hoverinfo="skip",
                opacity=1,
            )
        )

    # Add the bus points to the figure
    for bus in line_df.itertuples():
        # Assign color based on bus id
        color = colors[bus.bus % len(colors)]
        # Bus marker
        new_map.add_trace(
            go.Scattermapbox(
                lat=[bus.lat],
                lon=[bus.lon],
                mode="markers",
                marker=go.scattermapbox.Marker(size=27, opacity=1, color=color),
                text=[bus.bus],
                hoverinfo="text",
            )
        )
        new_map.add_trace(
            go.Scattermapbox(
                mode="text", text=str(bus.bus), lat=[bus.lat], lon=[bus.lon], hoverinfo="none"
            )
        )

    # And finally we return the map
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
                            (hw1.iloc[i + 1].busA + hw1.iloc[i + 1].busB) % len(colors2)
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
                            (hw2.iloc[i + 1].busA + hw2.iloc[i + 1].busB) % len(colors2)
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
        color = colors[bus.busB % len(colors)]

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

    graph.update_layout(xaxis_range=(0, max_ttls[line]))

    # Finally we return the graph
    return graph


def build_time_series_graph(series_df, model, conf):
    graph = go.Figure()

    # Set title and layout
    graph.update_layout(
        title="<b>1D HEADWAYS TIME SERIES</b> - (In seconds)",
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
                        (group_df.bus1.iloc[0] + group_df.bus2.iloc[0]) % len(colors2)
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

    # Read dict
    while True:
        try:
            with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
                hyperparams = json.load(f)
            conf = hyperparams[line]["conf"]
            break
        except:  # noqa: E722
            continue

    # Set title and layout
    graph.update_layout(
        title="<b>MAHALANOBIS DISTANCE</b>",
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
    # All bus names
    bus_names_all = ["bus" + str(i) for i in range(1, 8 + 2)]
    ["hw" + str(i) + str(i + 1) for i in range(1, 8 + 1)]

    if anomalies_df.shape[0] < 1:
        return "No anomalies were detected yet."

    # Build group names
    names = []
    for i in range(anomalies_df.shape[0]):
        group = [anomalies_df.iloc[i][bus_names_all[k]] for k in range(8 + 1)]
        name = str(group[0])
        for bus in group[1:]:
            if bus != 0:
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


# CALLBACKS


# CALLBACK 0a - New interval loading
@app.callback(
    [Output("new-interval-loading" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
    ],
)
def new_interval(n_intervals, n_clicks):
    return [html.H1("Loading", style={"display": "none"})]


# CALLBACK 0b - Title and sliders
@app.callback(
    [
        Output("tab-title" + location, "children"),
        Output("conf" + location, "children"),
        Output("size-th" + location, "children"),
    ],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
    ],
)
def update_title_sliders(n_intervals, n_clicks, pathname):
    line = pathname[17:]

    now = dt.now()
    now = now.replace(microsecond=0)

    with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
        hyperparams = json.load(f)

    conf = hyperparams[line]["conf"]
    size_th = hyperparams[line]["size_th"]

    # And return all of them
    switcher = html.Div(
        className="flex-gap flex-wrap",
        style={"marginTop": "0.5rem"},
        children=[
            html.Span(
                "Madrid Lines:",
                style={"color": "var(--text-muted)", "fontSize": "0.85rem", "fontWeight": "600"},
            ),
            dcc.Link(
                "Line 1",
                href="/realtime/madrid/1",
                className=f"badge-pill {'primary' if line == '1' else 'warning'}",
            ),
            dcc.Link(
                "Line 44",
                href="/realtime/madrid/44",
                className=f"badge-pill {'primary' if line == '44' else 'warning'}",
            ),
            dcc.Link(
                "Line 82",
                href="/realtime/madrid/82",
                className=f"badge-pill {'primary' if line == '82' else 'warning'}",
            ),
            dcc.Link(
                "Line 132",
                href="/realtime/madrid/132",
                className=f"badge-pill {'primary' if line == '132' else 'warning'}",
            ),
            dcc.Link(
                "Line 133",
                href="/realtime/madrid/133",
                className=f"badge-pill {'primary' if line == '133' else 'warning'}",
            ),
        ],
    )
    return [
        [html.H1(f"Madrid Line {line} ({now.time()})", className="title is-3"), switcher],
        [
            html.Label(
                [
                    "Confidence",
                    dcc.Slider(
                        id="conf-slider" + location,
                        min=90,
                        max=100,
                        step=0.05,
                        marks={i: str(i) + "%" for i in [90 + k * 1 for k in range(11)]},
                        value=conf * 100,
                    ),
                ],
            )
        ],
        [
            html.Label(
                [
                    "Size threshold",
                    dcc.Slider(
                        id="size-th-slider" + location,
                        min=1,
                        max=15,
                        marks={i: str(i) for i in range(1, 16)},
                        value=size_th,
                    ),
                ],
            )
        ],
    ]


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
    line = pathname[17:]
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


# CALLBACK 1 - Buses Position
@app.callback(
    [Output("buses-pos-div" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_buses_position(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
    line = pathname[17:]

    try:
        if "text" in hoverData["points"][0].keys():
            hover_buses = [
                int(hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0])
            ]
        else:
            hws_burst = read_df("hws_burst")

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

    burst = read_df("burst")

    # Line dataframe
    line_burst = burst.loc[burst.line == line]
    if hover_buses:
        line_burst = line_burst[line_burst.bus.isin(hover_buses)]

    if line_burst.shape[0] < 1:
        return [html.H1("No buses were found inside the line.", className="title is-5")]

    # Create map
    new_map = build_map(line_burst)

    # And return all of them
    return [
        dcc.Graph(
            id="map" + location,
            className="box",
            style={"height": box_height},
            figure=new_map,
            config={"displayModeBar": False},
        )
    ]


# CALLBACK 2 - Buses headways representation
@app.callback(
    [Output("flat-hws-div" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
    ],
)
def update_flat_hws(n_intervals, n_clicks, pathname, theme="dark"):
    line = pathname[17:]

    hws_burst = read_df("hws_burst")

    line_hws = hws_burst.loc[hws_burst.line == line]

    # Create graph
    flat_hws_graph = build_graph(line_hws)

    graph = dcc.Graph(
        id="flat-hws" + location,
        className="box",
        style={"height": "15vh"},
        figure=flat_hws_graph,
        config={
            "displayModeBar": False,
        },
        clear_on_unhover=True,
    )

    # And return all of them
    graph.figure.update_layout(**theme_layout(theme))

    return [graph]


# CALLBACK 3 - 1D Headways Time Series
@app.callback(
    [Output("time-series-hws-div" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_time_series_hws(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
    line = pathname[17:]

    try:
        if "text" in hoverData["points"][0].keys():
            hover_buses = [
                int(hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0])
            ]
        else:
            hws_burst = read_df("hws_burst")

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

    series = read_df("series")

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
            html.H1(
                "No headways to analyse. There are less than 2 buses inside each line direction.",
                className="title is-5",
            )
        ]

    now = dt.now()
    # Day type
    if (now.weekday() >= 0) and (now.weekday() <= 4):
        day_type = "LA"
    elif now.weekday() == 5:
        day_type = "SA"
    else:
        day_type = "FE"

    # Hour ranges to iterate over
    # hour_ranges = [[7,11], [11,15], [15,19], [19,23]]
    hour_ranges = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]

    # Hour range
    for h_range in hour_ranges:
        if (now.hour >= h_range[0]) and (now.hour < h_range[1]):
            hour_range = str(h_range[0]) + "-" + str(h_range[1])
            break
        elif h_range == hour_ranges[-1]:
            return [
                html.H1(
                    f"Hour range for {now.hour}:{now.minute} not defined. Waiting till 7am.",
                    className="subtitle is-3",
                )
            ]

    model = models_params_dict[line][day_type][hour_range]["1"]

    # Read dict
    while True:
        try:
            with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
                hyperparams = json.load(f)
            conf = hyperparams[line]["conf"]
            break
        except:  # noqa: E722
            continue

    time_series_graph = build_time_series_graph(line_series, model, conf)

    graph = dcc.Graph(
        id="time-series-hws" + location,
        className="box",
        style={"height": box_height},
        figure=time_series_graph,
        config={"displayModeBar": False},
    )

    # if n_intervals == 0 :
    # graph = dcc.Loading(type='cube',children = [graph])

    # And return all of them
    graph.figure.update_layout(**theme_layout(theme))

    return [graph]


# CALLBACK 4 - 2D Headways Time Series
@app.callback(
    [Output("2d-time-series-hws-div" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_2d_time_series_hws(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
    line = pathname[17:]

    try:
        if "text" in hoverData["points"][0].keys():
            hover_buses = [
                int(hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0])
            ]
        else:
            hws_burst = read_df("hws_burst")

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

    series = read_df("series")

    line_series = series.loc[(series.line == line) & (series.dim == 2)]

    if hover_buses:
        if len(hover_buses) == 1:
            line_series = line_series.loc[(line_series.bus2 == hover_buses[0])]
        elif len(hover_buses) == 2:
            return [html.H1("Click a bus, links not supported.", className="title is-5")]

    if line_series.shape[0] < 1:
        return [
            html.H1(
                "No 2d headways to analyse. Click a bus between two buses.", className="title is-5"
            )
        ]

    now = dt.now()
    # Day type
    if (now.weekday() >= 0) and (now.weekday() <= 4):
        day_type = "LA"
    elif now.weekday() == 5:
        day_type = "SA"
    else:
        day_type = "FE"

    # Hour ranges to iterate over
    # hour_ranges = [[7,11], [11,15], [15,19], [19,23]]
    hour_ranges = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]

    # Hour range
    for h_range in hour_ranges:
        if (now.hour >= h_range[0]) and (now.hour < h_range[1]):
            hour_range = str(h_range[0]) + "-" + str(h_range[1])
            break
        elif h_range == hour_ranges[-1]:
            return [
                html.H1(
                    f"Hour range for {now.hour}:{now.minute} not defined. Waiting till 7am.",
                    className="subtitle is-3",
                )
            ]

    try:
        model = models_params_dict[line][day_type][hour_range]["2"]
    except:  # noqa: E722
        return [html.H1("2D Model for this hour range not available.", className="subtitle is-3")]

    # Read dict
    while True:
        try:
            with open(resolve_path(location + "/Data/Anomalies/hyperparams.json")) as f:
                hyperparams = json.load(f)
            conf = hyperparams[line]["conf"]
            break
        except:  # noqa: E722
            continue

    time_series_graph = build_2d_time_series_graph(line_series, model, conf)

    graph = dcc.Graph(
        id="2d-time-series-hws" + location,
        className="box",
        style={"height": box_height},
        figure=time_series_graph,
        config={"displayModeBar": False},
    )

    # if n_intervals == 0 :
    # graph = dcc.Loading(type='cube',children = [graph])

    # And return all of them
    graph.figure.update_layout(**theme_layout(theme))

    return [graph]


# CALLBACK 5 - Mahalanobis Distance series
@app.callback(
    [Output("mdist-hws-div" + location, "children")],
    [
        Input("interval-component" + location, "n_intervals"),
        Input("update-button" + location, "n_clicks"),
        Input("url", "pathname"),
        Input("theme-store", "data"),
        Input("flat-hws" + location, "clickData"),
    ],
)
def update_mdist_series(n_intervals, n_clicks, pathname, hoverData, theme="dark"):
    try:
        line = pathname[17:]

        try:
            if "text" in hoverData["points"][0].keys():
                hover_buses = [
                    int(hoverData["points"][0]["text"].split("<b>Bus: ")[1].split("</b>")[0])
                ]
            else:
                hws_burst = read_df("hws_burst")

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

        series = read_df("series")

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
                html.H1(
                    "No headways to analyse. There are less than 2 buses inside each line direction.",
                    className="title is-5",
                )
            ]

        # Create mh dist graph
        m_dist_graph = build_m_dist_graph(line_series, line)

        graph = dcc.Graph(
            id="mdist-hws" + location,
            className="box",
            style={"height": box_height},
            figure=m_dist_graph,
            config={"displayModeBar": False},
        )

        # if n_intervals == 0 :
        # graph = dcc.Loading(type='cube',children = [graph])

        # And return all of them
        graph.figure.update_layout(**theme_layout(theme))

        return [graph]
    except:  # noqa: E722
        return [
            html.H1(
                "No headways to analyse. There are less than 2 buses inside each line direction.",
                className="title is-5",
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
    line = pathname[17:]

    anomalies = read_df("anomalies")

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
