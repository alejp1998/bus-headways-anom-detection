import json
import math
import os.path
import time
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2

# Lines to iterate over
lines = ["1", "44", "82", "132", "133"]
# Day types to iterate over
day_types = ["LA", "SA", "FE"]
# Hour ranges to iterate over
hour_ranges = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]
# hour_ranges = [[i,i+1] for i in range(7,23)]


# Lines collected dictionary
with open("../../Data/Static/lines_dict.json") as f:
    lines_dict = json.load(f)
# Models parameters dictionary
with open("../../Data/Anomalies/models_params.json") as f:
    models_params_dict = json.load(f)
# Load times between stops data
times_bt_stops = pd.read_csv(
    "../../Data/Processed/times_bt_stops.csv",
    dtype={
        "line": "str",
        "direction": "uint16",
        "st_hour": "uint16",
        "end_hour": "uint16",
        "stopA": "uint16",
        "stopB": "uint16",
        "bus": "uint16",
        "trip_time": "float16",
        "api_trip_time": "int16",
    },
)
# Parse the dates
times_bt_stops["date"] = pd.to_datetime(times_bt_stops["date"], format="%Y-%m-%d")


# Day types
day_type_dict = {  # 0 = Monday, 1 = Tuesday ...
    "LA": [0, 1, 2, 3, 4],  # LABORABLES
    "LJ": [0, 1, 2, 3],  # LUNES A JUEVES
    "VV": [4],  # VIERNES
    "SA": [5],  # SABADOS
    "FE": [6],  # DOMIGOS O FESTIVOS
}

# Bus and headways column names
bus_names_all = ["bus" + str(i) for i in range(1, 8 + 2)]
hw_names_all = ["hw" + str(i) + str(i + 1) for i in range(1, 8 + 1)]


def clean_data(df):
    """Vectorized validation and cleaning of raw bus telemetry records."""
    if df.empty:
        return df

    valid_mask = pd.Series(False, index=df.index)
    lines_series = df["line"].astype(str)

    for line, ldata in lines_dict.items():
        line_mask = lines_series == line
        if not line_mask.any():
            continue
        destinations = ldata.get("destinations", [])
        if len(destinations) < 2:
            continue
        dest2, dest1 = destinations[0], destinations[1]

        for d_key, dest_val in [("1", dest1), ("2", dest2)]:
            dir_mask = line_mask & (df["destination"] == dest_val)
            if not dir_mask.any():
                continue
            stops_set = {int(s) for s in ldata.get(d_key, {}).get("stops", [])}
            line_len = int(ldata.get(d_key, {}).get("length", 0))

            sub_df = df.loc[dir_mask]
            stop_cond = sub_df["stop"].astype(int).isin(stops_set)
            dist = sub_df["DistanceBus"].astype(float)
            eta = sub_df["estimateArrive"].astype(float)

            dist_cond = (dist >= 0) & (dist < line_len)
            speed_cond = (3.6 * dist / np.maximum(eta, 1e-5)) < 120
            eta_cond = (eta > 0) & (eta < (line_len / 2)) & speed_cond & (eta < 7200)

            valid_mask.loc[dir_mask] = stop_cond & dist_cond & eta_cond

    return df.loc[valid_mask].reset_index(drop=True)


# For every burst of data:
def get_headways(int_df, day_type, hour_range, ap_order_dict, now):
    rows_list = []
    hour_range = [int(hour) for hour in hour_range.split("-")]

    # Drop duplicate buses keeping the one with lowest estimateArrive
    int_df = int_df.sort_values("estimateArrive").drop_duplicates("bus", keep="first")

    # Burst time
    actual_time = int_df.iloc[0].datetime

    # Line
    line = int_df.iloc[0].line

    # Stops of each line reversed
    stops1 = lines_dict[str(line)]["1"]["stops"][-3:3:-1]
    stops2 = lines_dict[str(line)]["2"]["stops"][-3:3:-1]

    # Appearance order buses list
    ap_order_dir1 = ap_order_dict[line]["dir1"]
    ap_order_dir2 = ap_order_dict[line]["dir2"]
    bus_ttls1 = ap_order_dict[line]["ttls1"]
    bus_ttls2 = ap_order_dict[line]["ttls2"]
    bus_cons_disap1 = ap_order_dict[line]["cons_disap1"]
    bus_cons_disap2 = ap_order_dict[line]["cons_disap2"]
    bus_cons_ap1 = ap_order_dict[line]["cons_ap1"]
    bus_cons_ap2 = ap_order_dict[line]["cons_ap2"]

    # Assign destination values
    dest2, dest1 = lines_dict[line]["destinations"]

    # Add direction field to df (vectorized)
    int_df["direction"] = np.where(int_df["destination"] == dest1, 1, 2)

    # Process mean times between stops
    tims_bt_stops = times_bt_stops.loc[
        (times_bt_stops.line == line)
        & (times_bt_stops.date.dt.weekday.isin(day_type_dict[day_type]))
        & (times_bt_stops.st_hour >= hour_range[0])
        & (times_bt_stops.st_hour < hour_range[1])
    ]
    # Group and get the mean values
    tims_bt_stops = tims_bt_stops.groupby(["line", "direction", "stopA", "stopB"]).mean()
    tims_bt_stops = tims_bt_stops.reset_index()[
        ["line", "direction", "stopA", "stopB", "trip_time", "api_trip_time"]
    ]

    # All stops of the line
    stops = stops1 + stops2
    stop_df_list = []
    buses_out1, buses_out2 = [], []
    direction = 1
    for i in range(len(stops)):
        stop = stops[i]
        if i == 0:
            mean_time_to_stop = 0
        elif i == len(stops1):
            mean_time_to_stop_1 = mean_time_to_stop
            mean_time_to_stop = 0
            direction = 2
        else:
            mean_df = tims_bt_stops.loc[
                (tims_bt_stops.stopA == int(stop)) & (tims_bt_stops.direction == direction)
            ]
            if mean_df.shape[0] > 0:
                mean_time_to_stop += mean_df.iloc[0].trip_time
            else:
                continue

        stop_df = int_df.loc[(int_df.stop == int(stop)) & (int_df.direction == direction)]

        # Drop duplicates, recalculate estimateArrive and append to list
        stop_df = stop_df.drop_duplicates("bus", keep="first")

        if (stop == stops1[-1]) or (stop == stops2[-1]):
            if direction == 1:
                buses_out1 += stop_df.bus.unique().tolist()
            else:
                buses_out2 += stop_df.bus.unique().tolist()
        elif (stop == stops1[0]) or (stop == stops2[0]):
            buses_near = stop_df.loc[stop_df.estimateArrive < 60]
            if buses_near.shape[0] > 0:
                if direction == 1:
                    buses_out1 += buses_near.bus.unique().tolist()
                else:
                    buses_out2 += buses_near.bus.unique().tolist()

        stop_df.estimateArrive = stop_df.estimateArrive + mean_time_to_stop
        stop_df_list.append(stop_df)

    mean_time_to_stop_2 = mean_time_to_stop

    # Concatenate and group them
    stops_df = pd.concat(stop_df_list)

    # Eliminate TTLS longer than mean time to go through hole line direction
    stops_df = stops_df[
        (stops_df.direction == 1) & (stops_df.estimateArrive < mean_time_to_stop_1)
        | (stops_df.direction == 2) & (stops_df.estimateArrive < mean_time_to_stop_2)
    ]

    # Group by bus and direction
    # stops_df = stops_df.groupby(['bus','direction']).mean().sort_values(by=['estimateArrive'])
    stops_df = stops_df.sort_values(by=["estimateArrive"])
    stops_df = stops_df.reset_index().drop_duplicates("bus", keep="first")
    # Loc buses not given by first stop
    stops_df = stops_df.loc[
        ((stops_df.direction == 1) & (~stops_df.bus.isin(buses_out1)))
        | ((stops_df.direction == 2) & (~stops_df.bus.isin(buses_out2)))
    ]

    # Update appearance order lists
    if (ap_order_dir1 == []) & (ap_order_dir2 == []):
        TH = 0
    else:
        TH = 2

    # Set min difference bt last TH TTLSs of a bus to consider it is moving
    2 * TH

    stops_df_dest1 = stops_df[stops_df.direction == 1].sort_values(by=["estimateArrive"])
    if stops_df_dest1.shape[0] > 0:
        buses_dest1 = stops_df_dest1.bus.tolist()
        etas_dest1 = stops_df_dest1.estimateArrive.tolist()
        for i in range(len(buses_dest1)):
            if buses_dest1[i] not in bus_ttls1:
                bus_ttls1[buses_dest1[i]] = []
                bus_ttls1[buses_dest1[i]].append(etas_dest1[i])
            elif len(bus_ttls1[buses_dest1[i]]) < TH:
                bus_ttls1[buses_dest1[i]].append(etas_dest1[i])
            else:
                bus_ttls1[buses_dest1[i]].pop(0)
                bus_ttls1[buses_dest1[i]].append(etas_dest1[i])

            max(bus_ttls1[buses_dest1[i]]) - min(bus_ttls1[buses_dest1[i]])

            if buses_dest1[i] not in bus_cons_ap1:
                bus_cons_ap1[buses_dest1[i]] = 0
            if buses_dest1[i] not in bus_cons_disap1:
                bus_cons_disap1[buses_dest1[i]] = 0

            if bus_cons_ap1[buses_dest1[i]] > TH:
                if buses_dest1[i] not in ap_order_dir1:
                    # Append to apearance list
                    if i > 0:
                        bus_bef = buses_dest1[i - 1]
                        for k in range(len(ap_order_dir1)):
                            if ap_order_dir1[k] == bus_bef:
                                ap_order_dir1.insert(k + 1, buses_dest1[i])
                    else:
                        ap_order_dir1.insert(0, buses_dest1[i])

        # Update times without appering
        for bus in bus_cons_disap1:
            if bus not in buses_dest1:
                bus_cons_disap1[bus] += 1
                bus_cons_ap1[bus] = 0
                if bus_cons_disap1[bus] > TH:
                    if bus in ap_order_dir1:
                        ap_order_dir1.remove(bus)
            else:
                bus_cons_disap1[bus] = 0
                bus_cons_ap1[bus] += 1

    stops_df_dest2 = stops_df[stops_df.direction == 2].sort_values(by=["estimateArrive"])
    if stops_df_dest2.shape[0] > 0:
        buses_dest2 = stops_df_dest2.bus.tolist()
        etas_dest2 = stops_df_dest2.estimateArrive.tolist()
        for i in range(len(buses_dest2)):
            if buses_dest2[i] not in bus_ttls2:
                bus_ttls2[buses_dest2[i]] = []
                bus_ttls2[buses_dest2[i]].append(etas_dest2[i])
            elif len(bus_ttls2[buses_dest2[i]]) < TH:
                bus_ttls2[buses_dest2[i]].append(etas_dest2[i])
            else:
                bus_ttls2[buses_dest2[i]].pop(0)
                bus_ttls2[buses_dest2[i]].append(etas_dest2[i])

            max(bus_ttls2[buses_dest2[i]]) - min(bus_ttls2[buses_dest2[i]])

            if buses_dest2[i] not in bus_cons_ap2:
                bus_cons_ap2[buses_dest2[i]] = 0
            if buses_dest2[i] not in bus_cons_disap2:
                bus_cons_disap2[buses_dest2[i]] = 0

            if bus_cons_ap2[buses_dest2[i]] > TH:
                if buses_dest2[i] not in ap_order_dir2:
                    # Append to apearance list
                    if i > 0:
                        bus_bef = buses_dest2[i - 1]
                        for k in range(len(ap_order_dir2)):
                            if ap_order_dir2[k] == bus_bef:
                                ap_order_dir2.insert(k + 1, buses_dest2[i])

                    else:
                        ap_order_dir2.insert(0, buses_dest2[i])

        # Update times without appering
        for bus in bus_cons_disap2:
            if bus not in buses_dest2:
                bus_cons_disap2[bus] += 1
                bus_cons_ap2[bus] = 0
                if bus_cons_disap2[bus] > TH:
                    if bus in ap_order_dir2:
                        ap_order_dir2.remove(bus)
            else:
                bus_cons_disap2[bus] = 0
                bus_cons_ap2[bus] += 1

    # Reorder df according to appearance list
    rows, last_ttls1, last_bus1, last_ttls2, last_bus2 = [], 0, 0, 0, 0
    for bus in ap_order_dir1:
        bus_df = stops_df_dest1[stops_df_dest1.bus == bus]
        if bus_df.shape[0] > 0:
            if (last_ttls1 > bus_df.iloc[0].estimateArrive) & (TH + 1 < bus_cons_ap1[bus] < TH + 4):
                rows.pop(-1)
                rows.append(bus_df.iloc[0])

                ap_order_dir1.remove(last_bus1)
                bus_cons_ap1[last_bus1] = TH + 1
                bus_cons_disap1[last_bus1] = TH + 1
                break
            else:
                rows.append(bus_df.iloc[0])
                last_ttls1 = bus_df.iloc[0].estimateArrive
                last_bus1 = bus

    for bus in ap_order_dir2:
        bus_df = stops_df_dest2[stops_df_dest2.bus == bus]
        if bus_df.shape[0] > 0:
            if (last_ttls2 > bus_df.iloc[0].estimateArrive) & (TH + 1 < bus_cons_ap2[bus] < TH + 4):
                rows.pop(-1)
                rows.append(bus_df.iloc[0])

                ap_order_dir2.remove(last_bus2)
                bus_cons_ap2[last_bus2] = TH + 1
                bus_cons_disap2[last_bus2] = TH + 1
                break
            else:
                rows.append(bus_df.iloc[0])
                last_ttls2 = bus_df.iloc[0].estimateArrive
                last_bus2 = bus

    stops_df = pd.DataFrame(rows)

    # Update in dict
    ap_order_dict[line]["dir1"] = ap_order_dir1
    ap_order_dict[line]["dir2"] = ap_order_dir2
    ap_order_dict[line]["ttls1"] = bus_ttls1
    ap_order_dict[line]["ttls2"] = bus_ttls2
    ap_order_dict[line]["cons_disap1"] = bus_cons_disap1
    ap_order_dict[line]["cons_disap2"] = bus_cons_disap2
    ap_order_dict[line]["cons_ap1"] = bus_cons_ap1
    ap_order_dict[line]["cons_ap2"] = bus_cons_ap2

    """
    print('\n[MADRID] Line {} - Apearance Order Dict'.format(line))

    print('\nAp Order List')
    print(ap_order_dir1)
    print(ap_order_dir2)

    print('\nLast TTLSs')
    print(bus_ttls1)
    print(bus_ttls2)

    print('\nConsecutive Disapearances')
    print(bus_cons_disap1)
    print(bus_cons_disap2)

    print('\nConsecutive Apearances')
    print(bus_cons_ap1)
    print(bus_cons_ap2)
    """

    # Calculate time intervals
    if stops_df.shape[0] > 0:
        hw_pos1 = 0
        hw_pos2 = 0
        for i in range(stops_df.shape[0]):
            est1 = stops_df.iloc[i]

            direction = est1.direction
            if ((direction == 1) & (hw_pos1 == 0)) or ((direction == 2) & (hw_pos2 == 0)):
                # Create dataframe row
                row = {}
                row["datetime"] = actual_time
                row["line"] = line
                row["direction"] = direction
                row["busA"] = 0
                row["busB"] = est1.bus
                row["hw_pos"] = 0
                row["headway"] = 0
                row["busB_ttls"] = int(est1.estimateArrive)

                # Append row to the list of rows
                rows_list.append(row)

                # Increment hw pos
                if direction == 1:
                    hw_pos1 += 1
                else:
                    hw_pos2 += 1

            if i < (stops_df.shape[0] - 1):
                est2 = stops_df.iloc[i + 1]
            else:
                break

            if est1.direction == est2.direction:
                headway = int(est2.estimateArrive - est1.estimateArrive)

                # Create dataframe row
                row = {}
                row["datetime"] = actual_time
                row["line"] = line
                row["direction"] = direction
                row["busA"] = est1.bus
                row["busB"] = est2.bus
                row["hw_pos"] = hw_pos1 if direction == 1 else hw_pos2
                row["headway"] = headway
                row["busB_ttls"] = int(est2.estimateArrive)

                # Append row to the list of rows
                rows_list.append(row)

                # Increment hw pos
                if direction == 1:
                    hw_pos1 += 1
                else:
                    hw_pos2 += 1

    if len(rows_list) < 1:
        headways_df = pd.DataFrame(
            columns=[
                "datetime",
                "line",
                "direction",
                "busA",
                "busB",
                "hw_pos",
                "headway",
                "busB_ttls",
            ]
        )
    else:
        headways_df = pd.DataFrame(rows_list)

    return headways_df, ap_order_dict


def get_ndim_hws(df, dim, now):
    # Generate names for the columns of the dataframe to be built
    hw_names = ["hw" + str(i) + str(i + 1) for i in range(1, dim + 1)]
    bus_names = ["bus" + str(i) for i in range(1, dim + 2)]

    # Columns to build dictionary
    columns = {}
    names = ["datetime"] + bus_names + hw_names
    for name in names:
        columns[name] = []

    if df.shape[0] < 1:
        return pd.DataFrame(columns=names)

    # Unique datetime identifiers for the bursts
    burst_time = df.iloc[0].datetime

    burst_df1 = df.loc[(df.datetime == burst_time) & (df.direction == 1)].sort_values("hw_pos")
    burst_df2 = df.loc[(df.datetime == burst_time) & (df.direction == 2)].sort_values("hw_pos")

    for i in range(max(burst_df1.shape[0], burst_df2.shape[0]) - (dim - 1)):
        if i < (burst_df1.shape[0] - (dim - 1)):
            columns["datetime"].append(burst_time)
            columns[bus_names[0]].append(burst_df1.iloc[i].busA)
            for k in range(dim):
                columns[hw_names[k]].append(burst_df1.iloc[i + k].headway)
                columns[bus_names[k + 1]].append(burst_df1.iloc[i + k].busB)

        if i < (burst_df2.shape[0] - (dim - 1)):
            columns["datetime"].append(burst_time)
            columns[bus_names[0]].append(burst_df2.iloc[i].busA)
            for k in range(dim):
                columns[hw_names[k]].append(burst_df2.iloc[i + k].headway)
                columns[bus_names[k + 1]].append(burst_df2.iloc[i + k].busB)

    return pd.DataFrame(columns)


def process_hws_ndim_mh_dist(lines, day_type, hour_range, burst_df, ap_order_dict, now):
    windows_dfs, headways_dfs = [], []

    # Read dict
    while True:
        try:
            with open("../../Data/Anomalies/hyperparams.json") as f:
                hyperparams = json.load(f)
            break
        except:  # noqa: E722
            continue

    # For every line
    for line in lines:
        conf = hyperparams[line]["conf"]

        # Process the headways
        line_df = burst_df.loc[burst_df.line == line]

        # Calculate headways and append them
        if line_df.shape[0] > 0:
            headways, ap_order_dict = get_headways(
                line_df, day_type, hour_range, ap_order_dict, now
            )
            headways["line"] = line
            headways_dfs.append(headways)
        else:
            continue

        if headways.shape[0] < 1:
            continue

        # Eliminate hw_pos == 0
        hws = headways.loc[headways.hw_pos > 0]

        # Max dimensional model trained
        models = models_params_dict[line][day_type][hour_range]
        max_dim = models["max_dim"]

        # Process windows dfs
        dim, window_data_points = 1, 1
        while (window_data_points > 0) and (dim <= max_dim):
            window_df = get_ndim_hws(hws, dim, now)
            window_data_points = window_df.shape[0]

            hw_names = ["hw" + str(i) + str(i + 1) for i in range(1, dim + 1)]
            ["bus" + str(i) for i in range(1, dim + 2)]

            hw_names_rest = ["hw" + str(i) + str(i + 1) for i in range(dim + 1, 8 + 1)]
            bus_names_rest = ["bus" + str(i) for i in range(dim + 2, 8 + 2)]

            cov_matrix = models[str(dim)]["cov_matrix"]
            mean = models[str(dim)]["mean"]

            # Get Mahalanobis distance squared from which a certain percentage of the data is out
            m_th = math.sqrt(chi2.ppf(conf, df=dim))

            # Vectorized Mahalanobis distance calculation
            X = window_df[hw_names].to_numpy(dtype=float)
            if dim == 1:
                std = float(cov_matrix)
                m_dist = np.abs(X[:, 0] - float(mean)) / (std if std != 0 else 1.0)
            else:
                iv = np.linalg.pinv(cov_matrix)
                diff = X - mean
                m_dist = np.sqrt(np.maximum(0.0, np.sum((diff @ iv) * diff, axis=1)))

            window_df["m_dist"] = np.round(m_dist, 5)
            window_df["anom"] = (window_df["m_dist"] > m_th).astype(int)
            window_df["line"] = line
            window_df["dim"] = dim

            # Set to 0 unused columns
            for name in hw_names_rest + bus_names_rest:
                window_df[name] = 0

            window_df = window_df[
                ["line", "datetime", "dim", "m_dist", "anom"] + bus_names_all + hw_names_all
            ]
            windows_dfs.append(window_df)

            # Increase dim
            dim += 1

    # Concat them in a dataframe
    headways_df = pd.concat(headways_dfs)

    if len(windows_dfs) > 0:
        windows_df = pd.concat(windows_dfs)
    else:
        windows_df = pd.DataFrame(
            columns=["line", "datetime", "dim", "m_dist", "anom"] + bus_names_all + hw_names_all
        )

    return headways_df, windows_df, ap_order_dict


def build_series_anoms(series_df, windows_df, now):
    new_series, anomalies_dfs = [], []
    dims = windows_df.dim.unique().tolist()
    for dim in dims:
        dim_df = windows_df.loc[windows_df.dim == dim]

        bus_names = ["bus" + str(i) for i in range(1, dim + 2)]
        unique_groups = []
        for i in range(dim_df.shape[0]):
            group = [dim_df.iloc[i][bus_names[k]] for k in range(dim + 1)]
            unique_groups.append(group)

        for group in unique_groups:
            # Build indexing condition
            conds1 = [dim_df[bus_names[k]] == group[k] for k in range(dim + 1)]
            conds2 = [series_df[bus_names[k]] == group[k] for k in range(dim + 1)]
            final_cond1, final_cond2 = True, True

            for i in range(len(conds1)):
                final_cond1 &= conds1[i]
                final_cond2 &= conds2[i]

            # Current group status
            group_now = dim_df.loc[final_cond1].iloc[0]

            # Past group status
            group_df = series_df.loc[final_cond2]
            if group_df.shape[0] > 0:
                group_df = group_df.sort_values("datetime", ascending=False)
                last_size = group_df.iloc[0].anom_size
                last_time = group_df.iloc[0].datetime
                seconds_ellapsed = (now - last_time).total_seconds()

                if ((group_now["anom"] == 0) and (last_size > 0)) | (
                    (group_now["anom"] == 1) and (seconds_ellapsed > 120)
                ):
                    group_now["anom_size"] = 0

                    # We append the finished anomalies to the anomalies dfs
                    group_df["anom_size"] = last_size
                    anomalies_dfs.append(group_df.loc[group_df.anom == 1])
                    group_df["anom"] = 0
                else:
                    group_now["anom_size"] = last_size + group_now["anom"]
            else:
                group_now["anom_size"] = group_now["anom"]

            new_series.append(group_now)

    # Build current series dataframe and append it to the past series
    if len(new_series) > 0:
        new_series_df = pd.DataFrame(new_series)[
            ["line", "datetime", "dim", "m_dist", "anom", "anom_size"]
            + bus_names_all
            + hw_names_all
        ]
    else:
        new_series_df = pd.DataFrame(
            columns=["line", "datetime", "dim", "m_dist", "anom", "anom_size"]
            + bus_names_all
            + hw_names_all
        )

    series_df = pd.concat([series_df, new_series_df], ignore_index=True)

    return series_df, anomalies_dfs


def clean_series(series_df, anomalies_dfs, now):
    unique_groups = []
    unique_groups_df = series_df.drop_duplicates(bus_names_all)
    for i in range(unique_groups_df.shape[0]):
        group = [unique_groups_df.iloc[i][bus_names_all[k]] for k in range(8 + 1)]
        unique_groups.append(group)

    for group in unique_groups:
        # Build indexing conditions
        conds = [series_df[bus_names_all[k]] == group[k] for k in range(8 + 1)]
        final_cond = True
        for cond in conds:
            final_cond &= cond
        group_df = series_df.loc[final_cond]

        if group_df.shape[0] > 0:
            group_df = group_df.sort_values("datetime", ascending=False)
            last_time = group_df.iloc[0].datetime
            last_size = group_df.iloc[0].anom_size

            seconds_ellapsed = (now - last_time).total_seconds()

            if seconds_ellapsed > 180:
                # Delete series from last series df
                series_df = series_df.loc[~final_cond]

                # If it was an anomaly
                if last_size > 0:
                    group_df["anom_size"] = last_size
                    anomalies_dfs.append(group_df.loc[group_df.anom == 1])

    return series_df, anomalies_dfs


def detect_anomalies(burst_df, last_burst_df, series_df, ap_order_dict):
    # Read dict
    while True:
        try:
            with open("../../Data/Anomalies/hyperparams.json") as f:
                hyperparams = json.load(f)
            break
        except:  # noqa: E722
            continue

    # Check if the burst dataframe has changed
    if last_burst_df.equals(burst_df):
        # If it hasnt changed we return None
        return None

    # Detect day type and hour range from current datetime
    now = burst_df.datetime.max()

    # Day type
    if (now.weekday() >= 0) and (now.weekday() <= 4):
        day_type = "LA"
    elif now.weekday() == 5:
        day_type = "SA"
    else:
        day_type = "FE"
    # Hour range
    for h_range in hour_ranges:
        if (now.hour >= h_range[0]) and (now.hour < h_range[1]):
            hour_range = str(h_range[0]) + "-" + str(h_range[1])
            break
        elif h_range == hour_ranges[-1]:
            print(f"\nHour range for {now.hour}:{now.minute} not defined. Waiting till 7am.\n")
            return "Waiting"

    # Process headways and build dimensional dataframes
    headways_df, windows_df, ap_order_dict = process_hws_ndim_mh_dist(
        lines, day_type, hour_range, burst_df, ap_order_dict, now
    )

    # Check for anomalies and update the time series
    series_df, anomalies_dfs = build_series_anoms(series_df, windows_df, now)

    # Delete series from the list that havent appeared in the last 5 minutes
    series_df, anomalies_dfs = clean_series(series_df, anomalies_dfs, now)

    # Series df
    series_df = series_df.reset_index(drop=True)

    # Build anomalies dataframe
    anomalies_df = (
        pd.concat(anomalies_dfs).drop("anom", axis=1)
        if len(anomalies_dfs) > 0
        else pd.DataFrame(
            columns=["line", "datetime", "dim", "m_dist", "anom_size"]
            + bus_names_all
            + hw_names_all
        )
    )

    # Get anomalies with size over threshold
    lines_anomalies = []
    for line in lines:
        line_anomalies = anomalies_df[anomalies_df.line == line]
        if line_anomalies.shape[0] > 0:
            size_th = hyperparams[line]["size_th"]
            line_anomalies = line_anomalies[line_anomalies.anom_size >= size_th]
            lines_anomalies.append(line_anomalies)
    try:
        anomalies_df = pd.concat(lines_anomalies)
    except:  # noqa: E722
        anomalies_df = pd.DataFrame()

    return headways_df, series_df, anomalies_df, ap_order_dict


# MAIN
def main():
    # Initialize dataframes
    series_df = pd.DataFrame(
        columns=["line", "datetime", "dim", "m_dist", "anom", "anom_size"]
        + bus_names_all
        + hw_names_all
    )

    # Create dict
    ap_order_dict = {}
    for line in lines:
        ap_order_dict[line] = {
            "dir1": [],
            "dir2": [],
            "ttls1": {},
            "ttls2": {},
            "cons_disap1": {},
            "cons_disap2": {},
            "cons_ap1": {},
            "cons_ap2": {},
        }

    last_burst_df = pd.DataFrame(
        columns=["line", "destination", "stop", "bus", "datetime", "estimateArrive", "DistanceBus"]
    )

    # Inform of the selected parameters
    print("\n[MADRID] ----- Detecting anomalies and preprocessing server data -----\n")

    # Look for updated data every 5 seconds
    while True:
        try:
            # Read last burst of data
            burst_df = pd.read_csv(
                "../../Data/RealTime/buses_data_burst.csv",
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
            burst_df["datetime"] = pd.to_datetime(
                burst_df["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
            )

            # Clean burst df
            burst_df = clean_data(burst_df)
        except:  # noqa: E722
            print("\n[MADRID] ----- Last burst of data not found, waiting... -----\n")
            time.sleep(10)
            continue

        # try :
        result = detect_anomalies(burst_df, last_burst_df, series_df, ap_order_dict)
        # except :
        # result = None

        # If the data was updated write files
        if result:
            if len(result) == 4:
                headways_df, series_df, anomalies_df, ap_order_dict = result

                # Write new data to files
                f = "../../Data/"
                burst_df.to_csv(
                    f + "RealTime/buses_data_burst_cleaned.csv", header=True, index=False
                )

                try:
                    # Read week df
                    week_df = pd.read_csv(
                        f + "RealTime/buses_data_week_cleaned.csv",
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
                    )[["line", "destination", "stop", "bus", "datetime", "estimateArrive"]]

                    # Parse the dates
                    week_df["datetime"] = pd.to_datetime(
                        week_df["datetime"], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f"
                    )

                    # Leave only data belonging to last week
                    week_df = week_df[week_df.datetime > (dt.now() - timedelta(days=7))]

                    # Concat with data of last burst
                    week_df = pd.concat(
                        [
                            week_df,
                            burst_df[
                                ["line", "destination", "stop", "bus", "datetime", "estimateArrive"]
                            ],
                        ]
                    )

                    # Sort values by datetime
                    week_df = week_df.sort_values(by="datetime", ascending=True)
                except:  # noqa: E722
                    week_df = burst_df

                week_df.to_csv(f + "RealTime/buses_data_week_cleaned.csv", header=True, index=False)

                headways_df.to_csv(f + "RealTime/headways_burst.csv", header=True, index=False)

                series_df.to_csv(f + "RealTime/series.csv", header=True, index=False)

                if anomalies_df.shape[0] > 0:
                    if os.path.isfile(f + "Anomalies/anomalies.csv"):
                        anomalies_df.to_csv(
                            f + "Anomalies/anomalies.csv", mode="a", header=False, index=False
                        )
                    else:
                        anomalies_df.to_csv(
                            f + "Anomalies/anomalies.csv", mode="a", header=True, index=False
                        )

                print(
                    f"\n[MADRID] ----- Burst headways and series were processed. Anomalies detected were added - {dt.now()} -----\n"
                )

            elif result == "Wait":
                time.sleep(120)

        last_burst_df = burst_df
        time.sleep(5)


if __name__ == "__main__":
    main()
