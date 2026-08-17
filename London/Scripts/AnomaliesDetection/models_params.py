import json
import multiprocessing
from datetime import datetime as dt

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

num_cores = multiprocessing.cpu_count()

# Day types
day_type_dict = {  # 0 = Monday, 1 = Tuesday ...
    "LA": [0, 1, 2, 3, 4],  # LABORABLES
    "LJ": [0, 1, 2, 3],  # LUNES A JUEVES
    "VV": [4],  # VIERNES
    "SA": [5],  # SABADOS
    "FE": [6],  # DOMIGOS O FESTIVOS
}


# FUNCTIONS
def get_ndim_hws(df, dim):
    # Generate names for the columns of the dataframe to be built
    hw_names = ["hw" + str(i) + str(i + 1) for i in range(1, dim + 1)]
    bus_names = ["bus" + str(i) for i in range(1, dim + 2)]

    # Columns to build dictionary
    columns = {}
    names = ["datetime"] + bus_names + hw_names
    for name in names:
        columns[name] = []

    # Unique datetime identifiers for the bursts
    burst_times = df.datetime.unique()
    for burst_time in burst_times:
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


def get_model_params(df, dim):
    _cov_matrixes, _means = [], []
    # Locate only data columns
    dim_df = df.iloc[:, -dim:]
    # Check if dataframes dimensions match or if they are empty
    if dim_df.shape[1] != dim:
        return f"DataFrame index {dim - 1} has {dim_df.shape[1]} dimensions, {dim} expected."
    if dim_df.shape[0] < 1:
        return f"DataFrame index {dim - 1} is empty"

    # Get columns from df
    cols = [dim_df.iloc[:, k] for k in range(dim)]

    # Get the covariance matrix of the data
    if dim > 1:
        cov_matrix = np.cov(np.stack(cols, axis=0))
    else:
        cov_matrix = cols[0].std()

    # Get mean of the data points
    if dim > 1:
        mean = [col.mean() for col in cols]
    else:
        mean = cols[0].mean()

    return cov_matrix, mean


def calc_line_params(df, line, day_types, hour_ranges, min_points):
    models_params_dict = {}
    models_params_dict[line] = {}

    # Max dimensional model to train
    max_dim = 3
    for day_type in day_types:
        models_params_dict[line][day_type] = {}
        for hour_range in hour_ranges:
            models_params_dict[line][day_type][str(hour_range[0]) + "-" + str(hour_range[1])] = {}

            # Get data split
            split_df = df.loc[
                (df.line == line)
                & (df.datetime.dt.weekday.isin(day_type_dict[day_type]))
                & (df.datetime.dt.hour >= hour_range[0])
                & (df.datetime.dt.hour < hour_range[1])
            ]

            # Mean and standard deviation of the hws split
            std_val = split_df.headway.std()
            mean_val = split_df.headway.mean()

            # Remove outliers
            conf = 0.95
            ci = stats.norm.interval(conf, loc=mean_val, scale=std_val)
            split_df = split_df[(split_df.headway > ci[0]) & (split_df.headway < ci[1])]

            # Generate ndimensional windows dataframes
            window_data_points = min_points + 1
            dim = 1
            windows_dfs = []
            while (window_data_points > min_points) & (dim <= max_dim):
                window_df = get_ndim_hws(split_df, dim)
                window_data_points = window_df.shape[0]
                dim += 1

                if window_data_points > min_points:
                    windows_dfs.append(window_df)

            # Max dimension for which we are going to train a model
            max_dim = len(windows_dfs)
            models_params_dict[line][day_type][str(hour_range[0]) + "-" + str(hour_range[1])][
                "max_dim"
            ] = max_dim

            # For every window data
            for dim in range(1, max_dim + 1):
                models_params_dict[line][day_type][str(hour_range[0]) + "-" + str(hour_range[1])][
                    dim
                ] = {}

                # Calculate cov_matrix and mean and add it to the dictionary
                cov_matrix, mean = get_model_params(windows_dfs[dim - 1], dim)

                models_params_dict[line][day_type][str(hour_range[0]) + "-" + str(hour_range[1])][
                    dim
                ]["cov_matrix"] = cov_matrix.tolist()
                models_params_dict[line][day_type][str(hour_range[0]) + "-" + str(hour_range[1])][
                    dim
                ]["mean"] = mean

            print(
                "\nModel params for line {}-(Day type : {}, Hour range: {}) with max_dim = {} processed.\n".format(
                    line, day_type, str(hour_range[0]) + "-" + str(hour_range[1]), max_dim
                )
            )

    return models_params_dict


def train_models(df, min_points):
    models_params_dict = {}

    # Locate headways of the last three weeks
    # now = dt.now()
    # df = df.loc[df.datetime > (now - timedelta(days=21))]

    # Lines to iterate over
    lines = [18, 25]
    # Day types to iterate over
    day_types = ["LA", "SA", "FE"]
    # Hour ranges to iterate over
    hour_ranges = [
        [5, 7],
        [7, 9],
        [9, 11],
        [11, 13],
        [13, 15],
        [15, 17],
        [17, 19],
        [19, 21],
        [21, 23],
    ]
    # hour_ranges = [[i,i+1] for i in range(7,23)]

    # Train a model for each interval in parallel
    models_params_dict = {}
    line_dicts = Parallel(n_jobs=num_cores, max_nbytes=None)(
        delayed(calc_line_params)(df, line, day_types, hour_ranges, min_points) for line in lines
    )
    for line_dict in line_dicts:
        models_params_dict.update(line_dict)

    return models_params_dict


# MAIN
def main():
    # WE LOAD THE ARRIVAL TIMES DATA
    now = dt.now()
    print("\n-------------------------------------------------------------------")
    print(f"Reading the headways data... - {now}\n")
    # Headways data
    hws = pd.read_csv(
        "../../Data/Processed/headways.csv",
        dtype={
            "line": "uint16",
            "direction": "uint16",
            "busA": "str",
            "busB": "str",
            "hw_pos": "uint16",
            "headway": "int16",
            "busB_ttls": "uint16",
        },
    )[["line", "direction", "datetime", "hw_pos", "busA", "busB", "headway", "busB_ttls"]]

    # Parse the dates
    hws["datetime"] = pd.to_datetime(hws["datetime"], format="%Y-%m-%d %H:%M:%S.%f")

    # Eliminate 0 pos from headways
    hws = hws.loc[(hws.hw_pos != 0)]

    print(hws.info())
    lapsed_seconds = round((dt.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Preprocess data; adds day_trip, arrival_time and calculated coordinates attributes
    now = dt.now()
    print("-------------------------------------------------------------------")
    print(f"Processing models parameters... - {now}\n")

    models_params_dict = train_models(hws, 100)

    lapsed_seconds = round((dt.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Estimate models parameters
    f = "../../Data/Anomalies/models_params.json"
    now = dt.now()
    print("-------------------------------------------------------------------")
    print(f"Writting models parameters to {f}... - {now}")

    # Write result to file
    with open(f, "w") as fp:
        json.dump(models_params_dict, fp)

    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    print("New data is ready!\n")


if __name__ == "__main__":
    main()
