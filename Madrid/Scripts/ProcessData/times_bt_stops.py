import datetime
import json
import multiprocessing
from datetime import timedelta

import pandas as pd
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()


# Load line_stops_dict
with open("../../Data/Static/lines_dict.json") as f:
    lines_dict = json.load(f)


# FUNCTIONS
def process_hour_df(line_df, hour):
    """
    Returns the dataframe with the times between stops for the line and hour selected.

    Parameters
    -----------------
        line_df: Dataframe
        hour : int
    """
    # Hour df
    hour_df = line_df.loc[line_df.datetime.dt.hour == hour]
    if hour_df.shape[0] == 0:
        return pd.DataFrame([])
    # Line id
    line = hour_df.iloc[0].line

    start_date = hour_df.datetime.min()
    end_date = hour_df.datetime.max()
    date = start_date

    # Rows list to build
    rows_list = []

    # Iterate over dates
    while date < end_date:
        date_hour_df = hour_df.loc[
            (hour_df.datetime.dt.year == date.year)
            & (hour_df.datetime.dt.month == date.month)
            & (hour_df.datetime.dt.day == date.day)
        ]
        # Update date value for next iteration
        date = date + timedelta(days=1)

        if date_hour_df.shape[0] > 0:
            # Iterate over destinations
            dest2, dest1 = lines_dict[line]["destinations"]
            for destination in [dest1, dest2]:
                # Direction and destination values
                direction = "1" if destination == dest1 else "2"
                final_dest = dest2 if direction == "1" else dest1

                # Destination stops
                stops_dir = lines_dict[line][direction]["stops"]

                # Destination and final stop dfs
                dest_df = date_hour_df.loc[date_hour_df.destination == destination]
                final_stop_df = date_hour_df.loc[
                    (date_hour_df.stop == int(stops_dir[-1]))
                    & (date_hour_df.destination == final_dest)
                ]

                # Iterate over stops
                for i in range(len(stops_dir) - 1):
                    # Stops to analyse
                    stop1 = int(stops_dir[i])
                    stop2 = int(stops_dir[i + 1])

                    # Data for these 2 stops
                    if i == len(stops_dir) - 2:
                        stops_df_all = dest_df.loc[dest_df.stop == stop1]
                        stops_df_all = pd.concat([stops_df_all, final_stop_df])
                    else:
                        stops_df_all = dest_df.loc[dest_df.stop.isin([stop1, stop2])]

                    # Data ordered and without duplicates
                    stops_df = stops_df_all.drop_duplicates(
                        subset=["bus", "stop", "arrival_time"], keep="first"
                    )
                    stops_df = stops_df.sort_values(by=["bus", "arrival_time"], ascending=True)

                    n = 0
                    while n < (stops_df.shape[0] - 1):
                        first_stop = stops_df.iloc[n]
                        second_stop = stops_df.iloc[n + 1]
                        if (
                            (first_stop.stop == stop1)
                            and (second_stop.stop == stop2)
                            and (first_stop.bus == second_stop.bus)
                        ):
                            # We create the row dictionary
                            row = {}
                            row["date"] = first_stop.datetime.strftime("%Y-%m-%d")
                            row["line"] = line
                            row["direction"] = direction
                            row["st_hour"] = hour
                            row["end_hour"] = hour + 1
                            row["stopA"] = first_stop.stop
                            row["stopB"] = second_stop.stop
                            row["bus"] = first_stop.bus
                            # Time to next stop
                            time_between_stops = (
                                second_stop.arrival_time - first_stop.arrival_time
                            ).total_seconds()

                            # API time to next stop
                            api_estim = stops_df_all.loc[
                                (
                                    stops_df_all.datetime
                                    > (first_stop.arrival_time - timedelta(seconds=60))
                                )
                                & (stops_df_all.datetime < (first_stop.arrival_time))
                                & (stops_df_all.bus == second_stop.bus)
                            ]

                            if api_estim.shape[0] > 1:
                                estim_act = api_estim.loc[
                                    api_estim.stop == first_stop.stop
                                ].estimateArrive
                                estim_next = api_estim.loc[
                                    api_estim.stop == second_stop.stop
                                ].estimateArrive
                                if (
                                    (estim_act.shape[0] > 0) & (estim_next.shape[0] > 0)
                                    and (time_between_stops < 600)
                                    and (time_between_stops > 0)
                                ):
                                    row["trip_time"] = round(time_between_stops, 3)
                                    row["api_trip_time"] = estim_next.iloc[0] - estim_act.iloc[0]
                                    rows_list.append(row)
                            n += 2

                        else:
                            n += 1

    return pd.DataFrame(rows_list)


def get_time_between_stops(df):
    """
    Returns a dataframe with the times between stops for every day, line and hour range.

    Parameters
    -----------------
        df: Dataframe
            Data to process
    """
    # For every line collected
    lines = ["1", "44", "82", "F", "G", "U", "132", "133", "N2", "N6"]

    # Get new dictionaries and process the lines
    dfs_list = []
    for line in lines:
        hours = range(0, 6) if line in ["N2", "N6"] else range(7, 23)
        line_df = df.loc[df.line == line]
        dfs = Parallel(n_jobs=num_cores, max_nbytes=None)(
            delayed(process_hour_df)(line_df, hour) for hour in hours
        )
        dfs_list += dfs

    # Concatenate dataframes
    processed_df = (
        pd.concat(dfs_list)
        .sort_values(by=["line", "direction", "date", "st_hour"], ascending=True)
        .reset_index(drop=True)
    )
    return processed_df


# MAIN
def main():
    path = "../../Data/"
    f_orig = "Processed/arrival_times.csv"
    f_result = "Processed/times_bt_stops.csv"

    # WE LOAD THE ARRIVAL TIMES DATA
    now = datetime.datetime.now()
    print("\n-------------------------------------------------------------------")
    print(f"Reading the data cleaned with the arrival times included... - {now}\n")
    arrival_times = pd.read_csv(
        path + f_orig,
        dtype={
            "line": "str",
            "destination": "str",
            "stop": "uint16",
            "bus": "uint16",
            "estimateArrive": "int16",
            "DistanceBus": "int16",
        },
    )[
        [
            "line",
            "destination",
            "stop",
            "bus",
            "datetime",
            "DistanceBus",
            "estimateArrive",
            "arrival_time",
        ]
    ]

    # Parse the dates
    arrival_times["datetime"] = pd.to_datetime(
        arrival_times["datetime"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    arrival_times["arrival_time"] = pd.to_datetime(
        arrival_times["arrival_time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    print(arrival_times.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Estimate times bt stops
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Processing times between stops... - {now}\n")
    times_bt_stops = get_time_between_stops(arrival_times)
    print(times_bt_stops.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Final data info
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Writting new data to {path + f_result}... - {now}")
    # Write result to file
    times_bt_stops.to_csv(path + f_result, header=True, index=False)
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    print("New data is ready!\n")


if __name__ == "__main__":
    main()
