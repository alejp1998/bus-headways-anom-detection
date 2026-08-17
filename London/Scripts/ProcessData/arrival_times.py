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
def process_bus_df(bus, df_stop, threshold):
    """
    Processes the rows adding day_trip, arrival time and calculated coordinates attributes
    """
    df_bus = df_stop.loc[df_stop.bus == bus].sort_values(by="datetime").reset_index(drop=True)
    last_index = 0
    last_time = df_bus.iloc[0].datetime
    day_trip = 0
    trips = []

    for i in range(df_bus.shape[0]):
        # Estimate arrival time for each slice
        if ((df_bus.iloc[i].datetime - last_time).total_seconds() > 600) | (
            i == df_bus.shape[0] - 1
        ):
            # Trip dataframe
            df_trip = df_bus.iloc[last_index:i]

            if df_trip.shape[0] != 0:
                # Trip number inside the day
                day_trip += 1

                # Get first row with estimateArrive < threshold seconds
                df_close = df_trip.loc[df_trip.estimateArrive < threshold]
                if df_close.shape[0] != 0:
                    row = df_close.sort_values(by="datetime", ascending="True").iloc[0]
                else:
                    row = df_trip.loc[df_trip.estimateArrive == df_trip.estimateArrive.min()].iloc[
                        0
                    ]

                # Assign arrival time and day trip number
                df_trip = df_trip.assign(
                    day_trip=day_trip,
                    arrival_time=row.datetime + timedelta(seconds=int(row.estimateArrive)),
                )
                trips.append(df_trip)
                last_index = i
        # Update last time value
        last_time = df_bus.iloc[i].datetime
    return trips


# MAIN
def get_arrival_times(df, threshold):
    """
    Returns the dataframe with a new column with the estimation of the time when the bus has arrived the stop
    that is giving the estimation, and another column with the trip index in the day. The estimation
    is based on the value of ''estimateArrive'' for the first row that is less than threshold
    seconds away from the stop.

    May take a long time for big dataframes

    Parameters
    ----------------------
    df : The dataframe where we wish to add the column
    threshold : Seconds threshold to estimate arrival time
    """

    # Get number of days lapsed
    days = (df.datetime.max() - df.datetime.min()).days + 1
    first_date = df.datetime.min()

    # List to add the trip dataframes
    trips = []
    for day in range(days):
        df_day = df.loc[df.datetime.dt.date == (first_date + timedelta(days=day))]
        # For each line of the bus
        lines = df_day.line.unique().tolist()
        for line in lines:
            df_line = df_day.loc[df_day.line == line]
            # For each destination in that line
            for direction in [1, 2]:
                df_dir = df_line.loc[df.direction == direction]
                # For each stop in that line and destination
                for stop in lines_dict[str(line)][str(direction)]["stops"]:
                    df_stop = df_dir.loc[df_dir.stop == stop]
                    # For each bus in that line destination and stop
                    buses = df_stop.bus.unique().tolist()
                    trips += sum(
                        (
                            Parallel(n_jobs=num_cores)(
                                delayed(process_bus_df)(bus, df_stop, threshold) for bus in buses
                            )
                        ),
                        [],
                    )
    new_df = pd.concat(trips).sort_values(by="datetime", ascending="True")

    # Remove duplicate bus arrival times
    # new_df = new_df.drop_duplicates(subset=['id', 'arrival_time'], keep='last')

    # Remove non-useful columns
    new_df = new_df[
        ["line", "direction", "stop", "bus", "datetime", "estimateArrive", "arrival_time"]
    ]

    return new_df


# MAIN
def main():
    path = "../../Data/"
    f_orig = "Processed/buses_data_cleaned.csv"
    f_result = "Processed/arrival_times.csv"

    # WE LOAD THE CLEANED DATA
    now = datetime.datetime.now()
    print("\n-------------------------------------------------------------------")
    print(f"Reading the cleaned data... - {now}\n")
    buses_data_cleaned = pd.read_csv(
        path + f_orig,
        dtype={
            "id": "int32",
            "bus": "str",
            "stop": "str",
            "line": "uint16",
            "direction": "uint16",
            "bearing": "uint16",
            "estimateArrive": "int16",
        },
    )[
        [
            "id",
            "bus",
            "stop",
            "line",
            "direction",
            "bearing",
            "datetime",
            "estimateArrive",
            "expectedArrival",
        ]
    ]

    # Parse the dates
    buses_data_cleaned["datetime"] = pd.to_datetime(
        buses_data_cleaned["datetime"], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f"
    )
    buses_data_cleaned.datetime = buses_data_cleaned.datetime.dt.tz_localize(None)
    buses_data_cleaned["expectedArrival"] = pd.to_datetime(
        buses_data_cleaned["expectedArrival"], errors="coerce", format="%Y-%m-%dT%H:%M:%S"
    )
    buses_data_cleaned.expectedArrival = buses_data_cleaned.expectedArrival.dt.tz_localize(None)
    print(buses_data_cleaned.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Estimate Arrival Times
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Estimating the arrival times... - {now}\n")
    arrival_times = get_arrival_times(buses_data_cleaned, 25)
    print(arrival_times.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Arrival Times data info
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Writting new data to {path + f_result}... - {now}")
    # Write result to file
    arrival_times.to_csv(path + f_result, header=True, index=False)
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    print("New data is ready!\n")


if __name__ == "__main__":
    main()
