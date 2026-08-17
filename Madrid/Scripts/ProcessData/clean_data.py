import datetime
import json
from datetime import timedelta

import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize()

# Load line_stops_dict
with open("../../Data/Static/lines_dict.json") as f:
    lines_dict = json.load(f)


# FUNCTIONS
def clean_data(df):
    """
    Returns the dataframe without the rows that dont match the conditions specified

    Parameters
    ----------------
    df : DataFrame
        Dataframe to clean
    """

    def check_conditions(row):
        """
        Checks if every row in the dataframe matchs the conditions

        Parameters
        ----------------
        row : DataFrame
            Dataframe to clean
        processed : bool
            Boolean that indicates if the data has been processed
        """
        # Direction
        direction = "1" if row.destination == lines_dict[row.line]["destinations"][1] else "2"

        # Line destination stop coherence condition
        line_dest_stop_cond = False
        if row.line in lines_dict.keys():
            if row.destination in lines_dict[row.line]["destinations"]:
                if str(row.stop) in lines_dict[row.line][direction]["stops"]:
                    line_dest_stop_cond = True

        # DistanceBus values lower than the line length or negative
        dist_cond = (row.DistanceBus >= 0) and (
            row.DistanceBus < int(lines_dict[row.line][direction]["length"])
        )

        # estimateArrive values lower than the time it takes to go through the line at an speed
        # of 2m/s, instantaneous speed lower than 120 km/h and positive values and time remaining lower than 2 hours
        eta_cond = (
            (row.estimateArrive > 0)
            and (row.estimateArrive < (int(lines_dict[row.line][direction]["length"]) / 2))
            and ((3.6 * row.DistanceBus / row.estimateArrive) < 120) & (row.estimateArrive < 7200)
        )

        return line_dest_stop_cond and dist_cond and eta_cond

    # Remove data from more than 2 months ago
    df = df[df.datetime > df.datetime.max() - timedelta(days=60)]

    # Check conditions in df
    mask = df.parallel_apply(check_conditions, axis=1)
    # Select rows that match the conditions
    df = df.loc[mask].reset_index(drop=True)
    # Return cleaned DataFrame
    return df


# MAIN
def main():
    path = "../../Data/"
    f_orig = "Raw/buses_data.csv"
    f_result = "Processed/buses_data_cleaned.csv"

    # WE LOAD THE ARRIVAL TIMES DATA
    now = datetime.datetime.now()
    print("\n-------------------------------------------------------------------")
    print(f"Reading the original data... - {now}\n")
    buses_data = pd.read_csv(
        path + f_orig,
        error_bad_lines=False,
        dtype={
            "line": "str",
            "destination": "str",
            "stop": "uint16",
            "bus": "uint16",
            "given_coords": "int16",
            "pos_in_burst": "uint16",
            "estimateArrive": "int32",
            "DistanceBus": "int32",
            "request_time": "int32",
            "lat": "float32",
            "lon": "float32",
        },
    )[
        [
            "line",
            "destination",
            "stop",
            "bus",
            "datetime",
            "estimateArrive",
            "DistanceBus",
            "given_coords",
            "lat",
            "lon",
        ]
    ]

    # Parse the dates
    buses_data["datetime"] = pd.to_datetime(buses_data["datetime"], format="%Y-%m-%d %H:%M:%S.%f")
    print(buses_data.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Get data recorded from September 2020
    buses_data = buses_data[
        ((buses_data.datetime.dt.year == 2020) & (buses_data.datetime.dt.month >= 9))
        | (buses_data.datetime.dt.year > 2020)
    ]

    # Clean the data
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Cleaning the data... - {now}\n")
    buses_data_cleaned = clean_data(buses_data)
    print(buses_data.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Final data info
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Writting new data to {path + f_result}... - {now}")
    # Write result to file
    buses_data_cleaned.to_csv(path + f_result, header=True, index=False)
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    print("New data is ready!\n")


if __name__ == "__main__":
    main()
