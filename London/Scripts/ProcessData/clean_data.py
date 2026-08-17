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
    # Change column names to the unes used in EMT Algorithms
    df = df.rename(
        columns={
            "vehicleId": "bus",
            "naptanId": "stop",
            "lineId": "line",
            "timeToStation": "estimateArrive",
            "timestamp": "datetime",
        }
    )

    # Remove data from more than 2 months ago
    df = df[df.datetime > df.datetime.max() - timedelta(days=60)]

    # Drop duplicate rows
    df = df.drop_duplicates()

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

        # estimateArrive positive values and time remaining lower than 2 hours
        eta_cond = (row.estimateArrive >= 0) and (row.estimateArrive < 7200)

        return eta_cond

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
        dtype={
            "id": "int32",
            "operationType": "uint16",
            "vehicleId": "str",
            "naptanId": "str",
            "stationName": "str",
            "lineId": "uint16",
            "lineName": "str",
            "platformName": "str",
            "direction": "uint16",
            "bearing": "uint16",
            "destinationNaptanId": "str",
            "destinationName": "str",
            "timeToStation": "int16",
            "currentLocation": "str",
            "towards": "str",
            "modeName": "str",
        },
    )[
        [
            "id",
            "vehicleId",
            "naptanId",
            "lineId",
            "direction",
            "bearing",
            "timestamp",
            "timeToStation",
            "expectedArrival",
        ]
    ]

    # Parse the dates
    buses_data["timestamp"] = pd.to_datetime(
        buses_data["timestamp"], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f"
    )
    buses_data.timestamp = buses_data.timestamp.dt.tz_localize(None)
    buses_data["expectedArrival"] = pd.to_datetime(
        buses_data["expectedArrival"], errors="coerce", format="%Y-%m-%dT%H:%M:%S"
    )
    buses_data.expectedArrival = buses_data.expectedArrival.dt.tz_localize(None)
    print(buses_data.info())
    lapsed_seconds = round((datetime.datetime.now() - now).total_seconds(), 3)
    print(f"\nFinished in {lapsed_seconds} seconds")
    print("-------------------------------------------------------------------\n\n")

    # Clean and Reformat the data
    now = datetime.datetime.now()
    print("-------------------------------------------------------------------")
    print(f"Cleaning the data... - {now}\n")
    buses_data_cleaned = clean_data(buses_data)
    print(buses_data_cleaned.info())
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
