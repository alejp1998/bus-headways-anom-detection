import asyncio
import datetime
import json
import os.path
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Timer

import pandas as pd
import requests

# API CREDENTIALS
from api_credentials import app_key_1
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# WE LOAD THE STOPS AND LINES
with open("Data/Static/lines_dict.json") as f:
    lines_dict = json.load(f)


class RepeatedTimer:
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


# Check if time is inside range
def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end


# API FUNCTIONS
def requests_retry_session(
    retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None
):
    """
    Function to ensure we get a good response for the request
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_arrival_times(stopId):
    """
    Returns the arrival data of buses for the desired stop and line
        Parameters
        ----------
        stopId : string
            The stop code
    """

    try:
        # We make the request for the buses arriving the stop
        response = requests_retry_session().get(
            f"https://api.tfl.gov.uk/StopPoint/{stopId}/arrivals",
            headers={
                "app_key": app_key_1,
                "Content-Type": "application/json",  # We specify that we are doing an application with a json object
            },
            timeout=20,
        )

        # Return the response if we received it ok
        return response
    except Exception as e:
        print(f"Error in the request to the stop : {stopId}")
        print("There was an error in the request \n")
        print(e)
        print("\n")
        return "Error"


def get_arrival_data(requested_lines):
    """
    Returns the data of all the buses inside the requested lines
        Parameters
        ----------
        requested_lines : list
            List with the desired line ids
    """

    # We get the list of stops to ask for
    stops_of_lines = []
    for line_id in requested_lines:
        stops_of_lines += lines_dict[line_id]["1"]["stops"] + lines_dict[line_id]["2"]["stops"]

    # List of different stops
    stops_of_lines = list(set(stops_of_lines))

    # The keys for the dataframe that is going to be built
    keys = [
        "id",
        "operationType",
        "vehicleId",
        "naptanId",
        "stationName",
        "lineId",
        "lineName",
        "platformName",
        "direction",
        "bearing",
        "destinationNaptanId",
        "destinationName",
        "timestamp",
        "timeToStation",
        "currentLocation",
        "towards",
        "expectedArrival",
        "timeToLive",
        "modeName",
    ]

    # Function to perform the requests asynchronously, performing them concurrently would be too slow
    async def get_data_asynchronous():
        row_list = []

        # Información de la recogida de datos
        n_ok_answers = 0
        n_not_ok_answers = 0

        # We set the number of workers that is going to take care about the requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            # We create a loop object
            loop = asyncio.get_event_loop()
            # And a list of tasks to be performed by the loop
            tasks = [
                loop.run_in_executor(
                    executor,
                    get_arrival_times,  # Function that is gonna be called by the tasks
                    stopId,  # Parameter for the function
                )
                for stopId in stops_of_lines
            ]

            # And finally we perform the tasks and gather the information returned by them
            for response in await asyncio.gather(*tasks):
                if not response == "Error":
                    if response.status_code == 200:
                        arrival_data = response.json()
                        n_ok_answers = n_ok_answers + 1
                    else:
                        # If the response isnt okey we pass to the next iteration
                        n_not_ok_answers = n_not_ok_answers + 1
                        continue
                else:
                    # If the response isnt okey we pass to the next iteration
                    n_not_ok_answers = n_not_ok_answers + 1
                    continue

                # We get the buses data
                buses_data = arrival_data
                for bus in buses_data:
                    # Get the line rows for each direction if it belongs to the requested lines
                    line_id = bus["lineId"]
                    if line_id in requested_lines:
                        bus["direction"] = 1 if bus["direction"] == "outbound" else 2
                        values = [bus[key] for key in keys]
                        row_list.append(dict(zip(keys, values)))

        return row_list, n_ok_answers, n_not_ok_answers

    # We declare the loop and call it, then we run it until it is complete
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(get_data_asynchronous())
    loop.run_until_complete(future)

    # And once it is completed we gather the information returned by it like this
    future_result = future.result()
    if future_result is None:
        return None
    else:
        row_list = future_result[0]
        n_ok_answers = future_result[1]
        n_not_ok_answers = future_result[2]

        # We create the dataframe of the buses
        buses_df = pd.DataFrame(row_list, columns=keys)

        # And we append the data to the csv
        f = "Data/Raw/buses_data.csv"
        f_burst = "Data/RealTime/buses_data_burst.csv"
        if os.path.isfile(f):
            buses_df.to_csv(f, mode="a", header=False, index=False)
        else:
            buses_df.to_csv(f, mode="a", header=True, index=False)

        # Write to burst file
        buses_df.to_csv(f_burst, header=True, index=False)

        print(
            f"New burst - There were {n_ok_answers} ok responses and {n_not_ok_answers} not okey responses - {datetime.datetime.now()}"
        )
        print(f"{buses_df.shape[0]} new rows appended to {f}\n")


def main():
    rt_started = False

    # Normal buses hours range (1 hora menos al ser Londres)
    start_time_day = datetime.time(5, 0, 0)
    end_time_day = datetime.time(22, 0, 0)

    while True:
        # Retrieve data every interval seconds if we are between 6:00 and 23:00
        now = datetime.datetime.now()

        if time_in_range(start_time_day, end_time_day, now.time()):
            if not rt_started:
                print(
                    f"Retrieve data from lines 18 and 25 - 129 Stops - {datetime.datetime.now()}\n"
                )
                requested_lines = ["18", "25"]
                rt = RepeatedTimer(
                    30, get_arrival_data, requested_lines
                )  # Retrieve every 30 seconds
                rt_started = True
        else:
            # Stop timer if it exists
            if rt_started:
                print(
                    f"Stop retrieving data from lines 1,44,82,132,133 - 129 Stops - {datetime.datetime.now()}\n"
                )
                rt.stop()
                rt_started = False

        # Wait 10 seconds till next loop (no need to run the loop faster)
        time.sleep(10)


if __name__ == "__main__":
    main()
