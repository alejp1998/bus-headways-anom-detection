import asyncio
import datetime
import json
import os.path
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Timer

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# WE LOAD THE STOPS AND LINES
line_shapes = pd.read_csv("Data/Static/line_shapes.csv")
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


def get_access_token(email, password):
    """
    Returns the access token of the EMT Madrid API

        Parameters
        ----------
        email : string
            The email of the account
        password : string
            Password of the account
    """
    try:
        if account_index == 0:
            # Special request for the account with API
            response = requests_retry_session().get(
                "https://openapi.emtmadrid.es/v2/mobilitylabs/user/login/",
                headers={"X-ClientId": XClientId, "passKey": passKey},
                timeout=5,
            )
        else:
            response = requests_retry_session().get(
                "https://openapi.emtmadrid.es/v2/mobilitylabs/user/login/",
                headers={"email": email, "password": password},
                timeout=5,
            )
        json_response = response.json()
        return json_response
    except Exception as e:
        print("There was an error in the request \n")
        print(e)
        print("\n")
        return "Error"


def get_arrival_times(stopId, accessToken):
    """
    Returns the arrival data of buses for the desired stop and line

        Parameters
        ----------
        stopId : string
            The stop code
        accessToken: string
            The accessToken obtained in the login
    """

    # We build the body for the request
    body = {
        "cultureInfo": "EN",
        "Text_StopRequired_YN": "N",
        "Text_EstimationsRequired_YN": "Y",
        "Text_IncidencesRequired_YN": "N",
        "DateTime_Referenced_Incidencies_YYYYMMDD": "20200130",
    }

    # And we perform the request
    try:
        response = requests_retry_session().post(
            "https://openapi.emtmadrid.es/v2/transport/busemtmad/stops/{}/arrives/{}/".format(
                stopId, ""
            ),
            data=json.dumps(body),
            headers={"accessToken": accessToken, "Content-Type": "application/json"},
            timeout=5,
        )
        # Return the response if we received it ok
        return response
    except Exception as e:
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
        line_id = str(line_id)
        if lines_dict[line_id] is not None:
            if lines_dict[line_id]["1"] is not None:
                stops_of_lines = stops_of_lines + lines_dict[line_id]["1"]["stops"]
            if lines_dict[line_id]["2"] is not None:
                stops_of_lines = stops_of_lines + lines_dict[line_id]["2"]["stops"]

    # List of different stops
    stops_of_lines = list(set(stops_of_lines))

    # The keys for the dataframe that is going to be built
    keys = [
        "bus",
        "line",
        "stop",
        "datetime",
        "isHead",
        "destination",
        "deviation",
        "estimateArrive",
        "DistanceBus",
        "request_time",
        "given_coords",
        "lat",
        "lon",
    ]

    # Function to perform the requests asynchronously, performing them concurrently would be too slow
    async def get_data_asynchronous():
        global account_index
        global accessToken
        global out_of_hits
        global day_burst

        # We increase the day burst by one
        day_burst = day_burst + 1

        row_list = []

        # Información de la recogida de datos
        n_ok_answers = 0
        n_not_ok_answers = 0

        # We set the number of workers that is going to take care about the requests
        with ThreadPoolExecutor(max_workers=50) as executor:
            # We create a loop object
            loop = asyncio.get_event_loop()
            # And a list of tasks to be performed by the loop
            tasks = [
                loop.run_in_executor(
                    executor,
                    get_arrival_times,  # Function that is gonna be called by the tasks
                    *(stopId, accessToken),  # Parameters for the function
                )
                for stopId in stops_of_lines
            ]
            # We randomize the order of the tasks
            random.shuffle(tasks)
            # And finally we perform the tasks and gather the information returned by them into two lists
            for response in await asyncio.gather(*tasks):
                if not response == "Error":
                    if response.status_code == 200:
                        arrival_data = response.json()
                        n_ok_answers = n_ok_answers + 1
                        if arrival_data["code"] == "98":
                            # If we spend all the hits we switch the account and wait for next request
                            out_of_hits = True
                            print(
                                f"Hits of account_index = {account_index} spent - {datetime.datetime.now()}\n"
                            )
                            # Return the data gathered if the request that fails isnt the first
                            if len(row_list) == 0:
                                return None
                            else:
                                print("Appending data gathered before end of hits\n")
                                return row_list, n_ok_answers, n_not_ok_answers
                    else:
                        # If the response isnt okey we pass to the next iteration
                        n_not_ok_answers = n_not_ok_answers + 1
                        continue
                else:
                    # If the response isnt okey we pass to the next iteration
                    n_not_ok_answers = n_not_ok_answers + 1
                    continue
                try:
                    lapsed_time = int(
                        re.search("lapsed: (.*) millsecs", arrival_data["description"]).group(1)
                    )
                except:  # noqa: E722
                    lapsed_time = 0
                date_time = datetime.datetime.strptime(
                    arrival_data["datetime"], "%Y-%m-%dT%H:%M:%S.%f"
                )

                # We get the buses data
                buses_data = arrival_data["data"][0]["Arrive"]
                for bus in buses_data:
                    # Get the line rows for each direction
                    line_id = line_shapes.loc[line_shapes.line_sn == bus["line"]].iloc[0].line_id
                    if line_id in requested_lines:
                        # Given coordinates provided by the API
                        bus["lon"] = bus["geometry"]["coordinates"][0]
                        bus["lat"] = bus["geometry"]["coordinates"][1]
                        if (bus["lat"] != 0) and (bus["lon"] != 0):
                            bus["given_coords"] = 1
                        else:
                            bus["given_coords"] = 0

                        bus["datetime"] = date_time
                        bus["request_time"] = lapsed_time
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
            buses_df.to_csv(f, mode="a", header=False)
        else:
            buses_df.to_csv(f, mode="a", header=True)

        # Append to burst file
        buses_df.to_csv(f_burst, header=True, index=False)

        print(
            f"Burst {day_burst} - There were {n_ok_answers} ok responses and {n_not_ok_answers} not okey responses - {datetime.datetime.now()}"
        )
        print(f"{buses_df.shape[0]} new rows appended to {f}\n")


# Global vars
from api_credentials import XClientId, emails, passKey, passwords

account_index = 0
accessToken = ""
out_of_hits = True
day_burst = 0


def main():
    global account_index
    global accessToken
    global out_of_hits
    global day_burst

    rt_started = False
    rt = None

    # Initial value of accessToken
    for i in range(0, 5):
        account_index = i

        # Try until we get response to the login without errors
        json_response = "Error"
        while json_response == "Error":
            json_response = get_access_token(emails[account_index], passwords[account_index])

        if json_response["code"] == "98":
            print(f"Account {account_index} has no hits available\n")
            if i == 4:
                print("None of the accounts has hits available\n")
                i = 0
            else:
                pass
        elif (json_response["code"] == "00") | (json_response["code"] == "01"):
            accessToken = json_response["data"][0]["accessToken"]
            print(f"Making requests from account {account_index} - accessToken : {accessToken}\n")
            out_of_hits = False
            break

    # Normal buses hours range
    start_time_day = datetime.time(7, 0, 0)
    end_time_day = datetime.time(23, 0, 0)
    # Night buses hours range
    start_time_night = datetime.time(0, 0, 0)
    end_time_night = datetime.time(5, 30, 0)

    while True:
        # Retrieve data every interval seconds if we are between 6:00 and 23:30
        now = datetime.datetime.now()

        # If we have lost all the hits
        if out_of_hits:
            # Wait until we have hits available again
            if (
                (now.weekday() in [0, 1, 2, 3, 4])
                & (time_in_range(start_time_day, end_time_day, now.time()))
            ) | (now.weekday() in [5, 6]):
                for i in range(0, 5):
                    account_index = i

                    # Try until we get response to the login without errors
                    json_response = "Error"
                    while json_response == "Error":
                        json_response = get_access_token(
                            emails[account_index], passwords[account_index]
                        )

                    if json_response["code"] == "98":
                        print(f"Account {account_index} has no hits available\n")
                        if i == 4:
                            print(
                                f"None of the accounts has hits available - Sleeping 10 minutes - {datetime.datetime.now()}\n"
                            )
                            i = 0
                            if rt_started:
                                print(f"Stop retrieving data - {datetime.datetime.now()}")
                                if rt is not None:
                                    rt.stop()
                                rt_started = False
                            time.sleep(600)
                        else:
                            pass
                    elif (json_response["code"] == "00") | (json_response["code"] == "01"):
                        accessToken = json_response["data"][0]["accessToken"]
                        print(
                            f"Making requests from account {account_index} - accessToken : {accessToken}\n"
                        )
                        out_of_hits = False
                        break
            else:
                # We wait 10 minutes
                time.sleep(600)
            # We pass directly to next iteration
            continue

        # If we are not in the weekend
        if now.weekday() in [0, 1, 2, 3, 4]:
            # Retrieve data from lines 1,44,82,132,133 - 272 Stops
            if time_in_range(start_time_day, end_time_day, now.time()):
                if not rt_started:
                    print(
                        f"Retrieve data from lines 1,44,82,132,133 - 272 Stops - {datetime.datetime.now()}\n"
                    )  # 91(F),92(G),99(U)
                    requested_lines = [1, 44, 82, 132, 133]
                    rt = RepeatedTimer(50, get_arrival_data, requested_lines)
                    rt_started = True
            else:
                # Stop timer if it exists
                if rt_started:
                    print(
                        f"Stop retrieving data from lines 1,44,82,132,133 - 272 Stops - {datetime.datetime.now()}\n"
                    )
                    rt.stop()
                    rt_started = False
                    day_burst = 0
        # If we are in Saturday or Sunday
        else:
            # Retrieve data from lines 1,44,82,132,133 - 272 Stops
            if time_in_range(start_time_day, end_time_day, now.time()):
                if not rt_started:
                    print(
                        f"Retrieve data from lines 1,44,82,132,133 - 272 Stops - {datetime.datetime.now()}\n"
                    )
                    requested_lines = [1, 44, 82, 132, 133]
                    rt = RepeatedTimer(60, get_arrival_data, requested_lines)
                    rt_started = True
            # Retrieve data from lines 502,506 - 131 Stops
            elif time_in_range(start_time_night, end_time_night, now.time()):
                if not rt_started:
                    print(
                        f"Retrieve data from lines 502,506 - 131 Stops - {datetime.datetime.now()}\n"
                    )
                    requested_lines = [502, 506]
                    rt = RepeatedTimer(70, get_arrival_data, requested_lines)
                    rt_started = True
            else:
                # Stop timer if it exists
                if rt_started:
                    print(f"Stop retrieving data for weekends - {datetime.datetime.now()}\n")
                    rt.stop()
                    rt_started = False
                    day_burst = 0

        # Wait 10 seconds till next loop (no need to run the loop faster)
        time.sleep(10)


if __name__ == "__main__":
    main()
