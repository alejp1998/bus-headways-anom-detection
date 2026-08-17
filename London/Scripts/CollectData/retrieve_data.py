#!/usr/bin/env python3
"""London TfL Real-Time Telemetry Collector.

Queries the TfL Unified API for bus arrival predictions across monitored routes (18, 24, 25, 73).
Uses the direct Line/{id}/Arrivals endpoint for ultra-fast, rate-limit-friendly sub-second polling.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from threading import Timer

import pandas as pd
import requests

LONDON_DIR = Path(__file__).resolve().parent.parent.parent / "Data"
STATIC_DIR = LONDON_DIR / "Static"
RAW_DIR = LONDON_DIR / "Raw"
REALTIME_DIR = LONDON_DIR / "RealTime"

RAW_DIR.mkdir(parents=True, exist_ok=True)
REALTIME_DIR.mkdir(parents=True, exist_ok=True)

# Load stops and lines
with open(STATIC_DIR / "lines_dict.json") as f:
    lines_dict = json.load(f)

# Optional API Credentials
try:
    from api_credentials import app_key_1
except ImportError:
    app_key_1 = os.environ.get("TFL_APP_KEY", "")


class RepeatedTimer:
    """Threaded repeated timer for periodic execution."""

    def __init__(self, interval: float, function, *args, **kwargs):
        self._timer: Timer | None = None
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
        if self._timer is not None:
            self._timer.cancel()
        self.is_running = False


def time_in_range(start: dt.time, end: dt.time, x: dt.time) -> bool:
    """Return true if x is in the range [start, end]."""
    if start <= end:
        return start <= x <= end
    return start <= x or x <= end


def get_arrival_data(requested_lines: list[str]) -> pd.DataFrame | None:
    """Retrieve and process live arrival predictions directly from TfL Line endpoints."""
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

    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if app_key_1:
        headers["app_key"] = app_key_1

    row_list = []
    for line in requested_lines:
        try:
            url = f"https://api.tfl.gov.uk/Line/{line}/Arrivals"
            resp = session.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                arrivals = resp.json()
                if isinstance(arrivals, list):
                    for bus in arrivals:
                        bus_dict = dict(bus)
                        bus_dict["direction"] = 1 if bus.get("direction") == "outbound" else 2
                        row_list.append({k: bus_dict.get(k, "") for k in keys})
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching Line {line}: {e}")

    if not row_list:
        print(
            f"[{dt.datetime.now().strftime('%H:%M:%S')}] ⚠️ No buses active on requested routes in this tick."
        )
        return None

    buses_df = pd.DataFrame(row_list, columns=keys)

    f_raw = RAW_DIR / "buses_data.csv"
    f_burst = REALTIME_DIR / "buses_data_burst.csv"

    if f_raw.exists() and f_raw.stat().st_size > 0:
        buses_df.to_csv(f_raw, mode="a", header=False, index=False)
    else:
        buses_df.to_csv(f_raw, mode="w", header=True, index=False)

    buses_df.to_csv(f_burst, header=True, index=False)

    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{now_str}] 🚀 New TfL Burst: {len(buses_df)} predictions across {buses_df['lineId'].nunique()} lines"
    )
    return buses_df


def main():
    """Main collector worker loop."""
    rt_started = False
    rt: RepeatedTimer | None = None

    start_time_day = dt.time(5, 0, 0)
    end_time_day = dt.time(23, 30, 0)
    requested_lines = list(lines_dict.keys())  # ["18", "24", "25", "73"]

    print(f"Starting TfL Data Collection Worker for London routes: {requested_lines}")

    try:
        while True:
            now = dt.datetime.now()
            if time_in_range(start_time_day, end_time_day, now.time()):
                if not rt_started:
                    print(f"[{now.strftime('%H:%M:%S')}] Activating TfL polling (interval: 35s)...")
                    rt = RepeatedTimer(35, get_arrival_data, requested_lines)
                    rt_started = True
            else:
                if rt_started:
                    print(
                        f"[{now.strftime('%H:%M:%S')}] Outside operating hours. Pausing collector..."
                    )
                    if rt is not None:
                        rt.stop()
                    rt_started = False

            time.sleep(10)
    except KeyboardInterrupt:
        if rt is not None:
            rt.stop()
        print("\nCollector stopped gracefully.")


if __name__ == "__main__":
    main()
