#!/usr/bin/env python3
"""Live Telemetry Collector & Real-Time Anomaly Inference Service for London TfL.

Polls the TfL Line Arrival endpoints every 40 seconds for routes 18, 24, 25, 73:
1. Fetches real arrival predictions across all monitored lines via TfL Unified API.
2. Deduplicates vehicles preserving exact vehicle registration plates (e.g. LF75ONU, LTZ1209).
3. Computes consecutive bus headways and time-to-last-stop (TTLS) per route direction.
4. Evaluates real-time Mahalanobis anomaly distance against active models_params.json.
5. Updates SQLite database (Data/transit_telemetry.db) and real-time cache files.
6. Appends genuine telemetry to the weekly buffer (buses_data_week_cleaned.csv) for Monday rotation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import chi2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core import db

LONDON_DIR = ROOT_DIR / "London" / "Data"
STATIC_DIR = LONDON_DIR / "Static"
REALTIME_DIR = LONDON_DIR / "RealTime"
ANOM_DIR = LONDON_DIR / "Anomalies"

REALTIME_DIR.mkdir(parents=True, exist_ok=True)
ANOM_DIR.mkdir(parents=True, exist_ok=True)

# Load static lines and stops
with open(STATIC_DIR / "lines_dict.json") as f:
    LINES_DICT = json.load(f)

MONITORED_LINES = ["18", "24", "25", "73"]

HOUR_RANGES = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]


def get_current_stratum(now: dt.datetime) -> tuple[str, str | None]:
    """Determine day type and hour range key for current time."""
    weekday = now.weekday()
    day_type = "LA" if weekday <= 4 else ("SA" if weekday == 5 else "FE")

    hour_range = None
    for h_start, h_end in HOUR_RANGES:
        if h_start <= now.hour < h_end:
            hour_range = f"{h_start}-{h_end}"
            break

    return day_type, hour_range


def collect_live_burst(session: requests.Session) -> pd.DataFrame:
    """Fetch live arrival data across all monitored London lines via Line endpoints."""
    records = []
    headers = {"Content-Type": "application/json"}

    for line in MONITORED_LINES:
        try:
            url = f"https://api.tfl.gov.uk/Line/{line}/Arrivals"
            resp = session.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    records.extend(data)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching Line {line}: {e}")

    if not records:
        return pd.DataFrame()

    now_iso = dt.datetime.now().isoformat()
    # Load stop coordinates map for accurate live vehicle mapping
    stops_coords = {}
    stops_file = STATIC_DIR / "stops.csv"
    if stops_file.exists():
        try:
            _st = pd.read_csv(stops_file)
            for _, row in _st.iterrows():
                stops_coords[str(row["id"])] = (float(row["lat"]), float(row["lon"]))
        except Exception:
            pass

    rows = []
    for r in records:
        line_name = str(r.get("lineName", "")).strip()
        if line_name in MONITORED_LINES:
            vid = str(r.get("vehicleId") or r.get("id", "")).strip()
            if not vid:
                continue

            # Determine direction: 1 = outbound, 2 = inbound
            dir_val = 1 if r.get("direction") == "outbound" else 2
            naptan = str(r.get("naptanId", "")).strip()
            lat, lon = stops_coords.get(naptan, (51.5074, -0.1278))

            rows.append(
                {
                    "line": line_name,
                    "datetime": now_iso,
                    "bus": vid,
                    "vehicleId": vid,
                    "destination": str(r.get("destinationName", "")).strip(),
                    "direction": dir_val,
                    "stop": str(r.get("stationName") or naptan).strip(),
                    "estimateArrive": int(r.get("timeToStation", 0)),
                    "DistanceBus": int(r.get("timeToStation", 0)) * 5.0,
                    "lat": lat,
                    "lon": lon,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduplicate: each unique vehicle on a line is kept at its closest upcoming stop
    df = df.sort_values("estimateArrive").drop_duplicates(["line", "bus"], keep="first")
    return df.reset_index(drop=True)


def compute_live_headways(burst_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate headway spacing and TTLS for consecutive buses on each line and direction."""
    if burst_df.empty:
        return pd.DataFrame()

    hws_rows = []
    for (line, direction), dir_buses in burst_df.groupby(["line", "direction"]):
        # Sort buses by arrival time along route
        sorted_buses = dir_buses.sort_values("estimateArrive")
        n = len(sorted_buses)
        if n < 2:
            continue

        for i in range(n - 1):
            bus_a = sorted_buses.iloc[i]
            bus_b = sorted_buses.iloc[i + 1]
            hw_val = abs(bus_b["estimateArrive"] - bus_a["estimateArrive"])
            hws_rows.append(
                {
                    "line": str(line),
                    "datetime": bus_a["datetime"],
                    "direction": int(direction),
                    "busA": str(bus_a["bus"]),
                    "busB": str(bus_b["bus"]),
                    "hw_pos": i + 1,
                    "headway": float(hw_val),
                    "busA_ttls": float(bus_a["estimateArrive"]),
                    "busB_ttls": float(bus_b["estimateArrive"]),
                }
            )

    return pd.DataFrame(hws_rows) if hws_rows else pd.DataFrame()


def run_live_cycle(session: requests.Session) -> dict:
    """Execute one full live polling, headway derivation, and anomaly inference tick."""
    now = dt.datetime.now()
    now_str = now.strftime("%H:%M:%S")

    burst_df = collect_live_burst(session)
    if burst_df.empty:
        print(f"[{now_str}] ⚠️ No live bus arrivals returned by TfL API.")
        return {"status": "empty", "buses": 0}

    # 1. Update CSV burst & weekly buffer
    burst_df.to_csv(REALTIME_DIR / "buses_data_burst_cleaned.csv", index=False)
    week_csv = REALTIME_DIR / "buses_data_week_cleaned.csv"
    if not week_csv.exists() or week_csv.stat().st_size == 0:
        burst_df.to_csv(week_csv, index=False)
    else:
        burst_df.to_csv(week_csv, mode="a", header=False, index=False)

    # 2. Compute live headways
    hws_df = compute_live_headways(burst_df)
    if not hws_df.empty:
        hws_df.to_csv(REALTIME_DIR / "headways_burst.csv", index=False)

    # 3. Load baseline Gaussian models
    with open(ANOM_DIR / "models_params.json") as f:
        models = json.load(f)

    with open(ANOM_DIR / "hyperparams.json") as f:
        hyperparams = json.load(f)

    day_type, hour_range = get_current_stratum(now)
    if hour_range is None:
        hour_range = "17-19"  # Fallback hour window

    series_rows = []
    anom_rows = []

    if not hws_df.empty:
        for line, lgroup in hws_df.groupby("line"):
            line_str = str(line)
            conf = hyperparams.get(line_str, {}).get("conf", 0.98)
            line_model = models.get(line_str, {}).get(day_type, {}).get(hour_range, {})

            m1 = line_model.get("1", {})
            mu1 = float(m1.get("mean", 360.0))
            std1 = float(m1.get("cov_matrix", 120.0))
            m_th1 = math.sqrt(chi2.ppf(conf, df=1))

            for row in lgroup.itertuples():
                hw = float(row.headway)
                m_dist = abs(hw - mu1) / (std1 if std1 > 0 else 1.0)
                is_anom = int(m_dist > m_th1)

                record = {
                    "line": line_str,
                    "datetime": row.datetime,
                    "dim": 1,
                    "m_dist": round(m_dist, 3),
                    "anom": is_anom,
                    "bus1": str(row.busA),
                    "bus2": str(row.busB),
                    "bus3": "0",
                    "bus4": "0",
                    "bus5": "0",
                    "bus6": "0",
                    "bus7": "0",
                    "bus8": "0",
                    "bus9": "0",
                    "hw12": float(hw),
                    "hw23": 0.0,
                    "hw34": 0.0,
                    "hw45": 0.0,
                    "hw56": 0.0,
                    "hw67": 0.0,
                    "hw78": 0.0,
                    "hw89": 0.0,
                }
                series_rows.append(record)

                if is_anom:
                    anom_rows.append(
                        {
                            "line": line_str,
                            "datetime": row.datetime,
                            "dim": 1,
                            "m_dist": round(m_dist, 3),
                            "anom_size": 1,
                            "bus1": str(row.busA),
                            "bus2": str(row.busB),
                            "bus3": "0",
                            "bus4": "0",
                            "hw12": float(hw),
                            "hw23": 0.0,
                            "hw34": 0.0,
                        }
                    )

    # Save series and anomalies CSV
    if series_rows:
        new_series = pd.DataFrame(series_rows)
        series_file = REALTIME_DIR / "series.csv"
        if series_file.exists() and series_file.stat().st_size > 0:
            try:
                prev_series = pd.read_csv(series_file)
                combined = pd.concat([prev_series.tail(300), new_series], ignore_index=True)
            except Exception:
                combined = new_series
        else:
            combined = new_series
        combined.to_csv(series_file, index=False)

    if anom_rows:
        pd.DataFrame(anom_rows).to_csv(ANOM_DIR / "anomalies.csv", index=False)

    # 4. Insert directly into high-performance SQLite database
    try:
        db.insert_buses_burst("London", burst_df)
        if not hws_df.empty:
            db.insert_headways_burst("London", hws_df)
        if series_rows:
            db.insert_headways_series("London", pd.DataFrame(series_rows))
        if anom_rows:
            db.insert_anomaly_events("London", pd.DataFrame(anom_rows))
    except Exception as e:
        print(f"Database write error: {e}")

    buses_cnt = burst_df["bus"].nunique()
    lines_cnt = burst_df["line"].nunique()
    print(
        f"[{now_str}] 🚀 TfL Live Ingested: {len(burst_df)} live predictions | {buses_cnt} active buses across {lines_cnt} lines | {len(hws_df)} headways | {len(anom_rows)} anomalies"
    )
    return {
        "status": "ok",
        "buses": buses_cnt,
        "lines": lines_cnt,
        "headways": len(hws_df),
        "anomalies": len(anom_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="London TfL Real-Time Telemetry & Anomaly Worker")
    parser.add_argument(
        "--once", action="store_true", help="Run a single collection cycle and exit"
    )
    parser.add_argument(
        "--interval", type=int, default=40, help="Polling interval in seconds (default: 40)"
    )
    args = parser.parse_args()

    print("========================================================")
    print(" London TfL Real-Time Collector & Anomaly Inference")
    print(f" Monitored Routes: {', '.join(MONITORED_LINES)}")
    print(f" Polling Interval: {args.interval}s")
    print("========================================================\n")

    session = requests.Session()

    if args.once:
        run_live_cycle(session)
        return 0

    while True:
        try:
            run_live_cycle(session)
        except Exception as e:
            print(f"Error in collection loop: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
