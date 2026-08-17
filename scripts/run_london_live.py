#!/usr/bin/env python3
"""Live Telemetry Collector & Real-Time Anomaly Inference Service for London TfL.

Polls the TfL Line Arrival endpoints every 45 seconds for routes 18, 24, 25, 73:
1. Fetches arrival predictions across all monitored lines.
2. Cleans & deduplicates vehicle positions and arrival ETAs.
3. Computes consecutive bus headways and time-to-last-stop (TTLS).
4. Evaluates real-time Mahalanobis anomaly distance against active models_params.json.
5. Updates live burst files (headways_burst.csv, series.csv, anomalies.csv).
6. Appends to the weekly buffer (buses_data_week_cleaned.csv) for Monday auto-rotation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import chi2

ROOT_DIR = Path(__file__).resolve().parent.parent
LONDON_DIR = ROOT_DIR / "London" / "Data"
STATIC_DIR = LONDON_DIR / "Static"
REALTIME_DIR = LONDON_DIR / "RealTime"
ANOM_DIR = LONDON_DIR / "Anomalies"

REALTIME_DIR.mkdir(parents=True, exist_ok=True)
ANOM_DIR.mkdir(parents=True, exist_ok=True)

# Load static lines and stops
with open(STATIC_DIR / "lines_dict.json") as f:
    LINES_DICT = json.load(f)

MONITORED_LINES = list(LINES_DICT.keys())  # ["18", "24", "25", "73"]

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
    for line in MONITORED_LINES:
        try:
            url = f"https://api.tfl.gov.uk/Line/{line}/Arrivals"
            resp = session.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    records.extend(data)
            time.sleep(0.08)
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    now = dt.datetime.now()
    rows = []
    for r in records:
        line_name = str(r.get("lineName", "")).strip()
        if line_name in MONITORED_LINES:
            vehicle_id = r.get("vehicleId") or r.get("id")
            # Extract numeric id or hash string for vehicle
            bus_num = (
                int("".join(filter(str.isdigit, str(vehicle_id))))
                if any(c.isdigit() for c in str(vehicle_id))
                else abs(hash(str(vehicle_id))) % 9000 + 1000
            )
            rows.append(
                {
                    "line": line_name,
                    "datetime": now.isoformat(),
                    "bus": bus_num,
                    "vehicleId": str(vehicle_id),
                    "destination": r.get("destinationName", ""),
                    "stop": str(r.get("naptanId") or r.get("stationName", "")),
                    "estimateArrive": int(r.get("timeToStation", 0)),
                    "DistanceBus": int(r.get("timeToStation", 0)) * 5,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduplicate buses keeping closest arrival to current stop
    df = df.sort_values("estimateArrive").drop_duplicates(["line", "bus"], keep="first")
    return df.reset_index(drop=True)


def compute_live_headways(burst_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate headway spacing and TTLS for consecutive buses on each line."""
    if burst_df.empty:
        return pd.DataFrame()

    hws_rows = []
    for line, lgroup in burst_df.groupby("line"):
        # Sort buses by arrival time
        dir_buses = lgroup.sort_values("estimateArrive")
        n = len(dir_buses)
        if n < 2:
            continue

        for i in range(n - 1):
            bus_a = dir_buses.iloc[i]
            bus_b = dir_buses.iloc[i + 1]
            hw_val = abs(bus_b["estimateArrive"] - bus_a["estimateArrive"])
            hws_rows.append(
                {
                    "line": str(line),
                    "datetime": bus_a["datetime"],
                    "direction": 1,
                    "busA": bus_a["bus"],
                    "busB": bus_b["bus"],
                    "hw_pos": i + 1,
                    "headway": hw_val,
                    "busA_ttls": bus_a["estimateArrive"],
                    "busB_ttls": bus_b["estimateArrive"],
                }
            )

    return pd.DataFrame(hws_rows) if hws_rows else pd.DataFrame()


def run_live_cycle(session: requests.Session) -> dict:
    """Execute one full polling, headway derivation, and anomaly inference tick."""
    now = dt.datetime.now()
    now_str = now.strftime("%H:%M:%S")

    burst_df = collect_live_burst(session)
    if burst_df.empty:
        print(f"[{now_str}] ⚠️ No bus arrivals returned in this cycle.")
        return {"status": "empty", "buses": 0}

    # 1. Save cleaned burst
    burst_df.to_csv(REALTIME_DIR / "buses_data_burst_cleaned.csv", index=False)

    # 2. Append to weekly buffer
    week_csv = REALTIME_DIR / "buses_data_week_cleaned.csv"
    if not week_csv.exists() or week_csv.stat().st_size == 0:
        burst_df.to_csv(week_csv, index=False)
    else:
        burst_df.to_csv(week_csv, mode="a", header=False, index=False)

    # 3. Compute headways
    hws_df = compute_live_headways(burst_df)
    if not hws_df.empty:
        hws_df.to_csv(REALTIME_DIR / "headways_burst.csv", index=False)

    # 4. Load models and hyperparams for anomaly detection
    with open(ANOM_DIR / "models_params.json") as f:
        models = json.load(f)

    with open(ANOM_DIR / "hyperparams.json") as f:
        hyperparams = json.load(f)

    day_type, hour_range = get_current_stratum(now)
    if hour_range is None:
        hour_range = "17-19"  # Peak evening baseline fallback

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
                    "bus1": row.busA,
                    "bus2": row.busB,
                    "bus3": 0,
                    "bus4": 0,
                    "bus5": 0,
                    "bus6": 0,
                    "bus7": 0,
                    "bus8": 0,
                    "bus9": 0,
                    "hw12": int(hw),
                    "hw23": 0,
                    "hw34": 0,
                    "hw45": 0,
                    "hw56": 0,
                    "hw67": 0,
                    "hw78": 0,
                    "hw89": 0,
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
                            "bus1": row.busA,
                            "bus2": row.busB,
                            "bus3": 0,
                            "bus4": 0,
                            "hw12": int(hw),
                            "hw23": 0,
                            "hw34": 0,
                        }
                    )

    # Save series and anomalies
    if series_rows:
        new_series = pd.DataFrame(series_rows)
        series_file = REALTIME_DIR / "series.csv"
        if series_file.exists() and series_file.stat().st_size > 0:
            try:
                prev_series = pd.read_csv(series_file)
                combined = pd.concat([prev_series.tail(200), new_series], ignore_index=True)
            except Exception:
                combined = new_series
        else:
            combined = new_series
        combined.to_csv(series_file, index=False)

    if anom_rows:
        pd.DataFrame(anom_rows).to_csv(ANOM_DIR / "anomalies.csv", index=False)

        # Database persistence (SQLite WAL)
    try:
        import sys

        sys.path.insert(0, str(ROOT_DIR))
        from core import db

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
        f"[{now_str}] 🚀 TfL Burst: {len(burst_df)} live predictions | {buses_cnt} active buses across {lines_cnt} lines | {len(series_rows)} headways | {len(anom_rows)} anomalies"
    )
    return {"status": "ok", "buses": buses_cnt, "lines": lines_cnt, "anomalies": len(anom_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="London TfL Real-Time Telemetry & Anomaly Worker")
    parser.add_argument(
        "--once", action="store_true", help="Run a single collection cycle and exit"
    )
    parser.add_argument(
        "--interval", type=int, default=45, help="Polling interval in seconds (default: 45)"
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
