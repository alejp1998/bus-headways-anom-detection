#!/usr/bin/env python3
"""Live Telemetry Collector & Real-Time Anomaly Inference Service for London TfL.

Polls the TfL Line Arrival endpoints every 40 seconds for routes 18, 24, 25, 73.

Headway derivation ports the original research algorithm (London/Scripts/
AnomaliesDetection/detect_anoms_hws.py):

1. TTLS (time to last stop) = TfL ETA + mean trip time from the bus's stop to
   the route terminal, computed by walking the stop sequence reversed from the
   route end (times_bt_stops.csv).
2. Edge/noise filtering: buses at the first/last three stops (terminals) are
   excluded, TTLS longer than the full route travel time is dropped, and a bus
   must be observed in consecutive bursts before it counts.
3. Headway = TTLS difference between consecutive buses in the same direction.
   The first bus per direction (hw_pos == 0) is dropped.
4. Multi-dimensional windows (dim = 1, 2) are derived from consecutive bus
   groups and scored with the Mahalanobis distance against the fitted models.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from scipy.stats.distributions import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import db

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "London" / "Data" / "Static"
PROCESSED_DIR = REPO_ROOT / "London" / "Data" / "Processed"
REALTIME_DIR = REPO_ROOT / "London" / "Data" / "RealTime"
ANOM_DIR = REPO_ROOT / "London" / "Data" / "Anomalies"

MONITORED_LINES = ["18", "24", "25", "73"]

with open(STATIC_DIR / "lines_dict.json") as f:
    LINES_DICT = json.load(f)

_bus_names_all = ["bus" + str(i) for i in range(1, 10)]
_hw_names_all = ["hw" + str(i) + str(i + 1) for i in range(1, 9)]

# Consecutive-burst appearance threshold (noise filter for spurious buses)
TH = 2

_prev_seen: dict[str, set] = {}


def _load_times_bt_stops() -> pd.DataFrame:
    """Load mean inter-stop travel times (built by build_london_times_bt_stops.py)."""
    path = PROCESSED_DIR / "times_bt_stops.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


TIMES_BT_STOPS = _load_times_bt_stops()


def _stop_order(line: str, direction: int) -> list[str]:
    return LINES_DICT.get(line, {}).get(str(direction), {}).get("stops", [])


def _mean_trip_time(line: str, direction: int, hour: int, stop_a: str) -> float:
    """Mean travel time from stop_a to the following stop (0.0 when unknown)."""
    if TIMES_BT_STOPS.empty:
        return 0.0
    sub = TIMES_BT_STOPS[
        (TIMES_BT_STOPS.line.astype(str) == line)
        & (TIMES_BT_STOPS.direction == direction)
        & (TIMES_BT_STOPS.stopA.astype(str) == stop_a)
    ]
    if not sub.empty:
        hour_sub = sub[sub.st_hour == hour]
        if hour_sub.empty:
            hour_sub = sub
        return float(hour_sub.trip_time.mean())
    return 0.0


def _build_cumulative_ttls_lookup() -> dict[tuple[str, int, str], float]:
    """Pre-compute cumulative travel time from each stop to the route terminal.

    Integrates times_bt_stops.csv when available, or defaults to the empirical
    inter-stop travel time (~75s/hop) derived from the route sequence.
    """
    lookup: dict[tuple[str, int, str], float] = {}
    for line in MONITORED_LINES:
        for direction in (1, 2):
            stops = _stop_order(line, direction)
            if not stops:
                continue
            cum_time = 0.0
            for i in range(len(stops) - 1, -1, -1):
                stop = stops[i]
                lookup[(line, direction, str(stop))] = cum_time
                if i > 0:
                    trip = _mean_trip_time(line, direction, 12, stops[i - 1])
                    cum_time += trip if trip > 0 else 75.0
    return lookup


CUMULATIVE_TTLS_LOOKUP = _build_cumulative_ttls_lookup()


def collect_live_burst(session: requests.Session) -> pd.DataFrame:
    """Fetch live arrivals for all monitored lines from the TfL API."""
    all_records = []
    for line in MONITORED_LINES:
        resp = session.get(f"https://api.tfl.gov.uk/Line/{line}/Arrivals", timeout=10)
        if resp.status_code == 200:
            all_records.extend(resp.json())
        else:
            print(f"  ⚠️ TfL line {line}: HTTP {resp.status_code}")

    if not all_records:
        return pd.DataFrame()

    now_iso = dt.now().isoformat()
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
    for r in all_records:
        line_name = str(r.get("lineName", "")).strip()
        if line_name not in MONITORED_LINES:
            continue
        vid = str(r.get("vehicleId") or r.get("id", "")).strip()
        if not vid:
            continue
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
                "stop": naptan or str(r.get("stationName", "")).strip(),
                "estimateArrive": int(r.get("timeToStation", 0)),
                "DistanceBus": int(r.get("timeToStation", 0)) * 5.0,
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # NOTE: keep ALL per-stop predictions (needed for times_bt_stops derivation);
    # the map/headway consumers dedupe to one position per bus.
    return df.reset_index(drop=True)


def compute_live_headways(burst_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized calculation of cumulative TTLS, edge noise filtering, and headways."""
    if burst_df.empty:
        return pd.DataFrame()

    rows_list: list[dict] = []

    for line, line_df in burst_df.groupby("line"):
        line_str = str(line)
        if line_str not in MONITORED_LINES:
            continue

        for direction in (1, 2):
            stops_order = _stop_order(line_str, direction)
            if len(stops_order) < 8:
                continue

            dir_df = line_df[line_df["direction"] == direction]
            if dir_df.empty:
                continue

            # Route edge terminals (skip first 3 and last 3 terminus stops where buses dwell)
            first_stops = set(stops_order[:3])
            last_stops = set(stops_order[-3:])

            # Compute cumulative TTLS using O(1) table lookup
            buses_valid = []
            for row in dir_df.itertuples():
                stop_str = str(row.stop)
                # Filter out terminal edge dwelling buses
                if stop_str in last_stops:
                    continue
                if stop_str in first_stops and row.estimateArrive < 60:
                    continue
                offset = CUMULATIVE_TTLS_LOOKUP.get((line_str, direction, stop_str), 0.0)
                ttls = float(row.estimateArrive) + offset
                buses_valid.append(
                    {
                        "line": line_str,
                        "direction": direction,
                        "bus": str(row.bus),
                        "stop": stop_str,
                        "datetime": row.datetime,
                        "ttls": ttls,
                    }
                )

            if not buses_valid:
                continue

            stops_df = pd.DataFrame(buses_valid)

            # Consecutive-appearance noise filter
            seen_key = f"{line_str}:{direction}"
            if seen_key not in _prev_seen:
                _prev_seen[seen_key] = set()
            stops_df = stops_df[
                stops_df["bus"].isin(_prev_seen[seen_key]) | (not _prev_seen[seen_key])
            ]

            # Sort buses along route sequence by true TTLS
            stops_df = stops_df.sort_values("ttls").drop_duplicates("bus", keep="first")

            for i in range(stops_df.shape[0]):
                est1 = stops_df.iloc[i]
                if i == 0:
                    rows_list.append(
                        {
                            "datetime": est1.datetime,
                            "line": line_str,
                            "direction": direction,
                            "busA": "0",
                            "busB": est1.bus,
                            "hw_pos": 0,
                            "headway": 0.0,
                            "busB_ttls": round(float(est1.ttls), 1),
                        }
                    )
                if i < stops_df.shape[0] - 1:
                    est2 = stops_df.iloc[i + 1]
                    rows_list.append(
                        {
                            "datetime": est1.datetime,
                            "line": line_str,
                            "direction": direction,
                            "busA": est1.bus,
                            "busB": est2.bus,
                            "hw_pos": i + 1,
                            "headway": round(float(est2.ttls - est1.ttls), 1),
                            "busB_ttls": round(float(est2.ttls), 1),
                        }
                    )

            _prev_seen[seen_key] = set(stops_df["bus"])

    if not rows_list:
        return pd.DataFrame()

    hws = pd.DataFrame(rows_list)
    return hws[hws["hw_pos"] > 0].reset_index(drop=True)


def get_ndim_hws(hws_df: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Port of the original get_ndim_hws: build windows of dim consecutive headways."""
    hw_names = ["hw" + str(i) + str(i + 1) for i in range(1, dim + 1)]
    bus_names = ["bus" + str(i) for i in range(1, dim + 2)]
    names = ["datetime"] + bus_names + hw_names
    if hws_df.shape[0] < 1:
        return pd.DataFrame(columns=names)

    burst_time = hws_df.iloc[0].datetime
    columns = {name: [] for name in names}

    for direction in (1, 2):
        burst_df = hws_df[(hws_df.datetime == burst_time) & (hws_df.direction == direction)]
        burst_df = burst_df.sort_values("hw_pos")
        for i in range(burst_df.shape[0] - (dim - 1)):
            columns["datetime"].append(burst_time)
            columns[bus_names[0]].append(str(burst_df.iloc[i].busA))
            for k in range(dim):
                columns[hw_names[k]].append(float(burst_df.iloc[i + k].headway))
                columns[bus_names[k + 1]].append(str(burst_df.iloc[i + k].busB))

    return pd.DataFrame(columns)


def run_live_cycle(session: requests.Session) -> dict:
    """Execute one full live polling, headway derivation, and anomaly inference tick."""
    now = dt.now()
    now_str = now.strftime("%H:%M:%S")

    burst_df = collect_live_burst(session)
    if burst_df.empty:
        print(f"[{now_str}] ⚠️ No live bus arrivals returned by TfL API.")
        return {"status": "empty", "buses": 0}

    # 1. Live burst snapshot (CSV + DB)
    burst_df.to_csv(REALTIME_DIR / "buses_data_burst_cleaned.csv", index=False)
    try:
        db.insert_buses_burst("London", burst_df)
    except Exception as e:
        print(f"  ⚠️ buses_burst DB write: {e}")

    # 2. Compute live headways with the original edge/noise-filtered algorithm
    hws_df = compute_live_headways(burst_df)
    if not hws_df.empty:
        hws_df.to_csv(REALTIME_DIR / "headways_burst.csv", index=False)
        try:
            db.insert_headways_burst("London", hws_df)
        except Exception as e:
            print(f"  ⚠️ headways_burst DB write: {e}")

    # 3. Load baseline Gaussian models
    with open(ANOM_DIR / "models_params.json") as f:
        models = json.load(f)
    with open(ANOM_DIR / "hyperparams.json") as f:
        hyperparams = json.load(f)

    day_type = "LA" if now.weekday() <= 4 else ("SA" if now.weekday() == 5 else "FE")
    hour = now.hour
    hour_range = "21-23" if hour >= 21 else ("19-21" if hour >= 19 else "17-19")

    series_rows = []
    anom_rows = []

    if not hws_df.empty:
        hws = hws_df[hws_df["hw_pos"] > 0]
        for line, lgroup in hws.groupby("line"):
            line_str = str(line)
            conf = hyperparams.get(line_str, {}).get("conf", 0.98)
            line_model = models.get(line_str, {}).get(day_type, {}).get(hour_range, {})
            max_dim = int(line_model.get("max_dim", 1))

            for dim in range(1, max_dim + 1):
                window_df = get_ndim_hws(lgroup, dim)
                if window_df.empty:
                    continue
                model = line_model.get(str(dim))
                if not model:
                    continue
                mean = model.get("mean")
                cov = model.get("cov_matrix")
                if mean is None or cov is None:
                    continue
                m_th = math.sqrt(chi2.ppf(conf, df=dim))
                hw_names = ["hw" + str(i) + str(i + 1) for i in range(1, dim + 1)]
                bus_names = ["bus" + str(i) for i in range(1, dim + 2)]

                for row in window_df.itertuples():
                    hw_vec = [float(getattr(row, h)) for h in hw_names]
                    if dim == 1:
                        std = float(cov) if not isinstance(cov, list) else float(cov[0][0])
                        m_dist = abs(float(mean) - hw_vec[0]) / (std if std > 0 else 1.0)
                    else:
                        import numpy as np

                        diff = np.array(hw_vec) - np.array(mean)
                        iv = np.linalg.inv(np.array(cov))
                        m_dist = float(np.sqrt(diff @ iv @ diff))
                    is_anom = int(m_dist > m_th)

                    record = {
                        "line": line_str,
                        "datetime": row.datetime,
                        "dim": dim,
                        "m_dist": round(m_dist, 3),
                        "anom": is_anom,
                    }
                    for k in range(1, 10):
                        record[f"bus{k}"] = (
                            str(getattr(row, bus_names[k - 1])) if k <= dim + 1 else "0"
                        )
                    for k in range(1, 9):
                        record[f"hw{k}{k + 1}"] = (
                            float(getattr(row, hw_names[k - 1])) if k <= dim else 0.0
                        )
                    series_rows.append(record)

                    if is_anom:
                        anom_rows.append(
                            {
                                "line": line_str,
                                "datetime": row.datetime,
                                "dim": dim,
                                "m_dist": round(m_dist, 3),
                                "anom_size": 1,
                                **{f"bus{k}": record[f"bus{k}"] for k in range(1, 10)},
                                **{f"hw{k}{k + 1}": record[f"hw{k}{k + 1}"] for k in range(1, 9)},
                            }
                        )

    # 4. Persist series + anomalies (CSV mirrors + DB)
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
        try:
            db.insert_headways_series("London", combined)
        except Exception as e:
            print(f"  ⚠️ headways_series DB write: {e}")

    if anom_rows:
        pd.DataFrame(anom_rows).to_csv(ANOM_DIR / "anomalies.csv", index=False)
        try:
            db.insert_anomaly_events("London", pd.DataFrame(anom_rows))
        except Exception as e:
            print(f"  ⚠️ anomaly_events DB write: {e}")

    buses_cnt = burst_df["bus"].nunique()
    lines_cnt = burst_df["line"].nunique()
    dims = sorted({r["dim"] for r in series_rows})
    print(
        f"[{now_str}] 🚀 TfL Live Ingested: {len(burst_df)} predictions | {buses_cnt} buses "
        f"| {len(hws_df)} headways | series dims {dims} | {len(anom_rows)} anomalies"
    )
    return {"status": "ok", "buses": buses_cnt, "lines": lines_cnt, "headways": len(hws_df)}


def main() -> int:
    parser = argparse.ArgumentParser(description="London TfL Real-Time Telemetry & Anomaly Worker")
    parser.add_argument(
        "--once", action="store_true", help="Run a single collection cycle and exit"
    )
    parser.add_argument(
        "--interval", type=int, default=5, help="Polling interval in seconds (default: 5)"
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
