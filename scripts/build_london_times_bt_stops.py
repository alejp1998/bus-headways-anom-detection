#!/usr/bin/env python3
"""Build times_bt_stops.csv for London from the historical cleaned telemetry.

For each line/direction/hour, computes mean travel time between consecutive stops
by differencing the TfL API ETAs of the same bus in the same burst.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "London" / "Data" / "Static"
PROCESSED_DIR = ROOT / "London" / "Data" / "Processed"

with open(STATIC_DIR / "lines_dict.json") as f:
    LINES_DICT = json.load(f)

LINES = ["18", "24", "25", "73"]


def _infer_direction(row, lines_dict):
    """Infer bus direction from destination name, then stop membership."""
    line = str(row["line"])
    if line not in lines_dict:
        return None
    dest = str(row.get("destination", "")).strip()
    stops1 = set(lines_dict[line].get("1", {}).get("stops", []))
    stops2 = set(lines_dict[line].get("2", {}).get("stops", []))
    stop = str(row["stop"])
    if dest:
        d1 = str(lines_dict[line].get("destinations", ["", ""])[1])
        d2 = str(lines_dict[line].get("destinations", ["", ""])[0])
        if d1 and (d1.lower() in dest.lower() or dest.lower() in d1.lower()):
            return 1
        if d2 and (d2.lower() in dest.lower() or dest.lower() in d2.lower()):
            return 2
    if stop in stops1 and stop not in stops2:
        return 1
    if stop in stops2 and stop not in stops1:
        return 2
    return None


def build_times_bt_stops(buses: pd.DataFrame) -> pd.DataFrame:
    """Derive mean inter-stop travel times from per-bus ETA differences."""
    buses = buses.copy()
    buses["datetime"] = pd.to_datetime(buses["datetime"], errors="coerce")
    buses["stop"] = buses["stop"].astype(str)
    buses["line"] = buses["line"].astype(str)
    buses["estimateArrive"] = pd.to_numeric(buses["estimateArrive"], errors="coerce")
    buses = buses.dropna(subset=["datetime", "estimateArrive"])

    # Infer direction where missing
    if "direction" not in buses.columns or buses["direction"].fillna(0).eq(0).all():
        buses["direction"] = buses.apply(lambda r: _infer_direction(r, LINES_DICT) or 0, axis=1)

    rows = []
    for line in LINES:
        for direction in ["1", "2"]:
            stops_order = LINES_DICT.get(line, {}).get(direction, {}).get("stops", [])
            if not stops_order:
                continue
            next_stop = {s: stops_order[i + 1] for i, s in enumerate(stops_order[:-1])}
            line_df = buses[(buses["line"] == line) & (buses["direction"] == int(direction))]
            if line_df.empty:
                continue
            for burst_time, burst in line_df.groupby("datetime"):
                for _bus, bus_rows in burst.groupby("bus"):
                    # Closest stop first
                    bus_rows = bus_rows.sort_values("estimateArrive")
                    stops_seen = list(bus_rows["stop"])
                    for i, stop in enumerate(stops_seen[:-1]):
                        nxt = next_stop.get(stop)
                        if nxt is None:
                            continue
                        # Find this bus's ETA at the next stop in the same burst
                        nxt_rows = bus_rows[bus_rows["stop"] == nxt]
                        if nxt_rows.empty:
                            continue
                        eta_now = bus_rows.iloc[i]["estimateArrive"]
                        eta_next = nxt_rows.iloc[0]["estimateArrive"]
                        trip = eta_next - eta_now  # time from stop -> next stop
                        if 0 < trip < 900:
                            rows.append(
                                {
                                    "line": int(line),
                                    "direction": int(direction),
                                    "st_hour": burst_time.hour,
                                    "stopA": stop,
                                    "stopB": nxt,
                                    "trip_time": round(trip, 3),
                                }
                            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Mean per (line, direction, hour, stop pair)
    return (
        df.groupby(["line", "direction", "st_hour", "stopA", "stopB"], as_index=False)["trip_time"]
        .mean()
        .round(3)
    )


def main() -> None:
    """Build times_bt_stops from the clean live telemetry in the SQLite database."""
    import sqlite3

    conn = sqlite3.connect(ROOT / "Data" / "runtime" / "transit_telemetry.db")
    query = """
        SELECT line, direction, stop, bus, created_at AS datetime, destination,
               estimate_arrive AS estimateArrive
        FROM buses_burst
        WHERE city = 'London'
    """
    buses = pd.read_sql_query(query, conn)
    conn.close()

    # Map stop NAMES -> naptan IDs via stops.csv so historical rows match lines_dict
    stops_map = {}
    stops_csv = STATIC_DIR / "stops.csv"
    if stops_csv.exists():
        st = pd.read_csv(stops_csv)
        for _, row in st.iterrows():
            stops_map[str(row["name"]).strip()] = str(row["id"])
    buses["stop"] = buses["stop"].astype(str).str.strip()
    buses["stop"] = buses["stop"].map(lambda s: stops_map.get(s, s))

    print(f"  {len(buses)} rows from DB")
    if buses.empty:
        print("No data - run the collector first.")
        return
    tbs = build_times_bt_stops(buses)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "times_bt_stops.csv"
    tbs.to_csv(out, index=False)
    print(f"Wrote {out} with {len(tbs)} stop-pair/hour records")
    print(tbs.head(10).to_string())


if __name__ == "__main__":
    main()
