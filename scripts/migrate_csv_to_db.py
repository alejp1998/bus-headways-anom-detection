#!/usr/bin/env python3
"""Migrate existing CSV records and historical JSONs into SQLite database."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core import db

db.init_db()

print("Migrating CSV telemetry & history records into SQLite database...")

for city in ["Madrid", "London"]:
    city_dir = ROOT_DIR / city / "Data"
    realtime_dir = city_dir / "RealTime"
    anom_dir = city_dir / "Anomalies"
    hist_dir = city_dir / "History"

    # 1. Buses Burst
    burst_csv = realtime_dir / "buses_data_burst_cleaned.csv"
    if not burst_csv.exists() or burst_csv.stat().st_size == 0:
        burst_csv = realtime_dir / "buses_data_burst.csv"

    if burst_csv.exists() and burst_csv.stat().st_size > 0:
        df_burst = pd.read_csv(burst_csv)
        db.insert_buses_burst(city, df_burst)
        print(f"[{city}] Migrated {len(df_burst)} bus burst records.")

    # 2. Headways Burst
    hws_csv = realtime_dir / "headways_burst.csv"
    if hws_csv.exists() and hws_csv.stat().st_size > 0:
        df_hws = pd.read_csv(hws_csv)
        db.insert_headways_burst(city, df_hws)
        print(f"[{city}] Migrated {len(df_hws)} headway burst records.")

    # 3. Series CSV
    series_csv = realtime_dir / "series.csv"
    if series_csv.exists() and series_csv.stat().st_size > 0:
        df_series = pd.read_csv(series_csv)
        db.insert_headways_series(city, df_series)
        print(f"[{city}] Migrated {len(df_series)} series records.")

    # 4. Anomalies CSV
    anom_csv = anom_dir / "anomalies.csv"
    if anom_csv.exists() and anom_csv.stat().st_size > 0:
        df_anom = pd.read_csv(anom_csv)
        db.insert_anomaly_events(city, df_anom)
        print(f"[{city}] Migrated {len(df_anom)} anomaly records.")

    # 5. Weekly History JSONs
    if hist_dir.exists():
        count = 0
        for w_file in hist_dir.glob("weekly_*.json"):
            try:
                with open(w_file) as f:
                    doc = json.load(f)
                stats_data = doc.get("stats", {})
                models_data = doc.get("models", {})
                if stats_data:
                    db.upsert_weekly_history(city, stats_data, models_data)
                    count += 1
            except Exception as e:
                print(f"Error loading {w_file}: {e}")
        print(f"[{city}] Migrated {count} weekly history documents into database.")

print("\n✅ Database migration completed successfully!")
