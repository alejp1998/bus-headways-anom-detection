#!/usr/bin/env python3
"""Weekly Model Retraining, Parameter Rotation, and Data Archival Engine.

Runs every Monday at 00:00 (or on-demand via --run-now):
1. Ingests the past 7 days of bus telemetry for Madrid and London.
2. Vectorized computation of Gaussian parameters (mean, covariance, max_dim) per line, day type, and hour window.
3. Computes weekly QoS regularity, fleet stats, and anomaly metrics.
4. Atomically hot-swaps models_params.json for the live dashboard and anomaly workers.
5. Saves historical parameter summaries to Data/History/weekly_YYYY_WW.json.
6. Truncates raw weekly CSVs to reclaim disk space, keeping only compact distilled JSON summaries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent

# Day type mappings
DAY_TYPE_DICT = {
    "LA": [0, 1, 2, 3, 4],  # Weekdays (Mon-Fri)
    "SA": [5],  # Saturdays
    "FE": [6],  # Sundays & Holidays
}

HOUR_RANGES = [[7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]


def extract_ndim_windows(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Extract consecutive headway tuples of length `dim` from burst observations."""
    if df.empty or len(df) <= dim:
        return pd.DataFrame()

    hw_names = [f"hw{i}{i + 1}" for i in range(1, dim + 1)]
    bus_names = [f"bus{i}" for i in range(1, dim + 2)]

    records = []
    # Process direction by direction and burst by burst
    for (burst_time, _direction), group in df.groupby(["datetime", "direction"]):
        grp_sorted = group.sort_values("hw_pos")
        n = len(grp_sorted)
        if n < dim:
            continue

        for i in range(n - dim + 1):
            row_data = {"datetime": burst_time}
            row_data[bus_names[0]] = grp_sorted.iloc[i]["busA"]
            for k in range(dim):
                row_data[hw_names[k]] = grp_sorted.iloc[i + k]["headway"]
                row_data[bus_names[k + 1]] = grp_sorted.iloc[i + k]["busB"]
            records.append(row_data)

    return pd.DataFrame(records) if records else pd.DataFrame()


def estimate_gaussian_parameters(
    df: pd.DataFrame, dim: int
) -> tuple[float | list[list[float]], float | list[float]]:
    """Compute mean and covariance matrix (or variance/std for 1D)."""
    hw_names = [f"hw{i}{i + 1}" for i in range(1, dim + 1)]
    X = df[hw_names].to_numpy(dtype=float)

    if dim == 1:
        mean_val = float(np.mean(X[:, 0]))
        std_val = float(np.std(X[:, 0], ddof=1)) if len(X) > 1 else 1.0
        return std_val, mean_val
    else:
        mean_vec = [float(m) for m in np.mean(X, axis=0)]
        cov_mat = np.cov(X, rowvar=False)
        # Handle 1D return from cov when dim=1 or singular matrices
        if cov_mat.ndim == 0:
            cov_mat = np.array([[float(cov_mat)]])
        cov_list = [[float(val) for val in row] for row in cov_mat]
        return cov_list, mean_vec


def train_city_models(
    city: str,
    raw_df: pd.DataFrame,
    lines_dict: dict,
    min_samples: int = 15,
) -> tuple[dict, dict]:
    """Train Gaussian headway models for a city and return updated parameters + weekly stats."""
    models_dict = {}
    city_stats = {
        "city": city,
        "total_records": len(raw_df),
        "lines": {},
        "fleet_size": int(raw_df["bus"].nunique()) if "bus" in raw_df.columns else 0,
    }

    lines = list(lines_dict.keys())
    day_types = ["LA", "SA", "FE"]

    # Ensure datetime is parsed
    if not pd.api.types.is_datetime64_any_dtype(raw_df["datetime"]):
        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])

    raw_df["weekday"] = raw_df["datetime"].dt.weekday
    raw_df["hour"] = raw_df["datetime"].dt.hour

    for line in lines:
        line_str = str(line)
        line_data = raw_df[raw_df["line"].astype(str) == line_str]
        models_dict[line_str] = {}

        line_stats = {
            "total_bursts": len(line_data),
            "mean_headway": float(line_data["headway"].mean())
            if "headway" in line_data.columns and not line_data.empty
            else 0.0,
            "std_headway": float(line_data["headway"].std())
            if "headway" in line_data.columns and not line_data.empty
            else 0.0,
            "hourly_baselines": {},
        }

        for day_type in day_types:
            models_dict[line_str][day_type] = {}
            valid_days = DAY_TYPE_DICT[day_type]
            day_data = line_data[line_data["weekday"].isin(valid_days)]

            for h_start, h_end in HOUR_RANGES:
                hr_key = f"{h_start}-{h_end}"
                models_dict[line_str][day_type][hr_key] = {}

                split_df = day_data[(day_data["hour"] >= h_start) & (day_data["hour"] < h_end)]

                if split_df.empty or "headway" not in split_df.columns:
                    # Fallback default baseline
                    models_dict[line_str][day_type][hr_key] = {
                        "max_dim": 1,
                        "1": {"cov_matrix": 120.0, "mean": 360.0},
                    }
                    continue

                # Outlier rejection (95% CI on 1D headway)
                hw_vals = split_df["headway"].to_numpy(dtype=float)
                mean_hw = np.mean(hw_vals)
                std_hw = np.std(hw_vals)
                if std_hw > 0:
                    ci_low, ci_high = stats.norm.interval(0.95, loc=mean_hw, scale=std_hw)
                    clean_split = split_df[
                        (split_df["headway"] >= ci_low) & (split_df["headway"] <= ci_high)
                    ]
                else:
                    clean_split = split_df

                # Fit multi-dimensional models (d=1, 2, 3)
                max_dim = 1
                for d in range(1, 4):
                    win_df = extract_ndim_windows(clean_split, d)
                    if len(win_df) >= min_samples:
                        cov, mean = estimate_gaussian_parameters(win_df, d)
                        models_dict[line_str][day_type][hr_key][str(d)] = {
                            "cov_matrix": cov,
                            "mean": mean,
                        }
                        max_dim = d
                    elif d == 1:
                        # Fallback 1D if too few samples
                        models_dict[line_str][day_type][hr_key]["1"] = {
                            "cov_matrix": float(std_hw) if std_hw > 0 else 60.0,
                            "mean": float(mean_hw) if mean_hw > 0 else 360.0,
                        }

                models_dict[line_str][day_type][hr_key]["max_dim"] = max_dim

                if day_type == "LA":
                    line_stats["hourly_baselines"][hr_key] = {
                        "mean": float(mean_hw),
                        "std": float(std_hw),
                        "max_dim": max_dim,
                    }

        city_stats["lines"][line_str] = line_stats

    return models_dict, city_stats


def rotate_and_archive_city(
    city: str,
    now: dt.datetime | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute the full weekly rotation cycle for Madrid or London."""
    if now is None:
        now = dt.datetime.now()

    year, week_num, _ = now.isocalendar()
    week_id = f"{year}_W{week_num:02d}"

    city_dir = ROOT_DIR / city
    data_dir = city_dir / "Data"
    history_dir = data_dir / "History"
    anom_dir = data_dir / "Anomalies"
    realtime_dir = data_dir / "RealTime"
    static_dir = data_dir / "Static"

    history_dir.mkdir(parents=True, exist_ok=True)
    anom_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load static lines dict
    lines_file = static_dir / "lines_dict.json"
    with open(lines_file) as f:
        lines_dict = json.load(f)

    # 2. Ingest past week data (or week buffer if present, otherwise cleaned week sample)
    week_csv = realtime_dir / "buses_data_week_cleaned.csv"
    hws_csv = realtime_dir / "headways_burst.csv"

    if week_csv.exists() and os.path.getsize(week_csv) > 100:
        source_file = week_csv
    elif hws_csv.exists() and os.path.getsize(hws_csv) > 100:
        source_file = hws_csv
    else:
        print(f"[{city}] Warning: No active weekly telemetry found, using existing model baseline.")
        source_file = None

    if source_file is not None:
        raw_df = pd.read_csv(source_file, dtype={"line": "str"})
        raw_bytes = os.path.getsize(source_file)
    else:
        raw_df = pd.DataFrame()
        raw_bytes = 0

    # 3. Train models and generate statistics
    if not raw_df.empty:
        new_models, city_stats = train_city_models(city, raw_df, lines_dict)
    else:
        # Load existing models_params.json if available
        existing_params_file = anom_dir / "models_params.json"
        if existing_params_file.exists():
            with open(existing_params_file) as f:
                new_models = json.load(f)
        else:
            new_models = {}
        city_stats = {"city": city, "total_records": 0, "lines": {}, "fleet_size": 0}

    city_stats["week_id"] = week_id
    city_stats["timestamp"] = now.isoformat()
    city_stats["raw_data_bytes"] = raw_bytes
    city_stats["disk_space_saved_mb"] = round(raw_bytes / (1024 * 1024), 2)

    if dry_run:
        print(
            f"[{city}] Dry-run complete. Week: {week_id}, Processed: {city_stats['total_records']} rows."
        )
        return city_stats

    # 4. Atomic hot-swap of models_params.json
    models_file = anom_dir / "models_params.json"
    with tempfile.NamedTemporaryFile("w", dir=anom_dir, delete=False) as tmp:
        json.dump(new_models, tmp, indent=2)
        tmp_path = tmp.name
    shutil.move(tmp_path, models_file)
    print(f"[{city}] Updated live baseline: {models_file}")

    # 5. Save weekly historical record archive
    history_record_file = history_dir / f"weekly_{week_id}.json"
    with open(history_record_file, "w") as f:
        json.dump({"stats": city_stats, "models": new_models}, f, indent=2)
    print(f"[{city}] Archived historical summary: {history_record_file}")

    # 6. Update history index
    index_file = history_dir / "history_index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                history_index = json.load(f)
        except Exception:
            history_index = []
    else:
        history_index = []

    # Upsert current week entry
    history_index = [entry for entry in history_index if entry.get("week_id") != week_id]
    history_index.append(city_stats)
    history_index.sort(key=lambda x: x.get("week_id", ""), reverse=True)

    with open(index_file, "w") as f:
        json.dump(history_index, f, indent=2)

    # 6b. Update SQLite Database Weekly History
    try:
        sys.path.insert(0, str(ROOT_DIR))
        from core import db

        db.upsert_weekly_history(city, city_stats, new_models)
        db.prune_old_telemetry(days=7)
        print(f"[{city}] SQLite database updated with weekly history & pruned old telemetry.")
    except Exception as e:
        print(f"[{city}] Warning updating database: {e}")

    # 7. Truncate/reset raw weekly buffer to reclaim disk space
    if week_csv.exists() and source_file == week_csv:
        # Keep empty header-only CSV for the new week
        header = pd.read_csv(week_csv, nrows=0)
        header.to_csv(week_csv, index=False)
        print(
            f"[{city}] Reclaimed {city_stats['disk_space_saved_mb']} MB disk space. Buffer reset for new week."
        )

    return city_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Weekly model retraining and data rotation orchestrator"
    )
    parser.add_argument(
        "--run-now", action="store_true", help="Force immediate execution of weekly rotation"
    )
    parser.add_argument(
        "--city", choices=["all", "madrid", "london"], default="all", help="Target city"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform computation without writing/deleting"
    )
    args = parser.parse_args()

    now = dt.datetime.now()
    print("\n========================================================")
    print(" Transit Model Weekly Rotation & Archival Engine")
    print(
        f" Timestamp: {now.isoformat()} (ISO Week: {now.isocalendar()[0]}_W{now.isocalendar()[1]:02d})"
    )
    print("========================================================\n")

    cities = ["Madrid", "London"] if args.city == "all" else [args.city.capitalize()]

    for city in cities:
        print(f"\n--- Running rotation for {city} ---")
        stats_out = rotate_and_archive_city(city, now=now, dry_run=args.dry_run)
        print(
            f"✅ {city} finished. Total records: {stats_out.get('total_records', 0)}, Fleet: {stats_out.get('fleet_size', 0)}"
        )

    print("\n✅ Weekly rotation and parameter hot-swap completed successfully.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
