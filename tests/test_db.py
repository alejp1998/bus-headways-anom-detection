"""Unit tests for high-performance SQLite database engine (core/db.py)."""

from pathlib import Path

import pandas as pd
import pytest

from core import db


@pytest.fixture(autouse=True)
def setup_database(tmp_path):
    """Isolate tests from the production database via a temporary SQLite file."""
    db.DB_DIR = Path(tmp_path)
    db.DB_PATH = db.DB_DIR / "test_telemetry.db"
    db.init_db()


def test_insert_and_query_buses():
    """Test live vehicle positions insertion and latest snapshot retrieval."""
    test_df = pd.DataFrame(
        [
            {
                "line": "24",
                "bus": "9991",
                "vehicleId": "TEST24A",
                "destination": "Pimlico",
                "stop": "490000001A",
                "estimateArrive": 120,
                "DistanceBus": 600.0,
                "lat": 51.5074,
                "lon": -0.1278,
                "datetime": "2026-08-17T00:00:00",
            },
            {
                "line": "24",
                "bus": "9992",
                "vehicleId": "TEST24B",
                "destination": "Pimlico",
                "stop": "490000002B",
                "estimateArrive": 480,
                "DistanceBus": 2400.0,
                "lat": 51.5174,
                "lon": -0.1378,
                "datetime": "2026-08-17T00:00:00",
            },
        ]
    )

    db.insert_buses_burst("London", test_df)
    res = db.get_latest_burst_df("London", "24")

    assert not res.empty
    assert len(res) >= 2
    assert "estimateArrive" in res.columns
    assert "DistanceBus" in res.columns


def test_insert_and_query_headways():
    """Test consecutive headway spacing insertion and retrieval."""
    hws_df = pd.DataFrame(
        [
            {
                "line": "73",
                "direction": 1,
                "busA": "9993",
                "busB": "9994",
                "hw_pos": 1,
                "headway": 360.0,
                "busA_ttls": 120.0,
                "busB_ttls": 480.0,
                "datetime": "2026-08-17T00:00:00",
            }
        ]
    )

    db.insert_headways_burst("London", hws_df)
    res = db.get_latest_headways_df("London", "73")

    assert not res.empty
    assert float(res.iloc[0]["headway"]) == 360.0
    assert str(res.iloc[0]["busA"]) == "9993"


def test_insert_and_query_series_and_anomalies():
    """Test multi-dimensional series and anomaly query functions."""
    series_df = pd.DataFrame(
        [
            {
                "line": "18",
                "dim": 1,
                "m_dist": 2.45,
                "anom": 1,
                "bus1": "3001",
                "bus2": "3002",
                "bus3": "0",
                "bus4": "0",
                "bus5": "0",
                "bus6": "0",
                "bus7": "0",
                "bus8": "0",
                "bus9": "0",
                "hw12": 650.0,
                "hw23": 0.0,
                "hw34": 0.0,
                "hw45": 0.0,
                "hw56": 0.0,
                "hw67": 0.0,
                "hw78": 0.0,
                "hw89": 0.0,
                "datetime": "2026-08-17T00:00:00",
            }
        ]
    )
    db.insert_headways_series("London", series_df)
    res_series = db.get_series_df("London", "18", dim=1, limit=10)
    assert not res_series.empty
    assert any(float(r.m_dist) == 2.45 for r in res_series.itertuples())

    anom_df = pd.DataFrame(
        [
            {
                "line": "18",
                "dim": 1,
                "m_dist": 3.12,
                "anom_size": 2,
                "bus1": "3001",
                "bus2": "3002",
                "bus3": "0",
                "bus4": "0",
                "hw12": 720.0,
                "hw23": 0.0,
                "hw34": 0.0,
                "datetime": "2026-08-17T00:00:00",
            }
        ]
    )
    db.insert_anomaly_events("London", anom_df)
    res_anoms = db.get_anomalies_df("London", "18", limit=10)
    assert not res_anoms.empty
    assert any(float(r.m_dist) == 3.12 for r in res_anoms.itertuples())


def test_weekly_history_upsert_and_retrieval():
    """Test weekly aggregated history records."""
    stats = {
        "week_id": "2026_W34",
        "total_records": 12500,
        "fleet_size": 84,
        "overall_qos": 94.5,
        "disk_space_saved_mb": 1.25,
        "api_success_rate": 99.8,
    }
    models = {"24": {"LA": {"17-19": {"1": {"mean": 340.0, "cov_matrix": 80.0}}}}}

    db.upsert_weekly_history("London", stats, models)
    hist = db.get_all_weekly_history("London")
    assert any(h.get("week_id") == "2026_W34" for h in hist)

    single = db.get_single_week_data("London", "2026_W34")
    assert single.get("stats", {}).get("fleet_size") == 84
