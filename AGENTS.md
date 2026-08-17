# AGENTS.md - Bus Headways Analysis & Anomaly Detection

This repository contains the Bachelor Thesis and IEEE TITS research codebase for real-time bus headway statistical modeling, Quality of Service (QoS) estimation, and unsupervised anomaly detection for Madrid (EMT) and London (TfL) public bus networks.

---

## Project Overview

- **Primary Language**: Python (3.11+)
- **Core Frameworks**: Dash 4.x, Flask 3.x, Plotly 6.x, Pandas 3.x, NumPy 2.x, SciPy
- **Main Entry Points**:
  - Web Dashboard: `Dashboard/index.py` (serves Dash UI on port `8050`)
  - Server Launcher: `run_server.sh` (spawns anomaly detectors and web UI)
  - Data Collection: `Madrid/Scripts/CollectData/retrieve_data.py` / `London/Scripts/CollectData/retrieve_data.py`
  - Anomaly Detection: `Madrid/Scripts/AnomaliesDetection/detect_anoms_hws.py` / `London/Scripts/AnomaliesDetection/detect_anoms_hws.py`
- **Architecture**:
  - Ingestion: Live API polling (Madrid EMT MobilityLabs & London TfL)
  - Processing: Trajectory cleaning, stop arrival time extraction, travel time calculation, and headway computation
  - Analytics & Anomaly Detection: Empirical parameter fitting and statistical thresholding for headway irregularity detection
  - Visualization: Multi-page Dash application displaying real-time vehicle positions, headways, and anomaly alerts

---

## Coding Agent Workflow

1. **Environment**: Use Python 3.11 via the provided `.venv` (`uv venv --python 3.11 .venv && uv pip install -r requirements.txt`). The modern stack (Dash 4 / Plotly 6 / Pandas 3) is required — legacy `dash_core_components` imports no longer exist.
2. **Data Directory Integrity**: Telemetry and processed datasets reside in `Madrid/Data/` and `London/Data/`. Do not overwrite or delete static stop maps or processed CSV caches unless explicitly executing a re-processing pipeline.
3. **Preserve Notebooks**: The research notebooks in `Madrid/Notebooks/` and `London/Notebooks/` contain experimental figures and publication derivations. Do not modify notebook outputs without explicit instructions.
4. **API Credentials**: Never hardcode EMT/TfL API tokens into tracked scripts. Use `api_credentials.py` (gitignored) or environment variables (`MAPBOX_ACCESS_TOKEN`) for API authentication. GitHub secret scanning blocks pushes containing secrets.
5. **Paths**: Dashboard modules resolve data via `resolve_path()` relative to the repo root — run the dashboard from anywhere.
6. **Validation**: Test dashboard layout changes by running `Dashboard/index.py`, then check `http://localhost:8050`. Run the test suite with `pytest` and quality gates with `pre-commit run --all-files`.

---

## Quality Gates

The following checks are enforced by pre-commit (installed via `pre-commit install`):

| Gate       | Tool                | Command                                    | Threshold                                                                  |
| ---------- | ------------------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| Lint       | Ruff                | `ruff check .`                             | Zero findings (auto-fix with `ruff check --fix .`)                         |
| Format     | Ruff format         | `ruff format .`                            | Black-compatible, line length 100                                          |
| Complexity | Radon (gate script) | `python scripts/check_radon_complexity.py` | Blocks D/E/F in `Dashboard/` + `tests/`                                    |
| Complexity | Ruff McCabe (C901)  | part of `ruff check .`                     | max 15; legacy `Scripts/` excluded (research code)                         |
| Prettier   | prettier            | pre-commit mirror                          | CSS/YAML/JSON/Markdown formatting                                          |
| Generic    | pre-commit-hooks    | pre-commit                                 | trailing whitespace, EOF, JSON/YAML validity, merge conflicts, large files |
| Tests      | pytest              | `pytest`                                   | All tests pass (`tests/test_dashboard.py`)                                 |

Run everything at once:

```bash
pre-commit run --all-files
pytest
```

### Legacy Research Scripts

`Madrid/Scripts/` and `London/Scripts/` contain the original research pipelines (`get_headways` / `process_day_df` have cyclomatic complexity 60+). They are linted for style (E/W/F/I) but exempt from the McCabe C901 gate — refactoring them risks changing published research outputs. The Radon gate covers the application code (`Dashboard/`) where quality is actively maintained.

---

## Quick Start

### Running with Docker (Recommended)

```bash
# Build Docker image
docker build -t bus-headways:latest .

# Run container
docker run -d -p 8050:8050 --name bus-headways bus-headways:latest
```

The dashboard will be live at `http://localhost:8050`.

### Running Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch anomaly detection background workers and Dashboard
bash run_server.sh
```

---

## Project Structure

### Key Directories

| Directory         | Purpose                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `Dashboard/`      | Interactive Dash/Flask web application                                                    |
| `Dashboard/apps/` | Page components (`app_home`, `app_realtime_madrid`, `app_realtime_london`, `app_credits`) |
| `Madrid/`         | Madrid EMT pipeline: data, exploratory notebooks, and processing scripts                  |
| `London/`         | London TfL pipeline: data, distribution modeling notebooks, and anomaly scripts           |
| `Madrid/Scripts/` | Data collection, queue processing, and anomaly detection scripts                          |
| `London/Scripts/` | TfL data ingestion, preprocessing, and headway anomaly detection                          |

### Key Modules & Files

| File                                                    | Purpose                                                                     |
| ------------------------------------------------------- | --------------------------------------------------------------------------- |
| `run_server.sh`                                         | Orchestration script launching Madrid/London anomaly workers and web server |
| `Dockerfile`                                            | Container build configuration exposing port 8050                            |
| `requirements.txt`                                      | Pinned Python dependencies                                                  |
| `Dashboard/app.py`                                      | Dash app configuration, Bulma CSS styling, and server instance              |
| `Dashboard/index.py`                                    | Main URL routing, header navigation, and layout assembly                    |
| `Madrid/Scripts/AnomaliesDetection/detect_anoms_hws.py` | Real-time Madrid headway anomaly evaluation worker                          |
| `London/Scripts/AnomaliesDetection/detect_anoms_hws.py` | Real-time London headway anomaly evaluation worker                          |
| `Madrid/Scripts/CollectData/retrieve_data.py`           | Live EMT Madrid API ingestion worker                                        |

---

## Deployment & Service Management

For persistent production deployment on Linux hosts, refer to `service_commands.md` for Systemd unit file configurations for the background anomaly detection workers and the Dash web server.
