# Bus Headways Analysis & Anomaly Detection

**Real-Time Bus Headway Modeling, Quality of Service (QoS) Estimation, and Anomaly Detection for Madrid (EMT) and London (TfL)**

[![Python](https://img.shields.io/badge/Python-3.7-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-1.12-008DE4?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Flask](https://img.shields.io/badge/Flask-1.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![IEEE TITS](https://img.shields.io/badge/IEEE%20TITS-10.1109%2FTITS.2022.3155180-00629B)](https://doi.org/10.1109/TITS.2022.3155180)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Publication Reference

This repository contains the official implementation and research codebase for the publication:

> **A. Jarabo-Peñas, P. J. Zufiria and C. García-Mauriño**, *"Bus Headways Analysis for Anomaly Detection,"* in **IEEE Transactions on Intelligent Transportation Systems**, vol. 23, no. 10, pp. 18975-18988, Oct. 2022.
> **DOI:** [10.1109/TITS.2022.3155180](https://doi.org/10.1109/TITS.2022.3155180)

---

## Contents

- [Overview](#overview)
- [Key Technical Highlights](#key-technical-highlights)
- [System Architecture](#system-architecture)
  - [Data Pipeline](#data-pipeline)
  - [Statistical Modeling & QoS](#statistical-modeling--qos)
  - [Real-Time Anomaly Detection](#real-time-anomaly-detection)
- [Web Dashboard](#web-dashboard)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Docker Deployment (Recommended)](#docker-deployment-recommended)
  - [Local Python Setup](#local-python-setup)
- [Data Access & Ingestion](#data-access--ingestion)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Maintaining regular headways (the time interval between consecutive transit vehicles) is fundamental to operating high-quality urban bus systems. Headway irregularities lead to bus bunching, increased passenger wait times, and degraded service reliability.

This project delivers an end-to-end data processing, statistical modeling, and real-time monitoring platform for public bus networks in **Madrid (EMT)** and **London (TfL)**. Using live telemetry and arrival time data from municipal transport APIs, the system:
1. Models the statistical distribution and spatiotemporal evolution of headways across line stops.
2. Derives an unsupervised **Quality of Service (QoS)** index to quantify route performance.
3. Automatically detects headway anomalies (bunching, excessive gaps, missing vehicles) in real time.
4. Serves live interactive visualizations via a web-based dashboard.

---

## Key Technical Highlights

* **Multi-City Real-Time Ingestion**: Automated collectors interfacing with Madrid EMT MobilityLabs API and London Transport for London (TfL) Unified API.
* **Empirical & Parametric Headway Characterization**: Comprehensive statistical modeling of headways and inter-stop travel times across different times of day, line typologies, and stop sequences.
* **Online Unsupervised Anomaly Detection**: Statistical anomaly detection scheme that dynamically flags irregular headway deviations without requiring labeled historical training sets.
* **QoS Formulation**: Quantitative metric assessing deviation from scheduled vs. actual headway regularity.
* **Interactive Dash/Flask Web UI**: Responsive multi-page web application featuring live route topology maps, headway timeline gauges, and anomaly alerts.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External Transit APIs                    │
│   Madrid EMT MobilityLabs API   │   London TfL Unified API  │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Collection Layer                     │
│           (retrieve_data.py / real-time polling)            │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Pipeline                  │
│  Data Cleaning ──► Arrival Times ──► Inter-Stop Times ──►   │
│                    Headway Computation                      │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Statistical Modeling & Anomaly Engine            │
│  Parameter Estimation (models_params.py)                    │
│  Live Anomaly Inference (detect_anoms_hws.py)               │
│  QoS Index Formulation                                      │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Interactive Dashboard (Dash/Flask)             │
│  Route Visualizer ──► Live Headways ──► Anomaly Indicators  │
│                   (http://localhost:8050)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Web Dashboard

The web dashboard is built using Dash, Flask, Plotly, and Bulma CSS. It provides:
- **Home View**: System overview, methodology summary, and line selection.
- **Madrid EMT Real-Time**: Live headway tracking and anomaly status across selected Madrid lines (Lines 1, 44, 82, 132, 133, F, G).
- **London TfL Real-Time**: Live headway metrics across monitored London bus corridors.
- **Credits & Publication**: Citation links, institutional context, and author details.

---

## Getting Started

### Prerequisites

- **Docker** (recommended) or **Python 3.7+**
- Git

### Docker Deployment (Recommended)

Build and run the containerized dashboard with a single command:

```bash
# Build the Docker image
docker build -t bus-headways:latest .

# Run the container
docker run -d -p 8050:8050 --name bus-headways bus-headways:latest
```

Access the dashboard in your browser at `http://localhost:8050`.

### Local Python Setup

```bash
# 1. Create a virtual environment (Python 3.7 - 3.10 recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch anomaly detectors and dashboard server
bash run_server.sh
```

---

## Data Access & Ingestion

### Pre-Processed Datasets

Pre-processed datasets containing cleaned telemetry, stop arrival times, and computed headways are archived on Mega:
- **Dataset Archive:** [Mega Storage Link](https://mega.nz/folder/QRIGnQRZ#7fJVQcapLkSp7jGGz0WZeQ)
- **Files included:**
  - `buses_data_cleaned.csv`: Cleaned trajectory logs
  - `arrival_times.csv`: Bus arrival timestamps per stop
  - `time_bt_stops.csv`: Inter-stop transit travel times
  - `headways.csv`: Computed headways per line and stop

Place the downloaded CSVs into the respective `Madrid/Data/Processed/` or `London/Data/Processed/` directories.

### Live API Configuration

To enable real-time polling for Madrid EMT:
1. Register for an API token at [Madrid EMT MobilityLabs](https://mobilitylabs.emtmadrid.es/).
2. Create an `api_credentials.py` file in the project root:
   ```python
   EMT_CLIENT_ID = "your_client_id"
   EMT_PASSKEY = "your_passkey"
   ```
3. Run the collection script:
   ```bash
   python Madrid/Scripts/CollectData/retrieve_data.py
   ```

---

## Repository Structure

```
bus-headways-anom-detection/
├── Dashboard/                  # Dash/Flask web application
│   ├── app.py                  # Dash instance and server initialization
│   ├── index.py                # Main navigation layout and routing
│   ├── apps/                   # Page modules (home, madrid, london, credits)
│   └── assets/                 # CSS styles and static assets
├── Madrid/                     # Madrid EMT data, scripts, and notebooks
│   ├── Data/                   # Static stop data, raw/processed telemetry
│   ├── Notebooks/              # Exploratory data analysis and QoS modeling
│   └── Scripts/                # Collection, processing, and anomaly scripts
├── London/                     # London TfL data, scripts, and notebooks
│   ├── Data/                   # TfL network configurations and telemetry
│   ├── Notebooks/              # Headway analysis and distribution fitting
│   └── Scripts/                # TfL data ingest and anomaly detection
├── Dockerfile                  # Container definition for dashboard deployment
├── requirements.txt            # Python package dependencies
├── run_server.sh               # Startup script for anomaly workers + web server
└── service_commands.md         # Systemd service templates and operations guide
```

---

## Citation

If you use this codebase or methodology in your research, please cite our IEEE TITS paper:

```bibtex
@article{jarabo2022bus,
  author={Jarabo-Pe{\~n}as, Alejandro and Zufiria, Pedro J. and Garc{\'\i}a-Mauri{\~n}o, Carlos},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={Bus Headways Analysis for Anomaly Detection},
  year={2022},
  volume={23},
  number={10},
  pages={18975-18988},
  doi={10.1109/TITS.2022.3155180}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
