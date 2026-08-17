#!/usr/bin/env bash
# Weekly rotation & archival orchestrator — invoked by cron every Monday 00:00
set -euo pipefail
cd /home/alejp1998/dev/theses/bus-headways-anom-detection
.venv/bin/python scripts/weekly_rotation.py --run-now >> /tmp/weekly_rotation.log 2>&1
