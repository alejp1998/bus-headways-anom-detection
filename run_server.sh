#!/bin/bash
trap "kill 0" EXIT

# MADRID
# Enter Anomalies Detection Scripts Folder
cd Madrid/Scripts/AnomaliesDetection

# Process models parameters using headways data
#python3 models_params.py

# Start anomalies detection script
python3 detect_anoms_hws.py &


# LONDON
# Enter Anomalies Detection Scripts Folder
cd ../../../London/Scripts/AnomaliesDetection

# Process models parameters using headways data
#python3 models_params.py

# Start anomalies detection script
python3 detect_anoms_hws.py &

# Enter Dashboard Folder
cd ../../../Dashboard

# Run the Dashboard
python3 index.py &


wait
