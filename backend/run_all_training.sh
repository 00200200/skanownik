#!/bin/bash
set -e

echo "Starting full training pipeline..."

echo "----------------------------------------------------------------"
echo "1. Training Vehicle Classifier (src/train.py)"
echo "----------------------------------------------------------------"
uv run src/train.py

echo "----------------------------------------------------------------"
echo "2. Training Anomaly Detector - Dostawcze (src/train_anomaly_dostawcze.py)"
echo "----------------------------------------------------------------"
uv run src/train_anomaly_dostawcze.py

echo "----------------------------------------------------------------"
echo "3. Training Anomaly Detector - Osobowe (src/train_anomaly_osobowe.py)"
echo "----------------------------------------------------------------"
uv run src/train_anomaly_osobowe.py

echo "----------------------------------------------------------------"
echo "All trainings completed successfully!"
echo "----------------------------------------------------------------"
