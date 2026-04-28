#!/bin/bash
cd "$(dirname "$0")"
echo "Stopping Docker services..."
docker compose down
echo "Stopping MLflow..."
[ -f /tmp/mindcare_mlflow.pid ] && kill $(cat /tmp/mindcare_mlflow.pid) 2>/dev/null; rm -f /tmp/mindcare_mlflow.pid
echo "All MindCare services stopped."
