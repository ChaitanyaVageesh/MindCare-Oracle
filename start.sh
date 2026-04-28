#!/bin/bash
set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${GREEN}=== MindCare Oracle Startup ===${NC}"

# ── 1. Kill any old MLflow on port 5001 ───────────────────────────────────────
pid=$(lsof -ti tcp:5001 2>/dev/null || true)
[ -n "$pid" ] && kill -9 $pid 2>/dev/null && echo "  Freed port 5001" || true
sleep 1

# ── 2. Start MLflow natively (avoids Docker pip-install timeout) ──────────────
echo -e "${YELLOW}Starting MLflow on :5001 ...${NC}"
if [ ! -f ".venv/bin/mlflow" ]; then
  echo -e "${RED}  ERROR: .venv/bin/mlflow not found.${NC}"
  echo "  Run: python3 -m venv .venv && .venv/bin/pip install mlflow"
  exit 1
fi
.venv/bin/mlflow server \
  --backend-store-uri "file://$(pwd)/mlruns" \
  --default-artifact-root "$(pwd)/mlruns" \
  --host 0.0.0.0 --port 5001 \
  > /tmp/mindcare_mlflow.log 2>&1 &
echo $! > /tmp/mindcare_mlflow.pid
echo "  PID $(cat /tmp/mindcare_mlflow.pid) — logs at /tmp/mindcare_mlflow.log"
sleep 2

# ── 3. Build & start Docker services ──────────────────────────────────────────
echo -e "${YELLOW}Building Docker images (api + frontend) ...${NC}"
docker compose build api frontend 2>&1 | grep -E "Built|ERROR|error" || true

echo -e "${YELLOW}Starting all Docker services ...${NC}"
docker compose up -d

# ── 4. Wait for API ────────────────────────────────────────────────────────────
echo -n "  Waiting for API on :8001"
for i in $(seq 1 30); do
  curl -sf http://localhost:8001/health > /dev/null 2>&1 && echo " ✓" && break
  echo -n "."; sleep 2
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  All MindCare services running${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Frontend        →  http://localhost:3000"
echo "  FastAPI Swagger →  http://localhost:8001/docs"
echo "  MLflow UI       →  http://localhost:5001"
echo "  Grafana         →  http://localhost:3001   (admin / admin)"
echo "  Prometheus      →  http://localhost:9090"
echo "  Airflow         →  http://localhost:8082   (admin / admin)"
echo ""
echo "  To stop: ./stop.sh"
