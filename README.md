# MindCare Oracle

MindCare Oracle is an end-to-end AI application for predicting mental health treatment risk using survey-based features. The project is designed as a complete MLOps implementation with automated data preprocessing, model training, experiment tracking, API serving, monitoring, and orchestration.

## Screen Cast Link
https://drive.google.com/file/d/1bs8-N3KYzixOpWSxkEDIZhlAUcn8VXWS/view?usp=sharing

## Problem Statement

Given self-reported demographic and workplace-survey attributes, predict whether an individual is likely to require or seek mental health treatment.

The objective is to provide a practical risk-indication system that is:
- usable by non-technical users through a web interface,
- reproducible for engineering and academic evaluation,
- observable in production-like operation through monitoring tools.

## Key Features

- React frontend for structured user input and prediction display.
- FastAPI backend with endpoints for inference and system checks.
- Reproducible preprocessing and training pipelines (`src/data_preprocessing.py`, `src/train.py`).
- MLflow experiment tracking for parameters, metrics, and artifacts.
- DVC pipeline definitions for process reproducibility.
- Airflow DAG for orchestration of preprocessing and training stages.
- Prometheus metrics exporter and Grafana dashboards for observability.
- Docker Compose-based multi-service deployment.

## Architecture Overview

Main subsystems:
- Frontend (`frontend/`): React + Vite UI
- API (`api/`): FastAPI inference service
- ML Pipeline (`src/`): preprocessing, training, drift logic
- Orchestration (`airflow/`): DAG-based workflow scheduling
- Experiment Tracking (`mlruns/`, MLflow server)
- Monitoring (`prometheus/`, `grafana/`)

Data flow summary:
1. Raw data is cleaned and transformed.
2. Model is trained and evaluated.
3. Artifacts and metrics are logged to MLflow.
4. FastAPI loads trained model and serves predictions.
5. Prometheus scrapes API metrics; Grafana visualizes them.

## Repository Structure

`api/` - backend service, API tests, container config  
`frontend/` - UI application and frontend container config  
`src/` - preprocessing, training, drift detection modules  
`tests/` - model and preprocessing unit tests  
`airflow/dags/` - orchestration DAG definitions  
`prometheus/` - scrape and alert configuration  
`grafana/provisioning/` - datasource and dashboard provisioning  
`docs/` - architecture, HLD, LLD, test plan, user manual, final report  
`dvc.yaml` - DVC stage definitions  
`MLproject` - MLflow project entrypoint definition  
`docker-compose.yml` - local multi-service deployment definition  
`start.sh` / `stop.sh` - convenience scripts for lifecycle management

## API Endpoints

- `GET /health` - service liveness check
- `GET /ready` - model readiness check
- `GET /metrics` - Prometheus metrics endpoint
- `POST /predict` - inference endpoint
- `POST /drift/check` - drift scoring on incoming records

Primary API base URL in local deployment:
- `http://localhost:8001`

## Local Setup and Run

### Prerequisites

- Docker Desktop (or Docker Engine + Compose)
- Python 3.10+ (for local scripts and MLflow startup helper)
- Git

### Quick Start

1. Clone the repository.
2. From repository root, run:
   - `chmod +x start.sh stop.sh`
   - `./start.sh`
3. Open services:
   - Frontend: `http://localhost:3002`
   - API docs: `http://localhost:8001/docs`
   - MLflow: `http://localhost:5001`
   - Grafana: `http://localhost:3001` (admin/admin)
   - Prometheus: `http://localhost:9090`
   - Airflow: `http://localhost:8082` (admin/admin)

To stop:
- `./stop.sh`

## Reproducibility Workflow

- Data and pipeline stage definitions are codified in `dvc.yaml`.
- Training command and environment assumptions are defined in `MLproject` and `conda.yaml`.
- Model runs are tracked in MLflow with parameter and metric metadata.
- Docker Compose provides environment parity across local runs.

## Testing

Test assets include:
- `tests/test_preprocessing.py`
- `tests/test_model.py`
- `api/test_api.py`

Run tests (if environment has pytest installed):
- `pytest -q`

## Monitoring and Drift

- API exports request count, latency, and error metrics.
- Prediction and drift metrics are published for near-real-time monitoring.
- Prometheus collects metrics and Grafana dashboards visualize system health and trends.

## Documentation

Comprehensive project documentation is available in `docs/`:
- `Architecture.md`
- `HLD.md`
- `LLD.md`
- `TestPlan.md`
- `UserManual.md`
- `Final_Project_Report.txt`

## Academic and Submission Context

This project was built to align with AI application and MLOps evaluation guidelines emphasizing:
- automation,
- reproducibility,
- experiment tracking,
- observability,
- modular deployment,
- documentation quality.

## License

This repository is provided for academic/project use. Add a formal license file if distribution terms are required.

