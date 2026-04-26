# High Level Design (HLD)

## Project Overview
The MindCare Oracle is an end-to-end Machine Learning web application designed to predict the likelihood of an individual seeking mental health treatment based on survey inputs. It incorporates MLOps methodologies to ensure automated reproduction, versioning, monitoring, and robust serving.

## Design Principles
- **Loose Coupling**: Frontend and Backend interact strictly via REST API.
- **Reproducibility**: DVC controls data versions and orchestrates ML pipelines; MLflow handles experiment tracking and parameter/metric management.
- **Observability**: Prometheus captures NRT inference metrics and application status from the `/metrics` endpoint; Grafana visualizes metrics.

## System Architecture Overview
The system employs a multi-container Docker Compose setup containing:
1. **Frontend App**: React application built with Vite providing a rich glassmorphism UI.
2. **Backend API**: A FastAPI server running `uvicorn`. Connects to cached MLflow model artifacts.
3. **Tracking Server**: A local containerized MLflow server running natively.
4. **Monitoring Stack**: Prometheus and Grafana connected in tandem to scrape FastAPI throughputs.

## Key Subsystems
- **Data Versioning Stack**: Tracks `survey.csv` changes over git.
- **Model Training Pipeline**: Handles null imputations, bounds scaling, and Label Encoding dynamically through an `sklearn.compose.ColumnTransformer`, bundled with an `AdaBoostClassifier`.
- **Instrumentation**: Exposes custom counters mapping Predictions (`True` vs `False`), total request durations, and failures to assist with Drift detection.
