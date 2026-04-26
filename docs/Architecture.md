# Application Architecture Diagram

```mermaid
flowchart TD
    %% Define Nodes
    subgraph Client Application
        UI["Web App Frontend UI (React + CSS)"]
    end

    subgraph Monitoring Stack
        Prometheus["Prometheus Metrics Server"]
        Grafana["Grafana Visualizer"]
    end

    subgraph Application Stack
        API["FastAPI Inference Server"]
        MLflow["MLflow Model Artifacts"]
    end

    subgraph Data & Pipeline Stack
        DVC["DVC Pipeline Orchestrator"]
        Data["CSV Survey Data"]
        Process["Python Data Processor"]
        Train["Scikit-Learn Training Pipeline"]
    end

    %% Connections
    UI -- "REST: /predict" --> API
    API -- "In-Memory Load" --> MLflow
    API -- "Exposes /metrics" --> Prometheus
    Prometheus -- "Scrapes Data" --> API
    Grafana -- "Queries Data" --> Prometheus
    
    Data --> Process
    Process --> Train
    Train -- "Logs Artifacts" --> MLflow
    DVC -- "Triggers" --> Process
    DVC -- "Triggers" --> Train
```

## Description of Blocks

- **Web App Frontend UI**: Highly aesthetic, glassmorphic UI avoiding standard Tailwind limits, routing variables to the backend transparently. 
- **FastAPI Inference Server**: Loads localized `.pkl` models via MLflow libraries matching the DVC output and executes prediction schemas asynchronously.
- **Monitoring Stack**: Prometheus polls the FastAPI endpoint exposing error occurrences and prediction outcomes recursively. Grafana links sequentially as an aggregator.
- **DVC/Data Engine Stack**: Handles pipeline reproduction without redundant retraining schedules, strictly logging parameter hashes against code outputs.
