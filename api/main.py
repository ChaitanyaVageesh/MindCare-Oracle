import glob
import logging
import os
import pickle
import sys
import time
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mindcare.api")

app = FastAPI(
    title="MindCare Oracle API",
    description="Mental Health Prediction API",
    version="1.0",
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "http_status"]
)
REQUEST_ERROR_COUNT = Counter(
    "api_errors_total", "Total failed API requests", ["method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Request latency in seconds", ["method", "endpoint"]
)
PREDICTION_COUNT = Counter(
    "model_predictions_total", "Total model predictions", ["prediction_result"]
)
DRIFT_SCORE = Gauge(
    "model_data_drift_score", "Drift score per feature", ["feature"]
)

model = None
_baseline: dict = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PatientData(BaseModel):
    Age: int = Field(..., description="Age of the patient", examples=[30])
    Gender: str = Field(..., description="male | female | trans", examples=["male"])
    family_history: str = Field(..., description="Yes | No", examples=["Yes"])
    benefits: str = Field(..., description="Yes | No | Don't know", examples=["Yes"])
    care_options: str = Field(..., description="Yes | No | Not sure", examples=["Not sure"])
    anonymity: str = Field(..., description="Yes | No | Don't know", examples=["Yes"])
    leave: str = Field(
        ...,
        description="Very easy | Somewhat easy | Don't know | Somewhat difficult | Very difficult",
        examples=["Somewhat easy"],
    )
    work_interfere: str = Field(
        ...,
        description="Often | Rarely | Never | Sometimes | Don't know",
        examples=["Sometimes"],
    )


class PredictionResponse(BaseModel):
    treatment: bool
    probability: float


class DriftRequest(BaseModel):
    records: List[PatientData] = Field(..., description="Recent inputs to check for drift")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    _load_model()
    _load_baseline()


def _load_model():
    global model

    # Allow overriding the exact pkl path via env var
    pkl_path = os.environ.get("MODEL_PKL_PATH") or None

    if not pkl_path:
        for root in ["/app/mlruns", "./mlruns"]:
            matches = glob.glob(f"{root}/**/model.pkl", recursive=True)
            if matches:
                pkl_path = matches[0]
                break

    if not pkl_path:
        logger.error("No model.pkl found — make sure mlruns/ is mounted at /app/mlruns")
        return

    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from %s", pkl_path)
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)


def _load_baseline():
    global _baseline
    try:
        from src.drift_detection import load_baseline
        _baseline = load_baseline()
        logger.info("Drift baseline loaded with %d features", len(_baseline))
    except Exception as exc:
        logger.warning("Could not load drift baseline: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    start = time.time()
    if model is None:
        REQUEST_ERROR_COUNT.labels(method="POST", endpoint="/predict").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        elapsed = time.time() - start
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status="200").inc()
        PREDICTION_COUNT.labels(prediction_result=str(prediction)).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict").observe(elapsed)

        logger.info("Prediction: %d (prob=%.3f) in %.3fs", prediction, probability, elapsed)
        return {"treatment": bool(prediction), "probability": probability}

    except Exception as exc:
        REQUEST_ERROR_COUNT.labels(method="POST", endpoint="/predict").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status="500").inc()
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/drift/check")
def drift_check(req: DriftRequest):
    if not _baseline:
        raise HTTPException(status_code=503, detail="Drift baseline not loaded")

    try:
        from src.drift_detection import check_drift
        records = [r.model_dump() for r in req.records]
        result = check_drift(records, _baseline)

        for feature, stats in result.get("features", {}).items():
            score = stats.get("drift_score") or stats.get("psi") or 0.0
            DRIFT_SCORE.labels(feature=feature).set(score)

        logger.info("Drift check: %d records, drifted=%s", result["n_records"], result["is_drifted"])
        return result

    except Exception as exc:
        logger.exception("Drift check failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
