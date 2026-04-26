from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import Response
import mlflow.sklearn
import pandas as pd
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import os

app = FastAPI(title="MindCare Oracle API", description="Mental Health Prediction API", version="1.0")

# Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total count of requests", ["method", "endpoint", "http_status"])
REQUEST_ERROR_COUNT = Counter("api_errors_total", "Total count of failed requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Request latency", ["method", "endpoint"])
PREDICTION_COUNT = Counter("model_predictions_total", "Total predictions made", ["prediction_result"])

# Load model globally on startup
model = None

class PatientData(BaseModel):
    Age: int = Field(..., description="Age of the patient", example=30)
    Gender: str = Field(..., description="Gender (male, female, trans)", example="male")
    family_history: str = Field(..., description="Family history of mental illness (Yes, No)", example="Yes")
    benefits: str = Field(..., description="Does employer provide benefits (Yes, No, Don't know)", example="Yes")
    care_options: str = Field(..., description="Do you know options for care? (Yes, No, Not sure)", example="Not sure")
    anonymity: str = Field(..., description="Is anonymity protected? (Yes, No, Don't know)", example="Yes")
    leave: str = Field(..., description="How easy is medical leave (Somewhat easy, Somewhat difficult, Don't know, Very easy, Very difficult)", example="Somewhat easy")
    work_interfere: str = Field(..., description="Does condition interfere with work? (Often, Rarely, Never, Sometimes, Don't know)", example="Sometimes")

class PredictionResponse(BaseModel):
    treatment: bool
    probability: float

@app.on_event("startup")
def load_model():
    global model
    model_uri = os.environ.get("MODEL_URI", None)
    if not model_uri:
        import glob
        # Try to find MLmodel inside /app/mlruns (for Docker) or ./mlruns (for local)
        search_path = "/app/mlruns/**/MLmodel" if os.path.exists("/app") else "./mlruns/**/MLmodel"
        mlmodels = glob.glob(search_path, recursive=True)
        if mlmodels:
            model_uri = "file://" + os.path.abspath(os.path.dirname(mlmodels[0]))
        else:
            print("No model found. Make sure mlruns directory is mounted or available.")
            return
            
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully from {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}. Model might not be available yet.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    if model is not None:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    start_time = time.time()
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        input_df = pd.DataFrame([data.dict()])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Log metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=200).inc()
        PREDICTION_COUNT.labels(prediction_result=str(prediction)).inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict").observe(time.time() - start_time)

        return {"treatment": bool(prediction), "probability": float(probability)}

    except Exception as e:
        REQUEST_ERROR_COUNT.labels(method="POST", endpoint="/predict").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))
