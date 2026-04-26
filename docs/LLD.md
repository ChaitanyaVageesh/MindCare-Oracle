# Low Level Design (LLD)

## Project Name: MindCare Oracle

## Component 1: `PatientData` Schema (Data Model)
Using Pydantic validation:
- `Age` (Integer > 0): Patient reported age.
- `Gender` (String): Categorical enum (male, female, trans).
- `family_history` (String): Categorical.
- `benefits` (String): Categorical.
- `care_options` (String): Categorical.
- `anonymity` (String): Categorical.
- `leave` (String): Categorical.
- `work_interfere` (String): Categorical.

## Component 2: FastAPI Endpoints

### 1. `POST /predict`
**Description**: Primary inference gateway utilizing serialized scikit-learn Pipeline (AdaBoost).
- **Request Body**: `PatientData` JSON
- **Response**: `200 OK`
  ```json
  {
    "treatment": true,
    "probability": 0.81
  }
  ```
- **Error Exceptions**: Logs 500 error, exposes exception detail linearly.

### 2. `GET /metrics`
**Description**: Output sink for Prometheus Exporters.
- **Response Type**: `text/plain`
- **Output Properties**:
  - `api_requests_total`
  - `model_predictions_total`
  - `api_request_duration_seconds`
  - `api_errors_total`

### 3. `GET /health` & `GET /ready`
**Description**: Container verification pings for docker network status.
