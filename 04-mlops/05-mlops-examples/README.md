# 5. End-to-End MLOps Example: Credit Risk Classification

## What You Will Build

This lesson puts everything from lessons 01-04 into practice. You will build a **complete MLOps pipeline** that includes:

- **Model training** with MLflow experiment tracking
- **Model serving** via a REST API (FastAPI)
- **Containerization** with Docker Compose (MLflow server + API service)
- **Data drift monitoring** with Evidently AI

By the end, you will have a working system where you can train a model, serve predictions, simulate production traffic, and detect data drift — all running in containers.

---

## Learning Objectives

By the end of this lesson you will be able to:
1. Understand how MLflow, FastAPI, and Docker work together in an MLOps pipeline
2. Train a model and log experiments to MLflow
3. Serve predictions via a REST API that loads models from MLflow
4. Simulate production traffic and detect data drift
5. Orchestrate multiple services with Docker Compose

---

## Software You Will Use

| Tool | What It Does | Where It Lives |
|------|-------------|---------------|
| **MLflow** | Experiment tracking and model registry | `mlflow-server` container |
| **FastAPI** | REST API framework for model serving | `app-service` container |
| **Docker Compose** | Multi-container orchestration | Host machine |
| **scikit-learn** | Machine learning (RandomForest) | `app-service` container |
| **Evidently AI** | Data drift detection and reporting | `app-service` container |
| **pandas** | Data manipulation | Both containers |

> **Reference**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — experiment tracking and model registry.
> **Reference**: [FastAPI Documentation](https://fastapi.tiangolo.com/) — modern Python web framework.
> **Reference**: [Evidently AI Documentation](https://docs.evidentlyai.com/) — ML monitoring and data drift.

---

## Project Structure

```
05-mlops-examples/
├── docker-compose.yml           # Orchestrates mlflow-server + app-service
├── simulate_traffic.py          # Synthetic client for data drift demo
├── cred_analysis_app/
│   ├── Dockerfile               # Container definition for the app
│   ├── requirements.txt         # Python dependencies
│   ├── data/
│   │   └── german_credit_data.csv  # Dataset (you provide this)
│   └── src/
│       ├── train.py             # Training script → logs to MLflow
│       ├── api.py               # FastAPI app → serves predictions
│       └── monitoring.py        # Drift detection → generates HTML report
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                              │
│                                                                          │
│  ┌─────────────────────┐         ┌─────────────────────┐               │
│  │    mlflow-server     │         │    app-service       │               │
│  │                     │         │                     │               │
│  │  MLflow Tracking    │◄────────│  FastAPI (api.py)   │               │
│  │  + Model Registry   │  read   │  serves predictions │               │
│  │                     │  model  │  on port 8001       │               │
│  │  SQLite DB          │         │                     │               │
│  │  Artifacts (/mlruns)│         │  Logs predictions   │               │
│  │  Port: 5001         │         │  to CSV             │               │
│  └─────────┬───────────┘         └──────────┬──────────┘               │
│            │                                │                           │
│            │         ┌──────────────────────┘                          │
│            │         │                                                  │
│            ▼         ▼                                                  │
│  ┌─────────────────────────┐    ┌──────────────────────┐              │
│  │      /mlruns volume     │    │  data/production_logs │              │
│  │  (shared between both)  │    │       .csv            │              │
│  └─────────────────────────┘    └──────────┬───────────┘              │
│                                            │                           │
│                                            ▼                           │
│                                 ┌──────────────────────┐              │
│                                 │    monitoring.py      │              │
│                                 │  Evidently AI drift   │              │
│                                 │  report → .html       │              │
│                                 └──────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
[German Credit CSV] ──► train.py ──► MLflow Server ──► /mlruns (shared volume)
                                                            │
                                                            ▼
                              app-service (api.py) loads model from MLflow
                                        │
                                        ▼
[Client / simulate_traffic.py] ──► POST /predict ──► api.py
                                        │
                                        ▼
                              data/production_logs.csv
                                        │
                                        ▼
                              monitoring.py ──► drift_report.html
```

---

## Step-by-Step: Getting Started

### Step 1 — Provide the dataset

This example uses the **german_credit_data** dataset from Kaggle. You need to place the CSV file in the data directory:

```bash
# Download the dataset (or use your own)
# The file should be named: german_credit_data.csv
# Place it at: cred_analysis_app/data/german_credit_data.csv

# Example download page:
# https://www.kaggle.com/datasets/varunchawla30/german-credit-data
```

The dataset should have columns like `laufkont`, `laufzeit`, `moral`, etc. (German column names) which will be mapped to English names during training.

> **Reference**: [german_credit_data](https://www.kaggle.com/datasets/varunchawla30/german-credit-data) — Kaggle dataset used by this lesson.

### Step 2 — Start the services

```bash
cd 05-mlops-examples

# Build and start both containers
docker compose up --build -d
```

This starts:
- **MLflow UI** at `http://localhost:5001`
- **API Endpoint** at `http://localhost:8001`

If the MLflow UI shows `Invalid Host header - possible DNS rebinding attack detected`, restart the compose stack after using the updated `--allowed-hosts` setting in `docker-compose.yml`. The browser sends `localhost:5001` as the Host header, so that exact value must be allowed.

### Step 3 — Train the model

```bash
# Run training inside the container (important: uses internal network)
docker compose run --rm app-service python src/train.py
```

You should see output like:
```
Loading data from data/german_credit_data.csv...
Categorical columns: 13
Numerical columns: 7
Starting run: <run_id>
Accuracy: 0.7550
F1 Score: 0.8491
Model logged to MLflow.
```

### Step 4 — Restart the API to load the new model

If the API was already running before training, restart it so it reloads the newest registered model:

```bash
docker compose restart app-service
```

If the API is fresh and has not loaded a model yet, the first `/predict` request will load the registered model automatically.

### Step 5 — Test the API

```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "checking_status": "A11",
           "duration": 6,
           "credit_history": "A34",
           "purpose": "A43",
           "credit_amount": 1169,
           "savings_status": "A65",
           "employment": "A75",
           "installment_commitment": 4,
           "personal_status": "A93",
           "other_parties": "A101",
           "residence_since": 4,
           "property_magnitude": "A121",
           "age": 67,
           "other_payment_plans": "A143",
           "housing": "A152",
           "existing_credits": 2,
           "job": "A173",
           "num_dependents": 1,
           "own_telephone": "A192",
           "foreign_worker": "A201"
         }'
```

**Expected response:**
```json
{"risk_prediction": 1, "class": "Good Risk"}
```

---

## Step-by-Step: Data Drift Detection

This demo shows how to detect when production data diverges from training data.

### Step 1 — Ensure services are running and model is trained

```bash
docker compose up --build -d
docker compose run --rm app-service python src/train.py
docker compose restart app-service
```

### Step 2 — Install traffic simulation dependencies (on host)

```bash
# Create a virtual environment for the simulation script
python -m venv .venv_traffic
source .venv_traffic/bin/activate  # Linux/Mac
pip install pandas requests
```

### Step 3 — Run the traffic simulation

```bash
python simulate_traffic.py
```

This sends 100 requests to the API:
- **Phase 1** (50 requests): Normal traffic — data from the original dataset
- **Phase 2** (50 requests): Drift scenario — `credit_amount` divided by 10, `age` fixed at 20

The API logs each request to `cred_analysis_app/data/production_logs.csv`.

### Step 4 — Generate the drift report

```bash
docker compose exec app-service python src/monitoring.py
```

This compares the original training data against the production logs and generates an HTML report.

### Step 5 — View the report

```bash
# Open the report in your browser
open cred_analysis_app/data/drift_report.html    # macOS
xdg-open cred_analysis_app/data/drift_report.html  # Linux
```

The report shows which features drifted and the statistical significance of the drift.

---

## Understanding the Code

### `train.py` — Model Training

1. Loads the German Credit dataset
2. Maps German column names to English
3. Builds a preprocessing pipeline (StandardScaler + OneHotEncoder)
4. Trains a RandomForestClassifier
5. Logs metrics (accuracy, precision, recall, F1) to MLflow
6. Registers the model as "CreditRiskModel" in MLflow Model Registry

### `api.py` — Model Serving

1. Loads the latest model from MLflow Model Registry on the first prediction request
2. Exposes `POST /predict` endpoint that accepts credit application JSON
3. Returns risk prediction (Good Risk / Bad Risk)
4. Logs every prediction to `production_logs.csv` for monitoring

### `monitoring.py` — Drift Detection

1. Loads reference data (original training set)
2. Loads current data (production logs from API)
3. Uses Evidently AI's `DataDriftPreset` to detect statistical drift
4. Generates a self-contained HTML report

### `simulate_traffic.py` — Traffic Generator

1. Reads the German Credit dataset
2. Sends normal traffic (Phase 1) then drift traffic (Phase 2)
3. Used to demonstrate the monitoring pipeline

---

## Docker Compose Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `mlflow-server` | `ghcr.io/mlflow/mlflow:v3.13.0` | 5001 | MLflow tracking + model registry |
| `app-service` | Built from `cred_analysis_app/Dockerfile` | 8001 | FastAPI inference server |

### Volumes

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `./mlruns` | `/mlruns` | Shared MLflow artifacts (both services) |
| `./cred_analysis_app/data` | `/app/data` | Persistent production logs + drift reports |

---

## Local Development (Without Docker)

If you prefer to run locally:

```bash
cd cred_analysis_app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set MLflow to use local SQLite
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Train the model
python src/train.py

# Start the API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Practice Questions

1. **Why must the training script run inside the Docker container** rather than on the host? What would happen if you ran it outside?
2. **How does the API know which model to load?** Trace the path from MLflow Model Registry to the API.
3. **What is data drift**, and why is it important to monitor in production?
4. **How would you extend this pipeline** to automatically retrain when drift is detected?

---

## Further Reading

- [MLflow Quickstart](https://mlflow.org/docs/latest/getting-started/index.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/) — ML monitoring and data drift
- [Docker Compose Overview](https://docs.docker.com/compose/)
- [MLOps Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
