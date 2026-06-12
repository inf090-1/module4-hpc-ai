"""
api.py — Online Inference / Serving Module
==========================================
Role in the System Architecture:
    This is the ONLINE / SERVING component of the MLOps pipeline.

    Responsibilities:
    - Exposes a REST API (built with FastAPI) that accepts credit application
      data as JSON and returns a risk prediction (Good Risk / Bad Risk).
    - At startup, loads the trained model directly from the MLflow Model
      Registry ("CreditRiskModel/Latest"), making the API always serve the
      most recently registered model version.
    - Logs every inference request together with its prediction to a local
      CSV file (data/production_logs.csv).  This log is later used by the
      monitoring module to detect data drift.

    Endpoints:
        GET  /health   → Health check; reports whether the model is loaded.
        POST /predict  → Accepts a CreditApplication JSON body and returns
                         {"risk_prediction": 0|1, "class": "Bad Risk"|"Good Risk"}.

    Where it fits:
        [MLflow Model Registry] ──► api.py ──► [Client / simulate_traffic.py]
                                        └──► data/production_logs.csv ──► monitoring.py

    Deployed as a Docker container (app-service) via docker-compose, reachable
    on host port 8001 (container port 8000).
"""

import os
import pandas as pd
import mlflow.pyfunc
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any

app = FastAPI(title="Credit Risk Classification API")

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
# We assume the model is registered as "CreditRiskModel" and we pull "Production" or "Latest"
# For this example, we'll try to load from the latest run or a specific URI structure.
# In a real scenario, we'd use a Model Registry stage.
MODEL_NAME = "CreditRiskModel"
# Loading from standard MLflow location logic or simplify by assuming a path if local
# But best is to use the tracking server.
MODEL_URI = f"models:/{MODEL_NAME}/Latest" # Or 'Production'

DATA_LOG_PATH = "data/production_logs.csv"

# Global model variable
model = None

# --- Input Schema ---
# Defining a Pydantic model for one record.
# Using a generic dict approach for flexibility with the German Credit dataset columns
# but ideally should be explicit.
class CreditApplication(BaseModel):
    model_config = ConfigDict(extra="allow")

    checking_status: Optional[str | int] = None
    duration: Optional[int] = None
    credit_history: Optional[str | int] = None
    purpose: Optional[str | int] = None
    credit_amount: Optional[int] = None
    savings_status: Optional[str | int] = None
    employment: Optional[str | int] = None
    installment_commitment: Optional[int] = None
    personal_status: Optional[str | int] = None
    other_parties: Optional[str | int] = None
    residence_since: Optional[int] = None
    property_magnitude: Optional[str | int] = None
    age: Optional[int] = None
    other_payment_plans: Optional[str | int] = None
    housing: Optional[str | int] = None
    existing_credits: Optional[int] = None
    job: Optional[str | int] = None
    num_dependents: Optional[int] = None
    own_telephone: Optional[str | int] = None
    foreign_worker: Optional[str | int] = None

def ensure_model_loaded():
    global model
    if model is not None:
        return model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"Loading model from {MODEL_URI}...")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("WARNING: API starting without model. Predictions will fail.")

    return model

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

def log_prediction(input_data: Dict[str, Any], prediction: int, probability: Optional[float] = None):
    """
    Logs input data and prediction to a local CSV file.
    """
    file_exists = os.path.isfile(DATA_LOG_PATH)
    
    # Prepare row
    row = input_data.copy()
    row['timestamp'] = datetime.now().isoformat()
    row['prediction'] = prediction
    # row['probability'] = probability # Optional if we had proba
    
    fieldnames = list(row.keys())
    
    try:
        with open(DATA_LOG_PATH, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error logging to CSV: {e}")

@app.post("/predict")
def predict(application: CreditApplication):
    current_model = ensure_model_loaded()
    if not current_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_data = application.model_dump()
    df = pd.DataFrame([input_data])
    categorical_cols = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    try:
        # Predict
        prediction = current_model.predict(df)
        result = int(prediction[0])
        
        # Log data
        log_prediction(input_data, result)
        
        # 1 = Good, 0 = Bad (per user instruction/dataset standard)
        return {
            "risk_prediction": result, 
            "class": "Good Risk" if result == 1 else "Bad Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
