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
from pydantic import BaseModel, create_model
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
    checking_status: Optional[str]
    duration: Optional[int]
    credit_history: Optional[str]
    purpose: Optional[str]
    credit_amount: Optional[int]
    savings_status: Optional[str]
    employment: Optional[str]
    installment_commitment: Optional[int]
    personal_status: Optional[str]
    other_parties: Optional[str]
    residence_since: Optional[int]
    property_magnitude: Optional[str]
    age: Optional[int]
    other_payment_plans: Optional[str]
    housing: Optional[str]
    existing_credits: Optional[int]
    job: Optional[str]
    num_dependents: Optional[int]
    own_telephone: Optional[str]
    foreign_worker: Optional[str]

    class Config:
        extra = "allow" 

@app.on_event("startup")
def load_model():
    global model
    try:
        # Set tracking URI so it knows where to look
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"Loading model from {MODEL_URI}...")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("WARNING: API starting without model. Predictions will fail.")

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
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_data = application.dict()
    df = pd.DataFrame([input_data])
    
    try:
        # Predict
        prediction = model.predict(df)
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
