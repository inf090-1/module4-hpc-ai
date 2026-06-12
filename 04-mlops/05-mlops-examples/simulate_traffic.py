"""
simulate_traffic.py — Traffic Simulation Utility
=================================================
Role in the System Architecture:
    This is a DEVELOPMENT / DEMO utility that sits OUTSIDE the production
    services but acts as a synthetic client to drive the pipeline end-to-end.

    Responsibilities:
    - Reads the German Credit dataset and sends individual rows as JSON POST
      requests to the /predict endpoint of the running API service.
    - Simulates two traffic phases to demonstrate data drift detection:
        Phase 1 – Normal traffic: 50 random samples sent as-is.
        Phase 2 – Drift scenario: 50 samples with artificially modified
                  "credit_amount" (÷10) and "age" (fixed at 20) to mimic an
                  economic downturn / demographic shift.
    - The API logs each request to data/production_logs.csv, which is then
      analysed by monitoring.py to detect the introduced drift.

    Where it fits:
        simulate_traffic.py ──► POST /predict ──► api.py
                                                     └──► data/production_logs.csv

    Usage (run from the project root after the API is up):
        python simulate_traffic.py
"""

import os
import pandas as pd
import requests
import time
import random

# Configuration
DATA_PATH = os.getenv("DATA_PATH", "cred_analysis_app/data/german_credit_data.csv")
API_URL = os.getenv("API_URL", "http://localhost:8001/predict")
DELAY = float(os.getenv("DELAY", "0.5"))  # Seconds between requests

# Column Mapping (German -> English) to match API schema
COLUMN_MAPPING = {
    'laufkont': 'checking_status',
    'laufzeit': 'duration',
    'moral': 'credit_history',
    'verw': 'purpose',
    'hoehe': 'credit_amount',
    'sparkont': 'savings_status',
    'beszeit': 'employment',
    'rate': 'installment_commitment',
    'famges': 'personal_status',
    'buerge': 'other_parties',
    'wohnzeit': 'residence_since',
    'verm': 'property_magnitude',
    'alter': 'age',
    'weitkred': 'other_payment_plans',
    'wohn': 'housing',
    'bishkred': 'existing_credits',
    'beruf': 'job',
    'pers': 'num_dependents',
    'telef': 'own_telephone',
    'gastarb': 'foreign_worker'
    # 'kredit' is target, exclude from features
}

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}. Make sure you are running this from the project root.")
        exit(1)

def send_prediction(row):
    # Convert row to dict and map keys to English
    payload = {}
    for german_col, english_col in COLUMN_MAPPING.items():
        if german_col in row:
            val = row[german_col]
            # Ensure proper types (int/float conversion if needed happens in requests json dump, 
            # but we explicitly keep them as native python types)
            # Handle potential numpy types
            if hasattr(val, 'item'):
                val = val.item()
            payload[english_col] = val
            
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print(f"Status: {response.status_code} | Prediction: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")

def main():
    df = load_data()
    
    # Phase 1: Normal Traffic
    print("\n--- PHASE 1: Simulating Normal Traffic (50 requests) ---")
    normal_sample = df.sample(n=50)
    for index, row in normal_sample.iterrows():
        send_prediction(row)
        time.sleep(DELAY)

    # Phase 2: Drift / Crisis Scenario
    print("\n--- PHASE 2: Simulating Data Drift / Crisis (50 requests) ---")
    print("Modifying data: 'credit_amount' / 10 (Economic downturn), 'age' = 20 (Demographic shift)")
    
    drift_sample = df.sample(n=50)
    for index, row in drift_sample.iterrows():
        # Apply Drift
        # 'hoehe' is Credit Amount, 'alter' is Age in original CSV
        row['hoehe'] = int(row['hoehe'] / 10) 
        row['alter'] = 20
        
        send_prediction(row)
        time.sleep(DELAY)
        
    print("\nSimulation completed.")

if __name__ == "__main__":
    main()
