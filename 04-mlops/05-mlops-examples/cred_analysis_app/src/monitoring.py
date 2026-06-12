"""
monitoring.py — Data Drift Monitoring Module
============================================
Role in the System Architecture:
    This is the MONITORING component of the MLOps pipeline.

    Responsibilities:
    - Compares the REFERENCE distribution (the original German Credit training
      dataset) against the CURRENT distribution (production requests logged by
      the API in data/production_logs.csv).
    - Uses Evidently AI's DataDriftPreset to automatically compute and
      visualise statistical drift for every feature column.
    - Saves a self-contained HTML report to data/drift_report.html that can
      be opened in any browser for inspection.

    Where it fits:
        [data/german_credit_data.csv] ──┐
                                        ├──► monitoring.py ──► data/drift_report.html
        [data/production_logs.csv]   ───┘
                  ▲
           (written by api.py)

    Trigger: Run this script manually (or schedule it periodically) AFTER
    simulate_traffic.py has populated data/production_logs.csv.
    It is executed inside the app-service Docker container or locally.
"""

import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", os.path.join(DATA_DIR, "german_credit_data.csv"))
CURRENT_DATA_PATH = os.getenv("CURRENT_DATA_PATH", os.path.join(DATA_DIR, "production_logs.csv"))
REPORT_OUTPUT_PATH = os.getenv("REPORT_OUTPUT_PATH", os.path.join(DATA_DIR, "drift_report.html"))

# Mapping to ensure Reference Matches Current (English proper names)
# Note: production_logs.csv already has English headers from the API payload
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
}

def load_data():
    print("Loading reference data...")
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Reference data not found at {REFERENCE_DATA_PATH}")
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    
    # Rename reference columns to match production logs
    reference = reference.rename(columns=COLUMN_MAPPING)
    
    # Explicitly select only feature columns for comparison to avoid 'class' vs 'prediction' mismatch issues in drift
    features = list(COLUMN_MAPPING.values())
    reference = reference[features]
    
    # Cast categorical columns to string in REFERENCE to match what typically comes from JSON/CSV production logs
    # (Evidently needs types to match somewhat or it might infer differently)
    categorical_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
    ]
    for col in categorical_cols:
         if col in reference.columns:
             reference[col] = reference[col].astype(str)


    print("Loading current production logs...")
    if not os.path.exists(CURRENT_DATA_PATH):
        raise FileNotFoundError(f"Production logs not found at {CURRENT_DATA_PATH}. Run simulation first.")
    current = pd.read_csv(CURRENT_DATA_PATH)
    
    # Filter current to features as well
    # current might have 'timestamp', 'prediction' which we can ignore for Feature Drift
    current = current[features]
    
    # Ensure types align in current as well
    for col in categorical_cols:
         if col in current.columns:
             current[col] = current[col].astype(str)
             
    return reference, current

def generate_report(reference, current):
    print("Generating Evidently Data Drift Report...")
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    snapshot = report.run(current_data=current, reference_data=reference)
    
    print(f"Saving report to {REPORT_OUTPUT_PATH}...")
    snapshot.save_html(REPORT_OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    reference_data, current_data = load_data()
    generate_report(reference_data, current_data)
