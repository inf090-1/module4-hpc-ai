"""
train.py — Model Training Module
=================================
Role in the System Architecture:
    This is the OFFLINE / TRAINING component of the MLOps pipeline.

    Responsibilities:
    - Loads the German Credit dataset (data/german_credit_data.csv) and
      performs all preprocessing (column renaming, type casting).
    - Builds a scikit-learn Pipeline that chains a ColumnTransformer
      (StandardScaler + OneHotEncoder) with a RandomForestClassifier.
    - Logs hyperparameters, evaluation metrics (accuracy, precision, recall,
      F1) and the trained model artifact to the MLflow Tracking Server.
    - Registers the serialised model in the MLflow Model Registry under the
      name "CreditRiskModel" so that the serving API can load it by name.

    Where it fits:
        [Data CSV] ──► train.py ──► [MLflow Tracking Server]
                                        └──► Model Registry (CreditRiskModel)

    Run this script once (or on every re-training cycle) before starting the
    API service.  It connects to the MLflow server whose URI is set via the
    MLFLOW_TRACKING_URI environment variable (default: http://localhost:5000).
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "credit-risk-classification"
DATA_PATH = "data/german_credit_data.csv"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please ensure german_credit_data.csv is present.")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # German to English mapping
    column_mapping = {
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
        'gastarb': 'foreign_worker',
        'kredit': 'class'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Explicitly cast categorical columns to string to ensure OneHotEncoder treats them as categories
    # even if they are encoded as integers in the source CSV.
    categorical_cols_to_cast = [
        'checking_status', 'credit_history', 'purpose', 'savings_status', 
        'employment', 'personal_status', 'other_parties', 'property_magnitude',
        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
    ]
    
    for col in categorical_cols_to_cast:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading data...")
    df = load_data()
    
    # Target column in OpenML credit-g is 'class' ('good', 'bad')
    target_col = 'class'
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Target in german_credit_data.csv is 'class' (mapped from 'kredit'), already 0/1.
    # We will keep it as is. 
    # Typically in this dataset: 1 = Good, 0 = Bad (or sometimes 1/2). 
    # Let's inspect unique values if needed, but assuming 0/1 from user context.
    # If the CSV has 1 and 2 (common in Statlog), we might need to map. 
    # But user said "1=Good, 0=Bad". We trust the user/CSV.
    pass

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Define Model Pipeline
    # Using a simple RandomForest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Start MLflow Run
    with mlflow.start_run() as run:
        print(f"Starting run: {run.info.run_id}")
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log Params
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Log Model
        # Using sklearn autolog logic or manual log_model
        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name="CreditRiskModel"
        )
        
        print("Model logged to MLflow.")

if __name__ == "__main__":
    main()
