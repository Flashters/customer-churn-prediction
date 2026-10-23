"""
Project 2: Customer Churn Prediction - Inference Script
Description: Loads the saved production pipeline bundle and makes predictions on new customer data.
"""

import os
import joblib
import pandas as pd

def load_production_model(model_path: str):
    """Load the saved joblib pipeline bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found at {model_path}. Please run train.py first.")
    
    print(f"Loading model bundle from: {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline

if __name__ == "__main__":
    model_path = "models/churn_pipeline.pkl"
    
    # 1. Load the trained pipeline
    churn_pipeline = load_production_model(model_path)
    
    # 2. Create sample new customer data (matching the columns from your training set)
    sample_data = pd.DataFrame([
        {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85
        }
    ])
    
    print("\nPredicting churn for sample customer...")
    
    # 3. Generate prediction (0 = Stayed, 1 = Churned)
    prediction = churn_pipeline.predict(sample_data)
    prediction_proba = churn_pipeline.predict_proba(sample_data)
    
    churn_status = "Churned (Yes)" if prediction[0] == 1 else "Retained (No)"
    churn_probability = prediction_proba[0][1] * 100
    
    print(f"👉 Prediction Result: {churn_status}")
    print(f"📊 Churn Probability: {churn_probability:.2f}%")