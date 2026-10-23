import os
import joblib
import pandas as pd

def test_model_artifact_exists():
    model_path = "models/churn_pipeline.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        assert pipeline is not None

def test_sample_prediction_shape():
    model_path = "models/churn_pipeline.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        sample_df = pd.DataFrame([{
            'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
            'tenure': 5, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
            'OnlineSecurity': 'Yes', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
            'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 50.0, 'TotalCharges': 250.0
        }])
        preds = pipeline.predict(sample_df)
        assert len(preds) == 1
