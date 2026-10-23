"""
Project 2: Customer Churn Prediction
Author: Dean Matthew C. Flores
Description: End-to-end machine learning classification pipeline utilizing scikit-learn 
pipelines, stratified k-fold cross-validation, and model benchmarking (Logistic Regression vs. Random Forest).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ==========================================
# DRILL PART 1: Exploratory Data Analysis (EDA) & Data Cleaning Checkpoints
# ==========================================
def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    DRILL: Load raw data, check data types, inspect missing values, 
    and verify class distribution (churn vs. non-churn).
    """
    print("--- [DRILL 1] Loading and Inspecting Dataset ---")
    df = pd.read_csv(filepath)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values per Column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if 'Churn' in df.columns:
        print(f"Target Class Distribution:\n{df['Churn'].value_counts(normalize=True)}")
        
    return df

# ==========================================
# DRILL PART 2: Feature Engineering & Preprocessing Setup
# ==========================================
def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    DRILL: Identify numerical and categorical columns, construct 
    leakage-safe imputation and scaling/encoding transformers using ColumnTransformer.
    """
    print("\n--- [DRILL 2] Building Preprocessing Pipeline ---")
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical Features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# ==========================================
# DRILL PART 3: Model Training & Cross-Validation Benchmarking
# ==========================================
def train_and_benchmark(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer):
    """
    DRILL: Set up Stratified 5-Fold Cross-Validation, build pipelines for 
    Logistic Regression and Random Forest, and compare ROC-AUC scores.
    """
    print("\n--- [DRILL 3] Model Benchmarking & Stratified Cross-Validation ---")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_score = 0.0
    best_model_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)
        
        print(f"Model: {name} | Mean ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

        if mean_auc > best_score:
            best_score = mean_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n✨ Best Performing Model: {best_model_name} with ROC-AUC: {best_score:.4f}")
    
    best_pipeline.fit(X, y)
    return best_pipeline

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "churn_pipeline.pkl")
    
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(data_path):
        df = load_and_inspect_data(data_path)
        
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        X = df.drop(columns=['Churn'])
        y = df['Churn'].apply(lambda val: 1 if str(val).lower() in ['yes', '1', 'true'] else 0)
        
        preprocessor = build_preprocessing_pipeline(X)
        production_pipeline = train_and_benchmark(X, y, preprocessor)
        
        joblib.dump(production_pipeline, model_path)
        print(f"\n💾 Production pipeline successfully saved to: {model_path}")
    else:
        print(f"⚠️ Dataset not found at '{data_path}'! Please place your CSV file in data/raw/ to run training.")