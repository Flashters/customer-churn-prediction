# Customer Churn Prediction Pipeline

## 📌 Project Overview
An end-to-end machine learning classification project designed to predict telecom customer churn. By identifying at-risk accounts early, business teams can optimize proactive retention campaigns.

---

## 🛠️ Tech Stack & Architecture
- **Language:** Python
- **Core Libraries:** `pandas`, `scikit-learn`, `numpy`, `joblib`, `matplotlib`, `seaborn`
- **Validation Strategy:** Stratified 5-Fold Cross-Validation
- **Evaluation Metrics:** ROC-AUC, Precision, Recall

---

## 📂 Project Structure & Execution
- **`train.py`:** End-to-end machine learning script incorporating EDA drills, leakage-safe preprocessing pipelines, model benchmarking (Logistic Regression vs. Random Forest), and model serialization.
- **`inference.py`:** Prediction script that loads the serialized production pipeline bundle (`models/churn_pipeline.pkl`) to evaluate new customer data records.

---

## 🔍 Implementation Drills & Methodology
This codebase incorporates structured workflow checkpoints (drills) to ensure model rigor:
1. **EDA & Data Validation Checkpoint (`load_and_inspect_data`):** Validates raw file integrity, checks feature data types, scans for missing values, and evaluates target class distribution balance.
2. **Preprocessing & Feature Engineering (`build_preprocessing_pipeline`):** Configures isolated numerical scaling (median imputation + StandardScaler) and categorical encoding (constant imputation + OneHotEncoder) inside a `ColumnTransformer`.
3. **Model Benchmarking (`train_and_benchmark`):** Rigorously compares **Logistic Regression** against **Random Forest** classifiers across stratified folds to eliminate data leakage and guard against overfitting.

---

## 📖 Technical Glossary
- **Data Leakage:** An error where information from outside the training dataset contaminates model training. Prevented here by encapsulating all transformations within `scikit-learn` Pipelines.
- **Stratified K-Fold:** A cross-validation technique guaranteeing that each fold preserves the exact percentage of samples for each target class.
- **ROC-AUC:** Measures a model's capacity to discriminate between churn and non-churn classes across varying decision thresholds.