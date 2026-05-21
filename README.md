# PSB605IT — CVD Risk Prediction System
**Student:** Sudhan Nagarajan | **ID:** 050J0DAD  
**Supervisor:** Mr Raymond Ching Chi Man  
**Module:** PSB605IT Individual Computing Science Project  
**School:** PSB Academy | Academic Year 2025/2026

---

## Project Overview
Machine learning-based cardiovascular disease (CVD) risk prediction system
trained on two datasets using Logistic Regression, Random Forest and XGBoost
classifiers with SMOTE oversampling, GridSearchCV hyperparameter tuning,
SHAP explainability and a Streamlit prototype dashboard.

---

## Datasets Required
Place both files in the `data/` directory before running:

| File | Source |
|---|---|
| `data/cardio_train.csv` | https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset |
| `data/processed.cleveland.data` | https://www.kaggle.com/datasets/lourenswalters/uci-heart-disease-data-set |

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt