# ============================================================
# model_engine.py
# PSB605IT - CVD Risk Prediction System
# Student: Sudhan Nagarajan | ID: 050J0DAD
# Trains LR, RF, XGBoost with GridSearchCV hyperparameter tuning
# References: Hosmer et al. (2013), Breiman (2001),
#             Chen & Guestrin (2016), Pedregosa et al. (2011)
# ============================================================

import os
import time
import joblib
import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost                 import XGBClassifier


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACES
# ─────────────────────────────────────────────
PARAM_GRID_LR = {
    "C"      : [0.01, 0.1, 1, 10],
    "solver" : ["lbfgs", "saga"],
    "max_iter": [1000]
}

PARAM_GRID_RF = {
    "n_estimators" : [50, 100, 200],
    "max_depth"    : [None, 10, 20],
    "max_features" : ["sqrt", "log2"]
}

PARAM_GRID_XGB = {
    "n_estimators"  : [100, 200, 300],
    "learning_rate" : [0.01, 0.1, 0.3],
    "max_depth"     : [3, 5, 7],
    "subsample"     : [0.8, 1.0]
}


# ─────────────────────────────────────────────
# TRAIN SINGLE MODEL WITH GRID SEARCH
# ─────────────────────────────────────────────
def train_model(name, estimator, param_grid, X_train, y_train,
                cv_folds=5, scoring="f1", verbose=True):
    """
    Train a classifier using GridSearchCV with stratified k-fold CV.

    Parameters
    ----------
    name       : str   — model label (LR / RF / XGBoost)
    estimator  : obj   — sklearn-compatible classifier
    param_grid : dict  — hyperparameter search space
    X_train    : array — scaled training features
    y_train    : array — training labels
    cv_folds   : int   — number of CV folds (default 5)
    scoring    : str   — optimisation criterion (default F1)

    Returns
    -------
    best_model : fitted best estimator
    best_params: dict of best hyperparameters
    """
    print(f"\n  ── Training {name} ──")
    print(f"     Search space : {param_grid}")
    print(f"     CV folds     : {cv_folds}")
    print(f"     Scoring      : {scoring}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    if verbose:
        print(f"     Best params  : {grid.best_params_}")
        print(f"     Best CV F1   : {grid.best_score_:.4f}")
        print(f"     Training time: {elapsed:.1f}s")

    return grid.best_estimator_, grid.best_params_


# ─────────────────────────────────────────────
# TRAIN ALL THREE MODELS FOR ONE DATASET
# ─────────────────────────────────────────────
def train_all_models(X_train, y_train, dataset_tag="cardio",
                     save_models=True, model_dir="models"):
    """
    Train LR, RF and XGBoost classifiers on the provided training data.
    Saves serialised .pkl model files.

    Parameters
    ----------
    X_train    : np.ndarray — scaled, SMOTE-resampled training features
    y_train    : np.ndarray — resampled training labels
    dataset_tag: str        — 'cardio' or 'uci' (used in file names)
    save_models: bool       — whether to serialise trained models
    model_dir  : str        — directory for .pkl files

    Returns
    -------
    models : dict  — {model_name: fitted_estimator}
    params : dict  — {model_name: best_hyperparameters}
    """
    print("\n" + "=" * 60)
    print(f"  MODEL TRAINING — Dataset: {dataset_tag.upper()}")
    print("=" * 60)

    os.makedirs(model_dir, exist_ok=True)

    # ── Logistic Regression (Hosmer et al., 2013)
    lr_est = LogisticRegression(random_state=42, class_weight="balanced")
    lr_model, lr_params = train_model(
        "Logistic Regression", lr_est, PARAM_GRID_LR, X_train, y_train
    )

    # ── Random Forest (Breiman, 2001)
    rf_est = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_model, rf_params = train_model(
        "Random Forest", rf_est, PARAM_GRID_RF, X_train, y_train
    )

    # ── XGBoost (Chen & Guestrin, 2016)
    # scale_pos_weight handles class imbalance in XGBoost
    pos  = int(np.sum(y_train == 0))
    neg  = int(np.sum(y_train == 1))
    spw  = pos / neg if neg > 0 else 1

    xgb_est = XGBClassifier(
        random_state=42,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )
    xgb_model, xgb_params = train_model(
        "XGBoost", xgb_est, PARAM_GRID_XGB, X_train, y_train
    )

    models = {
        "Logistic Regression" : lr_model,
        "Random Forest"       : rf_model,
        "XGBoost"             : xgb_model
    }

    params = {
        "Logistic Regression" : lr_params,
        "Random Forest"       : rf_params,
        "XGBoost"             : xgb_params
    }

    # Save models
    if save_models:
        joblib.dump(lr_model,  f"{model_dir}/lr_{dataset_tag}.pkl")
        joblib.dump(rf_model,  f"{model_dir}/rf_{dataset_tag}.pkl")
        joblib.dump(xgb_model, f"{model_dir}/xgb_{dataset_tag}.pkl")
        print(f"\n  [Saved] Models → {model_dir}/[lr|rf|xgb]_{dataset_tag}.pkl")

    print("=" * 60)
    return models, params


# ─────────────────────────────────────────────
# LOAD SAVED MODELS
# ─────────────────────────────────────────────
def load_models(dataset_tag="cardio", model_dir="models"):
    """
    Load serialised model files for a given dataset tag.
    Returns dict of {model_name: fitted_estimator}.
    """
    models = {
        "Logistic Regression" : joblib.load(f"{model_dir}/lr_{dataset_tag}.pkl"),
        "Random Forest"       : joblib.load(f"{model_dir}/rf_{dataset_tag}.pkl"),
        "XGBoost"             : joblib.load(f"{model_dir}/xgb_{dataset_tag}.pkl")
    }
    print(f"  [Loaded] Models for dataset: {dataset_tag}")
    return models


# ─────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader  import load_cardio, load_uci
    from preprocessor import preprocess_cardio, preprocess_uci
    from balancer     import apply_smote

    print("\n[INFO] Running model_engine.py standalone...")
    print("[INFO] This will train all 6 models. Please wait...\n")

    # ── Dataset 1
    df_c = load_cardio()
    X_tr, X_v, X_te, y_tr, y_v, y_te, feat, _ = preprocess_cardio(df_c)
    X_res, y_res = apply_smote(X_tr, y_tr)
    models_c, params_c = train_all_models(X_res, y_res, dataset_tag="cardio")
    print("\n  Cardio models trained successfully.")

    # ── Dataset 2
    df_u = load_uci()
    X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2, feat2, _ = preprocess_uci(df_u)
    X_res2, y_res2 = apply_smote(X_tr2, y_tr2)
    models_u, params_u = train_all_models(X_res2, y_res2, dataset_tag="uci")
    print("\n  UCI models trained successfully.")

    print("\n[DONE] All models trained and saved.\n")