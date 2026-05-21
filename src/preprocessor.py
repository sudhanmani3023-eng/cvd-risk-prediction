# ============================================================
# preprocessor.py
# PSB605IT - CVD Risk Prediction System
# Student: Sudhan Nagarajan | ID: 050J0DAD
# Cleans, engineers features, scales and splits both datasets
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# DATASET 1: CARDIO PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_cardio(df, save_scaler=True, scaler_path="models/scaler_cardio.pkl"):
    """
    Full preprocessing pipeline for Cardiovascular Disease dataset (Ulianova, 2019).
    Steps:
      1. Drop ID column
      2. Remove physiologically impossible outliers
      3. Convert age from days to years
      4. Engineer BMI and pulse pressure features
      5. Stratified train/val/test split (70/15/15)
      6. StandardScaler fit on training data only
      7. Return scaled arrays
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING: DATASET 1 (Cardio)")
    print("=" * 60)

    df = df.copy()

    # Step 1: Drop ID
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
        print("  [Step 1] Dropped ID column.")

    # Step 2: Outlier removal (clinically impossible values)
    initial_rows = len(df)
    df = df[
        (df["ap_hi"] > 50) & (df["ap_hi"] < 250) &
        (df["ap_lo"] > 40) & (df["ap_lo"] < 200) &
        (df["height"] > 100) & (df["height"] < 220) &
        (df["weight"] > 30) & (df["weight"] < 200)
    ]
    removed = initial_rows - len(df)
    print(f"  [Step 2] Removed {removed} outlier rows. Remaining: {len(df)}")

    # Step 3: Convert age from days to years
    df["age"] = (df["age"] / 365.25).round(1)
    print("  [Step 3] Converted age from days to years.")

    # Step 4: Feature engineering
    df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    print("  [Step 4] Engineered: bmi, pulse_pressure.")

    # Step 5: Define features and target
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    feature_names = list(X.columns)
    print(f"  [Step 5] Features ({len(feature_names)}): {feature_names}")
    print(f"           Target distribution: {y.value_counts().to_dict()}")

    # Step 6: Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
    )  # 0.1765 of 85% ≈ 15% of total

    print(f"  [Step 6] Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 7: Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    print("  [Step 7] StandardScaler applied (fit on train only).")

    # Save scaler
    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  [Saved] Scaler → {scaler_path}")

    print("=" * 60)

    return (
        X_train_sc, X_val_sc, X_test_sc,
        np.array(y_train), np.array(y_val), np.array(y_test),
        feature_names, scaler
    )


# ─────────────────────────────────────────────
# DATASET 2: UCI PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_uci(df, save_scaler=True, scaler_path="models/scaler_uci.pkl"):
    """
    Full preprocessing pipeline for UCI Cleveland Heart Disease dataset
    (Detrano et al., 1989).
    Steps:
      1. Drop rows with missing values (ca, thal columns)
      2. Cast ca and thal to numeric
      3. Stratified train/val/test split (70/15/15)
      4. StandardScaler fit on training data only
      5. Return scaled arrays
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING: DATASET 2 (UCI Cleveland)")
    print("=" * 60)

    df = df.copy()

    # Step 1: Handle missing values
    initial_rows = len(df)
    df["ca"]   = pd.to_numeric(df["ca"],   errors="coerce")
    df["thal"] = pd.to_numeric(df["thal"], errors="coerce")
    df.dropna(inplace=True)
    removed = initial_rows - len(df)
    print(f"  [Step 1] Removed {removed} rows with missing values. Remaining: {len(df)}")

    # Step 2: Features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    feature_names = list(X.columns)
    print(f"  [Step 2] Features ({len(feature_names)}): {feature_names}")
    print(f"           Target distribution: {y.value_counts().to_dict()}")

    # Step 3: Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
    )

    print(f"  [Step 3] Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 4: Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    print("  [Step 4] StandardScaler applied (fit on train only).")

    # Save scaler
    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  [Saved] Scaler → {scaler_path}")

    print("=" * 60)

    return (
        X_train_sc, X_val_sc, X_test_sc,
        np.array(y_train), np.array(y_val), np.array(y_test),
        feature_names, scaler
    )


# ─────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_cardio, load_uci

    print("\n[INFO] Running preprocessor.py standalone...")

    df_cardio = load_cardio()
    X_tr, X_v, X_te, y_tr, y_v, y_te, feat, sc = preprocess_cardio(df_cardio)
    print(f"\n  Cardio → Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    df_uci = load_uci()
    X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2, feat2, sc2 = preprocess_uci(df_uci)
    print(f"  UCI   → Train: {X_tr2.shape}, Val: {X_v2.shape}, Test: {X_te2.shape}")

    print("\n[DONE] Preprocessing complete.\n")