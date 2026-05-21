# ============================================================
# balancer.py
# PSB605IT - CVD Risk Prediction System
# Student: Sudhan Nagarajan | ID: 050J0DAD
# Applies SMOTE oversampling to training data only
# Reference: Chawla et al. (2002)
# ============================================================

import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train, random_state=42, k_neighbors=5):
    """
    Apply SMOTE synthetic minority oversampling to training data only.
    Validation and test sets are NEVER resampled.

    Parameters
    ----------
    X_train      : np.ndarray — scaled training feature array
    y_train      : np.ndarray — training label array
    random_state : int        — reproducibility seed (default 42)
    k_neighbors  : int        — number of nearest neighbours for interpolation

    Returns
    -------
    X_res : np.ndarray — resampled feature array
    y_res : np.ndarray — resampled label array
    """
    print("\n" + "=" * 60)
    print("  SMOTE OVERSAMPLING (Chawla et al., 2002)")
    print("=" * 60)

    before = Counter(y_train)
    print(f"  Class distribution BEFORE SMOTE : {dict(before)}")

    # Adjust k_neighbors if minority class is very small (UCI dataset)
    min_class_count = min(before.values())
    k = min(k_neighbors, min_class_count - 1)
    if k < 1:
        k = 1
    if k != k_neighbors:
        print(f"  [Adjusted] k_neighbors set to {k} (minority class has {min_class_count} samples)")

    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    after = Counter(y_res)
    print(f"  Class distribution AFTER SMOTE  : {dict(after)}")
    print(f"  Samples added                   : {len(X_res) - len(X_train)}")
    print("=" * 60)

    return X_res, y_res


# ─────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader import load_cardio, load_uci
    from preprocessor import preprocess_cardio, preprocess_uci

    print("\n[INFO] Running balancer.py standalone...")

    # Dataset 1
    df_c = load_cardio()
    X_tr, X_v, X_te, y_tr, y_v, y_te, _, _ = preprocess_cardio(df_c)
    X_res, y_res = apply_smote(X_tr, y_tr)
    print(f"  Cardio resampled → X: {X_res.shape}, y: {y_res.shape}")

    # Dataset 2
    df_u = load_uci()
    X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2, _, _ = preprocess_uci(df_u)
    X_res2, y_res2 = apply_smote(X_tr2, y_tr2)
    print(f"  UCI resampled    → X: {X_res2.shape}, y: {y_res2.shape}")

    print("\n[DONE] Balancing complete.\n")