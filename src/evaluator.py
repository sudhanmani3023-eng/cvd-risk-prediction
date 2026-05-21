# ============================================================
# evaluator.py
# PSB605IT - CVD Risk Prediction System
# Student: Sudhan Nagarajan | ID: 050J0DAD
# Computes all evaluation metrics and generates output charts
# References: Hosmer et al. (2013), Pedregosa et al. (2011)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)


# ─────────────────────────────────────────────
# COMPUTE METRICS FOR ONE MODEL
# ─────────────────────────────────────────────
def compute_metrics(model, X_test, y_test, model_name="Model"):
    """
    Compute accuracy, sensitivity, specificity, precision, F1 and AUC
    for a single trained classifier on the test set.

    Returns dict of metric values.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)   # sensitivity / TPR
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # specificity

    return {
        "Model"       : model_name,
        "Accuracy"    : round(acc,  4),
        "Sensitivity" : round(rec,  4),
        "Specificity" : round(spec, 4),
        "Precision"   : round(prec, 4),
        "F1-Score"    : round(f1,   4),
        "AUC-ROC"     : round(auc,  4),
        "TP" : int(tp), "FP" : int(fp),
        "TN" : int(tn), "FN" : int(fn),
        "y_pred"  : y_pred,
        "y_proba" : y_proba
    }


# ─────────────────────────────────────────────
# EVALUATE ALL MODELS FOR ONE DATASET
# ─────────────────────────────────────────────
def evaluate_all(models, X_test, y_test, dataset_tag="cardio",
                 output_dir="outputs", save_csv=True):
    """
    Evaluate all three classifiers and print a formatted comparison table.
    Saves metrics to CSV and generates ROC curve chart.

    Parameters
    ----------
    models      : dict  — {model_name: fitted_estimator}
    X_test      : array — scaled test features
    y_test      : array — true test labels
    dataset_tag : str   — 'cardio' or 'uci'
    output_dir  : str   — directory for output files
    save_csv    : bool  — save metrics to CSV

    Returns
    -------
    results_df  : pd.DataFrame — all metrics for all models
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    roc_data    = []

    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — Dataset: {dataset_tag.upper()}")
    print("=" * 60)

    for name, model in models.items():
        res = compute_metrics(model, X_test, y_test, model_name=name)
        all_results.append(res)

        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        roc_data.append((name, fpr, tpr, res["AUC-ROC"]))

        print(f"\n  ── {name} ──")
        print(f"     Accuracy    : {res['Accuracy']:.4f}")
        print(f"     Sensitivity : {res['Sensitivity']:.4f}")
        print(f"     Specificity : {res['Specificity']:.4f}")
        print(f"     Precision   : {res['Precision']:.4f}")
        print(f"     F1-Score    : {res['F1-Score']:.4f}")
        print(f"     AUC-ROC     : {res['AUC-ROC']:.4f}")
        print(f"     Confusion   : TP={res['TP']} FP={res['FP']} TN={res['TN']} FN={res['FN']}")

    # Build summary DataFrame
    metric_cols = ["Model", "Accuracy", "Sensitivity", "Specificity",
                   "Precision", "F1-Score", "AUC-ROC"]
    results_df = pd.DataFrame(all_results)[metric_cols]

    print("\n" + "─" * 60)
    print("  SUMMARY TABLE")
    print("─" * 60)
    print(results_df.to_string(index=False))
    print("─" * 60)

    # Identify champion model
    champion = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
    print(f"\n  ★ Champion Model (highest F1): {champion}")

    # Save CSV
    if save_csv:
        csv_path = f"{output_dir}/metrics_{dataset_tag}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"  [Saved] Metrics → {csv_path}")

    # Plot ROC curves
    _plot_roc_curves(roc_data, dataset_tag, output_dir)

    print("=" * 60)
    return results_df


# ─────────────────────────────────────────────
# PLOT ROC CURVES
# ─────────────────────────────────────────────
def _plot_roc_curves(roc_data, dataset_tag, output_dir):
    """Generate and save ROC curve comparison chart."""
    colours = ["#2196F3", "#4CAF50", "#FF5722"]
    styles  = ["-", "--", "-."]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, fpr, tpr, auc) in enumerate(roc_data):
        ax.plot(fpr, tpr,
                color=colours[i % len(colours)],
                linestyle=styles[i % len(styles)],
                linewidth=2.5,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)",       fontsize=12)
    ax.set_title(
        f"ROC Curve Comparison — {dataset_tag.upper()} Dataset\n"
        f"(Hosmer et al., 2013 | Pedregosa et al., 2011)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    path = f"{output_dir}/roc_curves_{dataset_tag}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] ROC curves → {path}")


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test,
                          model_name="Model", dataset_tag="cardio",
                          output_dir="outputs"):
    """Generate and save a confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No CVD", "CVD"],
        yticklabels=["No CVD", "CVD"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(
        f"Confusion Matrix — {model_name}\n({dataset_tag.upper()} dataset)",
        fontsize=11, fontweight="bold"
    )

    path = f"{output_dir}/cm_{model_name.replace(' ','_').lower()}_{dataset_tag}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Confusion matrix → {path}")
    return path


# ─────────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader   import load_cardio, load_uci
    from preprocessor  import preprocess_cardio, preprocess_uci
    from balancer      import apply_smote
    from model_engine  import train_all_models

    print("\n[INFO] Running evaluator.py standalone...")

    # Dataset 1 - Cardio
    df_c = load_cardio()
    X_tr, X_v, X_te, y_tr, y_v, y_te, feat, _ = preprocess_cardio(df_c)
    X_res, y_res = apply_smote(X_tr, y_tr)
    models_c, _ = train_all_models(X_res, y_res, dataset_tag="cardio")
    evaluate_all(models_c, X_te, y_te, dataset_tag="cardio")

    # Dataset 2 - UCI
    df_u = load_uci()
    X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2, feat2, _ = preprocess_uci(df_u)
    X_res2, y_res2 = apply_smote(X_tr2, y_tr2)
    models_u, _ = train_all_models(X_res2, y_res2, dataset_tag="uci")
    evaluate_all(models_u, X_te2, y_te2, dataset_tag="uci")

    print("\n[DONE] Evaluation complete.\n")