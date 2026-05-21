import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import shap
import os

os.makedirs('outputs', exist_ok=True)

def compute_shap(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values, feature_names

def plot_shap_importance(shap_values, feature_names, model_name, tag):
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1]
    sorted_vals = mean_abs[idx]
    sorted_feats = [feature_names[i] for i in idx]

    modifiable = ['ap_hi','ap_lo','weight','bmi','pulse_pressure',
                  'cholesterol','gluc','smoke','alco','active',
                  'trestbps','chol','fbs','thalach','exang','oldpeak']

    colors = ['#FF5722' if f in modifiable else '#2196F3' for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(sorted_feats)), sorted_vals, color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels(sorted_feats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'SHAP Feature Importance — {model_name} ({tag})', fontweight='bold')

    patch_mod = mpatches.Patch(color='#FF5722', label='Modifiable feature')
    patch_non = mpatches.Patch(color='#2196F3', label='Non-modifiable feature')
    ax.legend(handles=[patch_mod, patch_non], loc='lower right', fontsize=8)

    plt.tight_layout()
    path = f'outputs/shap_{tag}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  [Saved] SHAP chart -> {path}')
    return path
