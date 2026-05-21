import joblib, os, warnings, numpy as np
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

from src.data_loader import load_cardio, load_uci
from src.preprocessor import preprocess_cardio, preprocess_uci
from src.explainer import compute_shap, plot_shap_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def get_splits(result):
    r = list(result)
    sc = r[-1]
    data = r[:-1]
    n = len(data) // 2
    X_parts = data[:n]
    y_parts = data[n:]
    return X_parts[0], X_parts[1], X_parts[2], y_parts[0], y_parts[1], y_parts[2], sc

def fast_pipeline(tag):
    print(f'\n>>> {tag.upper()} — loading...')
    if tag == 'cardio':
        df = load_cardio()
        result = preprocess_cardio(df)
    else:
        df = load_uci()
        result = preprocess_uci(df)

    print(f'    preprocess returned {len(result)} values')
    r = list(result)
    sc = r[-1]
    Xtr, Xv, Xte = r[0], r[1], r[2]
    ytr, yv, yte  = r[3], r[4], r[5]

    Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    joblib.dump(sc, f'models/scaler_{tag}.pkl')

    mdls = {
        'lr':  LogisticRegression(C=1, max_iter=500, random_state=42),
        'rf':  RandomForestClassifier(n_estimators=30, max_depth=8,
                                      n_jobs=-1, random_state=42),
        'xgb': XGBClassifier(n_estimators=30, max_depth=4,
                             learning_rate=0.1, verbosity=0,
                             random_state=42, eval_metric='logloss')
    }

    print(f'    {"Model":6s}  Acc    F1     AUC')
    print(f'    {"-----":6s}  -----  -----  -----')
    for name, m in mdls.items():
        m.fit(Xtr, ytr)
        p  = m.predict(Xte)
        pr = m.predict_proba(Xte)[:,1]
        acc = accuracy_score(yte, p)
        f1  = f1_score(yte, p)
        auc = roc_auc_score(yte, pr)
        print(f'    {name.upper():6s}  {acc:.3f}  {f1:.3f}  {auc:.3f}')
        joblib.dump(m, f'models/{name}_{tag}.pkl')

    print(f'  [SHAP] computing...')
    feat = list(Xtr.columns) if hasattr(Xtr,'columns') else [f'f{i}' for i in range(Xtr.shape[1])]
    sv, fn = compute_shap(mdls['xgb'], Xte, feat)
    plot_shap_importance(sv, fn, 'XGBoost', tag)
    print(f'  [DONE] {tag} complete.')

fast_pipeline('cardio')
fast_pipeline('uci')
print('\nALL DONE — run: streamlit run dashboard.py')
