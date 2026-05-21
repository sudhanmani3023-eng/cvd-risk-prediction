with open('main.py', encoding='utf-8') as f:
    content = f.read()

old = '''def run_shap(model, X_te, features, model_name, tag):
    print(f'\\n  [SHAP] Computing SHAP values — {model_name} ({tag.upper()})')
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_te)
    print(f'  [SHAP] Values shape: {shap_vals.shape}')'''

new = '''def run_shap(model, X_te, features, model_name, tag):
    print(f'\\n  [SHAP] Computing SHAP values — {model_name} ({tag.upper()})')
    model_type = type(model).__name__
    try:
        if model_type == 'LogisticRegression':
            explainer = shap.LinearExplainer(model, X_te)
            shap_vals = explainer.shap_values(X_te)
        elif model_type in ['RandomForestClassifier','XGBClassifier']:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_te)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_te, 50))
            shap_vals = explainer.shap_values(X_te, nsamples=50)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
    except Exception as e:
        print(f'  [SHAP] Warning: {e}')
        print(f'  [SHAP] Falling back to KernelExplainer...')
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_te, 30))
        shap_vals = explainer.shap_values(X_te, nsamples=30)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    print(f'  [SHAP] Values shape: {shap_vals.shape}')'''

if old in content:
    content = content.replace(old, new)
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('FIXED: run_shap now auto-selects correct explainer')
else:
    print('WARNING: pattern not found — applying line-based fix')
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'explainer  = shap.TreeExplainer(model)' in line:
            print(f'  Found TreeExplainer at line {i+1}')
            break
