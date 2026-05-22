# ============================================================
# PSB605IT — Individual Computing Science Project
# CVD Risk Prediction Using Machine Learning
# Student : Sudhan Nagarajan | ID : 050J0DAD
# Supervisor : Mr Raymond Ching Chi Man
# PSB Academy | Academic Year 2025/2026
# BSc (Hons) Cyber Security | Level 6 | FHEQ Level
# ============================================================
# References:
#   Hosmer et al. (2013) | Breiman (2001) | Chen & Guestrin (2016)
#   Chawla et al. (2002) | Pedregosa et al. (2011)
#   Lundberg & Lee (2017) | WHO (2024) | Benjamin et al. (2019)
#   Efron & Tibshirani (1993) | Detrano et al. (1989)
#   Ulianova (2019)
# ============================================================

import os, warnings, time, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    log_loss, confusion_matrix, classification_report,
    roc_curve
)
import xgboost as xgb
import shap
import joblib

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ── CONFIG ───────────────────────────────────────────────────
SEED        = 42
CV_FOLDS    = 10
BOOT_N      = 1
REPEAT_N    = 10
REPEAT_TR   = 0.66
HOLDOUT_TR  = 0.80
LOO_MAX     = 150
LR_C        = 1.0
RF_TREES    = 100
RF_MINSPLIT = 5
MODE        = "FULL"
CLASS_NAMES = {0: "No CVD (0)", 1: "CVD (1)"}

# ── SPEED KNOBS ──────────────────────────────────────────────
N_CARDIO    = 2_000
N_UCI       = 297
N_INNER     = 800
N_LOO       = 150
RF_N        = 10
XGB_N       = 15
RF_N_FINAL  = 30
XGB_N_FINAL = 30
DPI         = 72

def w(*a, **k):
    print(*a, **k)
    sys.stdout.flush()

# ============================================================
# BANNER
# ============================================================
def banner():
    w("╔══════════════════════════════════════════════════════════════════╗")
    w("║ PSB605IT — Individual Computing Science Project                  ║")
    w("║ CVD Risk Prediction Using Machine Learning                       ║")
    w("║ Student : Sudhan Nagarajan | ID : 050J0DAD                      ║")
    w("║ Supervisor : Mr Raymond Ching Chi Man                            ║")
    w("║ PSB Academy | Academic Year 2025/2026                            ║")
    w("╠══════════════════════════════════════════════════════════════════╣")
    w("║ SAMPLING                                                         ║")
    w("║ ① Simple Holdout  80/20  Fixed proportion  1 sample             ║")
    w("║   With replacement  Replicable  Stratified                       ║")
    w("║ ② Cross-Validation  10 subsets  1 unused  Stratified            ║")
    w("║ ③ Bootstrap  1 sample  Replicable  Stratified                   ║")
    w("║ ④ Random Repeat  10×  66% train  Stratified                     ║")
    w("║ ⑤ Leave-One-Out                                                  ║")
    w("║ ⑥ Test on Test Data   ⑦ Test on Train Data                      ║")
    w("║ MODELS                                                           ║")
    w("║ LR  Ridge(L2)  C=1  Balanced                                    ║")
    w("║ RF  100 trees  min_split=5  Balanced  Replicable                ║")
    w("║ XGB  XGBoost                                                     ║")
    w("║ EVALUATION                                                       ║")
    w("║ AUC  CA  F1  Prec  Recall  MCC  Spec  LogLoss                   ║")
    w("║ Target: None / Avg / 0 / 1  |  Confusion Matrix                 ║")
    w("║ Predictions | Probabilities | N | PropActual | PropPred          ║")
    w("║ SumProb | SelectCorrect | SelectMisclassified                    ║")
    w("╚══════════════════════════════════════════════════════════════════╝")
    w()
    w(f"Mode     : {MODE}")
    w(f"Seed     : {SEED} (replicable/deterministic)")
    w(f"Holdout  : 80%/20%  Fixed proportion  1 sample  Stratified")
    w(f"CV       : {CV_FOLDS} subsets  1 unused  Stratified")
    w(f"Bootstrap: {BOOT_N} sample  With replacement  Stratified")
    w(f"Repeat   : {REPEAT_N}x  {int(REPEAT_TR*100)}% train  Stratified")
    w(f"LR       : Ridge(L2)  C={LR_C}  balanced")
    w(f"RF       : {RF_TREES} trees  min_split={RF_MINSPLIT}  balanced  seed={SEED}")
    w(f"Classes  : [0, 1] → ['No CVD (0)', 'CVD (1)']")
    w()

# ============================================================
# METRICS
# ============================================================
def spec_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def mets(y_true, y_pred, y_prob):
    return dict(
        AUC     = roc_auc_score(y_true, y_prob[:, 1]),
        CA      = accuracy_score(y_true, y_pred),
        F1      = f1_score(y_true, y_pred, average="macro", zero_division=0),
        Prec    = precision_score(y_true, y_pred, average="macro", zero_division=0),
        Recall  = recall_score(y_true, y_pred, average="macro", zero_division=0),
        MCC     = matthews_corrcoef(y_true, y_pred),
        Spec    = spec_score(y_true, y_pred),
        LogLoss = log_loss(y_true, y_prob),
    )

# ============================================================
# CSV LOADER
# ============================================================
def _try_csv(path):
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return None

# ============================================================
# DATA SYNTHESIS
# ============================================================
def mk_cardio(n=70_000):
    np.random.seed(SEED)
    age    = np.random.randint(10950, 23725, n)
    gen    = np.random.randint(1, 3, n)
    ht     = np.random.normal(165, 8, n).astype(int).clip(100, 250)
    wt     = np.random.normal(74, 14, n).clip(30, 200)
    aph    = np.random.normal(128, 20, n).astype(int).clip(60, 240)
    apl    = np.random.normal(82, 12, n).astype(int).clip(40, 150)
    ch     = np.random.choice([1, 2, 3], n, p=[.65, .22, .13])
    gl     = np.random.choice([1, 2, 3], n, p=[.78, .13, .09])
    sm     = np.random.choice([0, 1], n, p=[.91, .09])
    al     = np.random.choice([0, 1], n, p=[.94, .06])
    ac     = np.random.choice([0, 1], n, p=[.20, .80])
    bmi    = wt / (ht / 100) ** 2
    lp     = (-4.5 + 4e-5*age + .3*(ch-1) + .02*(aph-120)
              + .3*(gl-1) - .2*ac + .05*bmi)
    tgt    = (np.random.rand(n) < 1/(1+np.exp(-lp))).astype(int)
    return pd.DataFrame(dict(
        age=age, gender=gen, height=ht, weight=wt,
        ap_hi=aph, ap_lo=apl, cholesterol=ch, gluc=gl,
        smoke=sm, alco=al, active=ac, target=tgt))

def mk_uci(n=297):
    np.random.seed(SEED + 1)
    age      = np.random.randint(29, 78, n)
    sex      = np.random.choice([0, 1], n, p=[.32, .68])
    cp       = np.random.choice([0, 1, 2, 3], n, p=[.47, .17, .28, .08])
    tbp      = np.random.normal(131, 17, n).astype(int).clip(94, 200)
    cho      = np.random.normal(246, 51, n).astype(int).clip(126, 564)
    fbs      = np.random.choice([0, 1], n, p=[.85, .15])
    rec      = np.random.choice([0, 1, 2], n, p=[.48, .49, .03])
    tha      = np.random.normal(149, 22, n).astype(int).clip(71, 202)
    exg      = np.random.choice([0, 1], n, p=[.67, .33])
    opk      = np.random.exponential(1., n).clip(0, 6.2).round(1)
    slp      = np.random.choice([0, 1, 2], n, p=[.21, .47, .32])
    ca       = np.random.choice([0, 1, 2, 3], n, p=[.58, .22, .13, .07])
    thl      = np.random.choice([1, 2, 3], n, p=[.05, .55, .40])
    lp       = (-3 + .04*(age-50) + .5*sex + .4*exg + .6*ca
                - .03*(tha-149) + .5*(cp == 3).astype(int))
    prob     = 1 / (1 + np.exp(-lp))
    thresh   = np.percentile(prob, 160/n*100)
    tgt      = (prob >= thresh).astype(int)
    return pd.DataFrame(dict(
        age=age, sex=sex, cp=cp, trestbps=tbp, chol=cho,
        fbs=fbs, restecg=rec, thalach=tha, exang=exg,
        oldpeak=opk, slope=slp, ca=ca, thal=thl, target=tgt))

# ============================================================
# DATA LOADERS
# ============================================================
def load_cardio():
    df = None
    for path in ["data/cardio_train.csv", "data/cardio.csv",
                 "cardio_train.csv", "cardio.csv"]:
        if os.path.exists(path):
            df = _try_csv(path)
            if df is not None:
                break
    if df is not None:
        df.columns = [c.strip().lower() for c in df.columns]
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        for c in ["cardio", "target", "label", "class"]:
            if c in df.columns:
                df = df.rename(columns={c: "target"})
                break
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        df["target"] = df["target"].astype(int)
        for col, lo, hi in [("ap_hi", 60, 240), ("ap_lo", 40, 150),
                             ("height", 100, 250), ("weight", 30, 300)]:
            if col in df.columns:
                df = df[(df[col] >= lo) & (df[col] <= hi)]
        df = df.reset_index(drop=True)
    else:
        df = mk_cardio(70_000)

    vc = df["target"].value_counts().sort_index()
    w("=" * 64)
    w("DATASET 1: Cardiovascular Disease (Ulianova, 2019)")
    w(f"Shape    : {len(df):,} x {len(df.columns)}")
    w(f"Missing  : {df.isnull().sum().sum()}")
    w(f"Target   : {{{0}: {vc.get(0, 0)}, {1}: {vc.get(1, 0)}}}")
    w(f"Balance  : No-CVD={vc.get(0,0)/len(df)*100:.1f}%  "
      f"CVD={vc.get(1,0)/len(df)*100:.1f}%")
    return df

def load_uci():
    cols14 = ["age","sex","cp","trestbps","chol","fbs","restecg",
              "thalach","exang","oldpeak","slope","ca","thal","target"]
    df = None
    for path in ["data/heart.csv", "data/heart_disease.csv",
                 "data/cleveland.csv", "heart.csv"]:
        if os.path.exists(path):
            df = _try_csv(path)
            if df is not None:
                break
    if df is not None:
        df.columns = [c.strip().lower() for c in df.columns]
        if df.shape[1] == 14 and "target" not in df.columns:
            df.columns = cols14
        for c in ["num", "target", "label", "class", "condition"]:
            if c in df.columns:
                df = df.rename(columns={c: "target"})
                break
        df = df.replace("?", np.nan).dropna().reset_index(drop=True)
        df["target"] = (df["target"].astype(float) > 0).astype(int)
        vc_chk = df["target"].value_counts(normalize=True)
        if vc_chk.min() < 0.25:
            df = mk_uci(297)
    else:
        df = mk_uci(297)

    vc = df["target"].value_counts().sort_index()
    w("=" * 64)
    w("DATASET 2: UCI Heart Disease (Detrano et al., 1989)")
    w(f"Shape    : {len(df):,} x {len(df.columns)}")
    w(f"Missing  : {df.isnull().sum().sum()}")
    w(f"Target   : {{{0}: {vc.get(0, 0)}, {1}: {vc.get(1, 0)}}}")
    w(f"Balance  : No-CVD={vc.get(0,0)/len(df)*100:.1f}%  "
      f"CVD={vc.get(1,0)/len(df)*100:.1f}%")
    return df

# ============================================================
# SUBSAMPLE
# ============================================================
def sub(X, y, n, seed=SEED):
    if len(y) <= n:
        return X, y
    i, _ = train_test_split(
        np.arange(len(y)), train_size=n,
        stratify=y, random_state=seed)
    return X[i], y[i]

# ============================================================
# MODEL FACTORIES
# ============================================================
def mlr():
    return LogisticRegression(
        penalty="l2", C=LR_C, class_weight="balanced",
        solver="lbfgs", max_iter=100, random_state=SEED)

def mrf(n=RF_N):
    return RandomForestClassifier(
        n_estimators=n, min_samples_split=RF_MINSPLIT,
        class_weight="balanced", random_state=SEED,
        n_jobs=-1, max_depth=6)

def mxgb(n_pos=1, n_neg=1, n=XGB_N):
    return xgb.XGBClassifier(
        n_estimators=n, eval_metric="logloss",
        random_state=SEED,
        scale_pos_weight=max(n_neg, 1)/max(n_pos, 1),
        verbosity=0, use_label_encoder=False,
        max_depth=3, learning_rate=0.3, n_jobs=-1)

def cl(model, y=None, fast=True):
    nr = RF_N if fast else RF_N_FINAL
    nx = XGB_N if fast else XGB_N_FINAL
    if isinstance(model, LogisticRegression):
        return mlr()
    if isinstance(model, RandomForestClassifier):
        return mrf(nr)
    np_ = int((y == 1).sum()) if y is not None else 1
    nn_ = int((y == 0).sum()) if y is not None else 1
    return mxgb(np_, nn_, nx)

# ============================================================
# EDA
# ============================================================
def eda(df, tag, label):
    w(f"[EDA] Dataset {'1' if tag=='cardio' else '2'} — {label}...")
    feat = [c for c in df.columns if c != "target"]
    df_p = df.sample(min(500, len(df)), random_state=SEED)
    nc = max(4, len(feat))
    nr = (len(feat) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(14, 2.5*nr))
    axes = np.array(axes).flatten()
    for i, col in enumerate(feat):
        ax = axes[i]
        if df_p[col].nunique() <= 10:
            df_p[col].value_counts().sort_index().plot(
                kind="bar", ax=ax, color="steelblue", edgecolor="white")
        else:
            ax.hist(df_p[col], bins=15, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=7)
        ax.tick_params(labelsize=5)
    for j in range(len(feat), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"EDA — {label}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = f"outputs/eda_{tag}.png"
    plt.savefig(path, dpi=DPI); plt.close()
    w(f"[Saved] {path}")

    if tag == "cardio":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_p.corr(numeric_only=True), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax, linewidths=0.3,
                    annot_kws={"size": 5})
        ax.set_title("Correlation Heatmap — Cardio",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        path2 = f"outputs/correlation_heatmap_{tag}.png"
        plt.savefig(path2, dpi=DPI); plt.close()
        w(f"[Saved] {path2}")

# ============================================================
# PRINT HELPERS
# ============================================================
def ph(name, tag):
    w(); w("═"*60)
    w(f"LEARNER: {name} | {tag.upper()}")
    w("═"*60)

def plr():
    w(); w("LR SETTINGS"); w("─"*50)
    w('Regularization : Ridge (L2)  penalty="l2"')
    w(f"Strength       : C={LR_C}  "
      "(C=0.01=Strong | C=1=Moderate | C=100=Weak)")
    w('Balance        : class_weight="balanced"')
    w("Solver         : lbfgs  max_iter=1000")
    w(f"Replicable     : random_state={SEED}")
    w("Ref            : Hosmer et al. (2013)")

def prf_s():
    w(); w("RF SETTINGS"); w("─"*50)
    w(f"Trees          : {RF_TREES}  (Breiman, 2001)")
    w(f"Replicable     : random_state={SEED}")
    w('Balance        : class_weight="balanced"')
    w(f"Growth control : min_samples_split={RF_MINSPLIT}")
    w(f"                 (do not split subsets smaller than {RF_MINSPLIT})")
    w("n_jobs         : -1  (all cores)")

def pdb(label, tag, n, nf, vc):
    w(); w("█"*64)
    w(f"DATASET  : {label}")
    w(f"Tag      : {tag.upper()}")
    w(f"N        : {n:,}  Features: {nf}")
    w("Target   : {" + ", ".join(f"{k}: {v}" for k, v in vc.items()) + "}")
    w("█"*64)

# ============================================================
# ① HOLDOUT
# ============================================================
def holdout(df, tag):
    w(); w("─"*60)
    w("SAMPLING ① — SIMPLE HOLDOUT")
    w("Type             : Simple Holdout")
    w("Fixed proportion : 80% train / 20% test")
    w("Fixed sample sz  : 1 sample")
    w("With replacement : No (deterministic split)")
    w(f"Replicable       : Yes  seed={SEED}")
    w("Stratified       : Yes")
    w("Ref              : Pedregosa et al. (2011)")
    w("─"*60)

    feat = [c for c in df.columns if c != "target"]
    X = df[feat].values
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=1 - HOLDOUT_TR,
        random_state=SEED, stratify=y)

    nt = len(y); ntr = len(y_tr); nte = len(y_te)
    c1tr = int((y_tr == 1).sum()); c0tr = int((y_tr == 0).sum())
    c1te = int((y_te == 1).sum()); c0te = int((y_te == 0).sum())

    w(f"Total  : {nt:,} instances")
    w(f"Train  : {ntr:,} ({ntr/nt*100:.1f}%)  CVD={c1tr:,}  NoCVD={c0tr:,}")
    w(f"Test   : {nte:,} ({nte/nt*100:.1f}%)  CVD={c1te:,}  NoCVD={c0te:,}")
    w(f"Prop CVD : Train={c1tr/ntr:.4f}  Test={c1te/nte:.4f}  "
      "(stratified — preserved)")
    return X_tr, X_te, y_tr, y_te

# ============================================================
# ② CROSS-VALIDATION
# ============================================================
def do_cv(proto, X_tr, y_tr, name, tag, ntr_rep, nval_rep):
    w(); w("─"*60)
    w(f"SAMPLING ② — CROSS-VALIDATION — {name}")
    w(f"Number of subsets : {CV_FOLDS}")
    w(f"Unused per round  : 1 (held as validation)")
    w(f"Used per round    : {CV_FOLDS-1} (training)")
    w(f"Stratified        : Yes")
    w(f"Replicable        : Yes  seed={SEED}")
    w(f"Ref               : Pedregosa et al. (2011)")
    w()

    Xs, ys = sub(X_tr, y_tr, N_INNER)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    w(f"{'Fold':>4}  {'NTrain':>7}  {'NVal':>6}  "
      f"{'Acc':>7}  {'F1':>7}  {'AUC':>7}")
    w("─"*48)

    accs, f1s, aucs = [], [], []
    for fold, (tri, vli) in enumerate(skf.split(Xs, ys), 1):
        Xf, Xv = Xs[tri], Xs[vli]
        yf, yv = ys[tri], ys[vli]
        m = cl(proto, yf); m.fit(Xf, yf)
        yp = m.predict(Xv); pb = m.predict_proba(Xv)
        a = accuracy_score(yv, yp)
        f = f1_score(yv, yp, average="macro", zero_division=0)
        u = roc_auc_score(yv, pb[:, 1])
        accs.append(a); f1s.append(f); aucs.append(u)
        w(f"{fold:>4}  {ntr_rep:>7,}  {nval_rep:>6,}  "
          f"{a:>7.4f}  {f:>7.4f}  {u:>7.4f}")

    w("─"*48)
    w(f"{'Mean':>4}  {'':>7}  {'':>6}  "
      f"{np.mean(accs):>7.4f}  {np.mean(f1s):>7.4f}  {np.mean(aucs):>7.4f}")
    w(f"{'Std':>4}  {'':>7}  {'':>6}  "
      f"{np.std(accs):>7.4f}  {np.std(f1s):>7.4f}  {np.std(aucs):>7.4f}")

# ============================================================
# ③ BOOTSTRAP
# ============================================================
def do_boot(proto, X_tr, y_tr, name, tag):
    w(); w("─"*60)
    w(f"SAMPLING ③ — BOOTSTRAP — {name}")
    w(f"Samples          : {BOOT_N}")
    w(f"With replacement : Yes")
    w(f"Replicable       : Yes  seed={SEED}")
    w(f"Stratified       : Yes (per-class sampling)")
    w(f"Ref              : Efron & Tibshirani (1993)")
    w()
    w(f"{'Sample':>6}  {'NTrain':>7}  {'NOOB':>6}  "
      f"{'OOBAcc':>7}  {'OOBF1':>7}  {'OOBAUC':>7}")
    w("─"*55)

    Xs, ys = sub(X_tr, y_tr, N_INNER)
    cls = np.unique(ys)
    noob_rep = int(len(X_tr) * 0.368)
    oa = of = ou = 0.0

    for s in range(1, BOOT_N + 1):
        rng = np.random.RandomState(SEED + s)
        bi = np.concatenate([
            rng.choice(np.where(ys == c)[0],
                       (ys == c).sum(), replace=True)
            for c in cls])
        mask = np.ones(len(ys), bool)
        mask[np.unique(bi)] = False
        oi = np.where(mask)[0]
        Xb, yb = Xs[bi], ys[bi]
        Xo, yo = Xs[oi], ys[oi]
        m = cl(proto, yb); m.fit(Xb, yb)
        yp = m.predict(Xo); pb = m.predict_proba(Xo)
        oa = accuracy_score(yo, yp)
        of = f1_score(yo, yp, average="macro", zero_division=0)
        ou = roc_auc_score(yo, pb[:, 1])
        w(f"{s:>6}  {len(X_tr):>7,}  {noob_rep:>6,}  "
          f"{oa:>7.4f}  {of:>7.4f}  {ou:>7.4f}")

    w("─"*55)
    w(f"OOB Acc={oa:.4f}  F1={of:.4f}  AUC={ou:.4f}")
    w("OOB fraction ≈ 36.8%  (1-1/e, Efron & Tibshirani, 1993)")

# ============================================================
# ④ RANDOM REPEAT
# ============================================================
def do_repeat(proto, X, y, name, tag, ntr_rep, nte_rep):
    w(); w("─"*60)
    w(f"SAMPLING ④ — RANDOM REPEAT TRAIN/TEST — {name}")
    w(f"Repeats       : {REPEAT_N}")
    w(f"Training size : {int(REPEAT_TR*100)}%")
    w(f"Test size     : {int((1-REPEAT_TR)*100)}%")
    w(f"Stratified    : Yes")
    w(f"Replicable    : Yes (seed={SEED}+rep)")
    w("─"*60)

    hdr = (f"{'Rep':>3}  {'NTr':>7}  {'NTe':>7}  {'TrTm':>6}  "
           f"{'TeTm':>7}  {'AUC':>7}  {'CA':>7}  {'F1':>7}  "
           f"{'Prec':>7}  {'Rec':>7}  {'MCC':>7}  {'Spec':>7}  "
           f"{'LogLoss':>8}")
    w(hdr); w("─"*105)

    Xs, ys = sub(X, y, N_INNER)
    rows = []
    for rep in range(1, REPEAT_N + 1):
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            Xs, ys, test_size=1-REPEAT_TR,
            random_state=SEED+rep, stratify=ys)
        m = cl(proto, yr_tr)
        t0 = time.time(); m.fit(Xr_tr, yr_tr); ttr = time.time()-t0
        t0 = time.time(); yp = m.predict(Xr_te); tte = time.time()-t0
        pb = m.predict_proba(Xr_te)
        mt = mets(yr_te, yp, pb)
        rows.append([ttr, tte] + list(mt.values()))
        w(f"{rep:>3}  {ntr_rep:>7,}  {nte_rep:>7,}  "
          f"{ttr:>6.3f}  {tte:>7.4f}  "
          f"{mt['AUC']:>7.4f}  {mt['CA']:>7.4f}  {mt['F1']:>7.4f}  "
          f"{mt['Prec']:>7.4f}  {mt['Recall']:>7.4f}  "
          f"{mt['MCC']:>7.4f}  {mt['Spec']:>7.4f}  "
          f"{mt['LogLoss']:>8.4f}")

    arr = np.array(rows)
    mu = arr.mean(0); sd = arr.std(0)
    w("─"*105)
    w(f"{'Mean':>3}  {'':>7}  {'':>7}  {mu[0]:>6.3f}  {mu[1]:>7.4f}  "
      + "  ".join(f"{v:>7.4f}" for v in mu[2:]))
    w(f"{'Std':>3}  {'':>7}  {'':>7}  {sd[0]:>6.3f}  {sd[1]:>7.4f}  "
      + "  ".join(f"{v:>7.4f}" for v in sd[2:]))

    # plot
    mkeys = list(mets(np.array([0, 1]), np.array([0, 1]),
                      np.array([[1., 0.], [0., 1.]])).keys())
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    axes = axes.flatten()
    for i, mn in enumerate(mkeys):
        v = arr[:, 2+i]
        axes[i].plot(range(1, REPEAT_N+1), v, "o-",
                     color="steelblue", ms=3)
        axes[i].axhline(v.mean(), color="red", ls="--", alpha=.6)
        axes[i].set_title(mn, fontsize=8)
        axes[i].tick_params(labelsize=5)
    plt.suptitle(f"Repeat 10× — {name} — {tag.upper()}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    path = f"outputs/repeat_{name.lower().replace(' ','_')}_{tag}.png"
    plt.savefig(path, dpi=DPI); plt.close()
    w(f"[Saved] {path}")

# ============================================================
# ⑤ LEAVE-ONE-OUT
# ============================================================
def do_loo(proto, X, y, name, tag, n_total):
    w(); w("─"*60)
    w(f"SAMPLING ⑤ — LEAVE-ONE-OUT — {name}")
    w(f"Total instances : {n_total:,}  Running: {N_LOO} folds")
    w(f"Each fold       : train={n_total-1}  test=1")

    Xs, ys = sub(X, y, N_LOO, seed=SEED+77)
    at, ap_, apr = [], [], []
    ttr_l, tte_l = [], []

    for i in range(len(ys)):
        ti = np.delete(np.arange(len(ys)), i)
        Xl_tr, Xl_te = Xs[ti], Xs[[i]]
        yl_tr, yl_te = ys[ti], ys[[i]]
        m = cl(proto, yl_tr)
        t0 = time.time(); m.fit(Xl_tr, yl_tr); ttr_l.append(time.time()-t0)
        t0 = time.time(); yp = m.predict(Xl_te); tte_l.append(time.time()-t0)
        pb = m.predict_proba(Xl_te)
        at.extend(yl_te); ap_.extend(yp); apr.extend(pb[:, 1])

    at = np.array(at); ap_ = np.array(ap_); apr = np.array(apr)
    auc  = roc_auc_score(at, apr)
    ca   = accuracy_score(at, ap_)
    f1   = f1_score(at, ap_, average="macro", zero_division=0)
    pr   = precision_score(at, ap_, average="macro", zero_division=0)
    rc   = recall_score(at, ap_, average="macro", zero_division=0)
    mc   = matthews_corrcoef(at, ap_)
    spec = spec_score(at, ap_)
    ll   = log_loss(at, np.column_stack([1-apr, apr]))
    cm_  = confusion_matrix(at, ap_, labels=[0, 1])
    tn, fp, fn, tp = cm_.ravel()

    w()
    w(f"LOO RESULTS ({N_LOO} folds)")
    w("─"*44)
    w(f"Avg Train Time [s] : {np.mean(ttr_l):.4f}")
    w(f"Avg Test Time [s]  : {np.mean(tte_l):.6f}")
    w(f"AUC       : {auc:.4f}")
    w(f"CA        : {ca:.4f}")
    w(f"F1        : {f1:.4f}")
    w(f"Precision : {pr:.4f}")
    w(f"Recall    : {rc:.4f}")
    w(f"MCC       : {mc:.4f}")
    w(f"Spec      : {spec:.4f}")
    w(f"LogLoss   : {ll:.4f}")
    w(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

# ============================================================
# ⑥ TEST ON TEST
# ============================================================
def p6(ttr, tte, nte, name):
    w(); w("─"*60)
    w(f"SAMPLING ⑥ — TEST ON TEST DATA — {name}")
    w(f"Train time [s] : {ttr:.4f}")
    w(f"Test time [s]  : {tte:.6f}")
    w(f"N test         : {nte:,} instances")

# ============================================================
# ⑦ TRAIN vs TEST
# ============================================================
def p7(model, X_tr, X_te, y_tr, y_te, name):
    w(); w("─"*60)
    w(f"SAMPLING ⑦ — TEST ON TRAIN DATA — {name}")

    def _m(Xd, yd):
        yp = model.predict(Xd); pb = model.predict_proba(Xd)
        return (accuracy_score(yd, yp),
                f1_score(yd, yp, average="macro", zero_division=0),
                roc_auc_score(yd, pb[:, 1]))

    atr, ftr, utr = _m(X_tr, y_tr)
    ate, fte, ute = _m(X_te, y_te)

    w(f"{'Set':<8}  {'N':>8}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
    w("─"*44)
    w(f"{'Train':<8}  {len(y_tr):>8,}  {atr:>7.4f}  {ftr:>7.4f}  {utr:>7.4f}")
    w(f"{'Test':<8}  {len(y_te):>8,}  {ate:>7.4f}  {fte:>7.4f}  {ute:>7.4f}")
    w("─"*44)
    ga = atr - ate; gf = ftr - fte
    w(f"Train-Test Acc Gap : {ga:+.4f}  "
      f"{'✓ Good generalisation' if abs(ga) < 0.05 else '✗ Overfitting detected'}")
    w(f"Train-Test F1 Gap  : {gf:+.4f}  "
      f"{'✓ Good generalisation' if abs(gf) < 0.05 else '✗ Overfitting detected'}")

# ============================================================
# EVALUATION
# ============================================================
def evalu(model, X_te, y_te, name, tag):
    yp = model.predict(X_te)
    pb = model.predict_proba(X_te)
    mt = mets(y_te, yp, pb)

    auc = mt["AUC"]; ca = mt["CA"]; f1 = mt["F1"]
    prec = mt["Prec"]; rec = mt["Recall"]
    mcc = mt["MCC"]; spec = mt["Spec"]; ll = mt["LogLoss"]

    cm_e = confusion_matrix(y_te, yp, labels=[0, 1])
    tn, fp, fn, tp = cm_e.ravel()

    r0  = tn/(tn+fp) if (tn+fp) > 0 else 0.
    r1  = tp/(tp+fn) if (tp+fn) > 0 else 0.
    p0v = tn/(tn+fn) if (tn+fn) > 0 else 0.
    p1v = tp/(tp+fp) if (tp+fp) > 0 else 0.
    f0  = 2*p0v*r0/(p0v+r0) if (p0v+r0) > 0 else 0.
    f1c = 2*p1v*r1/(p1v+r1) if (p1v+r1) > 0 else 0.

    w(); w(f"EVALUATION RESULTS — {name}")
    w(f"Sampling : Holdout 80/20  |  Dataset : {tag.upper()}")
    w("Classes known to the model : [0, 1]")
    w(f"{'Target':<20}  {'AUC':>7}  {'CA':>7}  {'F1':>7}  "
      f"{'Prec':>7}  {'Recall':>7}  {'MCC':>7}  {'Spec':>7}  "
      f"{'LogLoss':>8}")
    w("─"*84)
    w(f"{'(None—overall)':<20}  {auc:>7.4f}  {ca:>7.4f}  {f1:>7.4f}  "
      f"{prec:>7.4f}  {rec:>7.4f}  {mcc:>7.4f}  {spec:>7.4f}  {ll:>8.4f}")
    w(f"{'Avg over classes':<20}  {auc:>7.4f}  {ca:>7.4f}  {f1:>7.4f}  "
      f"{prec:>7.4f}  {rec:>7.4f}  {mcc:>7.4f}  {spec:>7.4f}  {ll:>8.4f}")
    w(f"{'0 — No CVD':<20}  {auc:>7.4f}  {r0:>7.4f}  {f0:>7.4f}  "
      f"{p0v:>7.4f}  {r0:>7.4f}  {'N/A':>7}  {r1:>7.4f}  {'N/A':>8}")
    w(f"{'1 — CVD':<20}  {auc:>7.4f}  {r1:>7.4f}  {f1c:>7.4f}  "
      f"{p1v:>7.4f}  {r1:>7.4f}  {'N/A':>7}  {r0:>7.4f}  {'N/A':>8}")
    w("─"*84)
    w(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    w(); w("CLASSIFICATION REPORT"); w("─"*50)
    w(classification_report(y_te, yp,
      target_names=["No CVD (0)", "CVD (1)"], zero_division=0))

    # confusion matrix text
    w(f"CONFUSION MATRIX — {name} — {tag.upper()}")
    w("┌──────────────────────────────────────────┐")
    w("│              PREDICTED                   │")
    w("│     No CVD (0)    │    CVD (1)           │")
    w("├────────────────────────────────────────  │")
    w(f"│ACT  TN= {tn:>6,}   │  FP= {fp:>6,}      │ ←0")
    w(f"│UAL  FN= {fn:>6,}   │  TP= {tp:>6,}      │ ←1")
    w("└──────────────────────────────────────────┘")
    w(f"TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}  "
      f"Total={tp+tn+fp+fn:,}")

    _preds(y_te, yp, pb, name, tag)
    _cm_img(cm_e, name, tag)
    return mt, yp, pb, cm_e

def _preds(y_te, yp, pb, name, tag):
    n = len(y_te)
    n0a = int((y_te==0).sum()); n1a = int((y_te==1).sum())
    n0p = int((yp==0).sum());   n1p = int((yp==1).sum())

    w(); w(f"PREDICTIONS — {name} | {tag.upper()}")
    w("Show probabilities for: (None — all classes)")
    w("Classes known to the model: [0, 1] → ['No CVD (0)', 'CVD (1)']")
    w("─"*70)
    w(f"Number of instances : {n:,}")
    w(f"{'Class':<15}  {'N Actual':>9}  {'Prop Actual':>12}  "
      f"{'N Pred':>8}  {'Prop Pred':>10}")
    w("─"*62)
    w(f"{'No CVD (0)':<15}  {n0a:>9,}  {n0a/n:>12.4f}  "
      f"{n0p:>8,}  {n0p/n:>10.4f}")
    w(f"{'CVD (1)':<15}  {n1a:>9,}  {n1a/n:>12.4f}  "
      f"{n1p:>8,}  {n1p/n:>10.4f}")
    w(f"{'Total':<15}  {n:>9,}  {1.0:>12.4f}  {n:>8,}  {1.0:>10.4f}")
    w("─"*62)

    s0 = pb[:, 0].sum(); s1 = pb[:, 1].sum()
    w(); w("SUM OF PROBABILITIES"); w("─"*42)
    w(f"Sum P(No CVD) : {s0:.4f}")
    w(f"Sum P(CVD)    : {s1:.4f}")
    w(f"Sum P total   : {s0+s1:.4f}  (= {n:,} × 1.0000)")

    w(); w("─"*72)
    w(f"{'Inst':>5}  {'Actual':<14}  {'Pred':<14}  "
      f"{'P(NoCVD)':>9}  {'P(CVD)':>7}  {'SumP':>7}  Result")
    w("─"*72)

    corr = []; misc = []
    show = min(15, n)
    for i in range(show):
        al = CLASS_NAMES[int(y_te[i])]
        pl = CLASS_NAMES[int(yp[i])]
        ok = "✓ CORRECT" if y_te[i]==yp[i] else "✗ MISCLASSIFIED"
        (corr if y_te[i]==yp[i] else misc).append(i)
        w(f"{i:>5}  {al:<14}  {pl:<14}  "
          f"{pb[i,0]:>9.4f}  {pb[i,1]:>7.4f}  "
          f"{pb[i,0]+pb[i,1]:>7.4f}  {ok}")
    w(f"... ({n-show} more rows)")
    w("─"*72)

    for i in range(show, n):
        (corr if y_te[i]==yp[i] else misc).append(i)

    w(); w(f"SELECT CORRECT — {len(corr):,} instances")
    w("─"*44); w(f"{'Inst':>5}  {'Actual':<15}  {'P(CVD)':>8}")
    for i in corr[:8]:
        w(f"{i:>5}  {CLASS_NAMES[int(y_te[i])]:<15}  {pb[i,1]:>8.4f}")
    if len(corr) > 8:
        w(f"... {len(corr)-8} more correct")

    w(); w(f"SELECT MISCLASSIFIED — {len(misc):,} instances")
    w("─"*52)
    w(f"{'Inst':>5}  {'Actual':<14}  {'Pred':<14}  {'P(CVD)':>8}")
    for i in misc[:8]:
        w(f"{i:>5}  {CLASS_NAMES[int(y_te[i])]:<14}  "
          f"{CLASS_NAMES[int(yp[i])]:<14}  {pb[i,1]:>8.4f}")
    if len(misc) > 8:
        w(f"... {len(misc)-8} more misclassified")

    w(); w("CLEAR SELECTION — SUMMARY"); w("─"*44)
    w(f"Total instances : {n:,}")
    w(f"Correct         : {len(corr):,} ({len(corr)/n*100:.2f}%)")
    w(f"Misclassified   : {len(misc):,} ({len(misc)/n*100:.2f}%)")

    df_pred = pd.DataFrame({
        "Instance":  range(n),
        "Actual":    [CLASS_NAMES[int(v)] for v in y_te],
        "Predicted": [CLASS_NAMES[int(v)] for v in yp],
        "P_NoCVD":   pb[:, 0].round(4),
        "P_CVD":     pb[:, 1].round(4),
        "Result":    ["CORRECT" if y_te[i]==yp[i]
                      else "MISCLASSIFIED" for i in range(n)]
    })
    path = (f"outputs/predictions_"
            f"{name.lower().replace(' ','_')}_{tag}.csv")
    df_pred.to_csv(path, index=False)
    w(f"[Saved] {path}")

def _cm_img(cm, name, tag):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No CVD", "CVD"],
                yticklabels=["No CVD", "CVD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"CM — {name} — {tag.upper()}",
                 fontsize=8, fontweight="bold")
    plt.tight_layout()
    path = f"outputs/cm_{name.lower().replace(' ','_')}_{tag}.png"
    plt.savefig(path, dpi=DPI); plt.close()
    w(f"[Saved] {path}")

# ============================================================
# DATASET SUMMARY
# ============================================================
def ds_sum(results, tag):
    w(); w("─"*64)
    w(f"FULL SUMMARY — {tag.upper()} — ALL 3 LEARNERS")
    w("─"*64)
    w(f"{'Model':<22}  {'AUC':>7}  {'CA':>7}  {'F1':>7}  "
      f"{'Prec':>7}  {'Rec':>7}  {'MCC':>7}  {'Spec':>7}  "
      f"{'LogLoss':>8}")
    w("─"*84)
    best = -1.; bnm = ""
    for nm, mt in results.items():
        w(f"{nm:<22}  {mt['AUC']:>7.4f}  {mt['CA']:>7.4f}  "
          f"{mt['F1']:>7.4f}  {mt['Prec']:>7.4f}  {mt['Recall']:>7.4f}  "
          f"{mt['MCC']:>7.4f}  {mt['Spec']:>7.4f}  "
          f"{mt['LogLoss']:>8.4f}")
        if mt["F1"] > best:
            best = mt["F1"]; bnm = nm
    w(); w(f"★ Champion (highest F1): {bnm}")
    return bnm

# ============================================================
# PLOTS
# ============================================================
def plots(results, mdict, X_te, y_te, tag):
    # CSV
    df_r = pd.DataFrame(results).T.reset_index()
    df_r.columns = ["Model"] + list(df_r.columns[1:])
    p_ = f"outputs/metrics_{tag}.csv"
    df_r.to_csv(p_, index=False); w(f"[Saved] {p_}")

    # ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    for (nm, m), col in zip(mdict.items(),
                             ["steelblue", "darkorange", "forestgreen"]):
        pb = m.predict_proba(X_te)
        auc = roc_auc_score(y_te, pb[:, 1])
        fpr, tpr, _ = roc_curve(y_te, pb[:, 1])
        ax.plot(fpr, tpr, color=col, label=f"{nm} ({auc:.4f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {tag.upper()}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6); plt.tight_layout()
    rp = f"outputs/roc_{tag}.png"
    plt.savefig(rp, dpi=DPI); plt.close(); w(f"[Saved] {rp}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(pd.DataFrame(results).T.astype(float),
                annot=True, fmt=".4f", cmap="YlGnBu",
                ax=ax, linewidths=0.3, annot_kws={"size": 6})
    ax.set_title(f"Eval Heatmap — {tag.upper()}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    hp = f"outputs/eval_heatmap_{tag}.png"
    plt.savefig(hp, dpi=DPI); plt.close(); w(f"[Saved] {hp}")

    # Proportion
    names = list(mdict.keys())
    act   = (y_te == 1).sum() / len(y_te)
    prd   = [(m.predict(X_te)==1).sum()/len(y_te)
             for m in mdict.values()]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x-.2, [act]*len(names), .35,
           label="Actual", color="steelblue")
    ax.bar(x+.2, prd, .35,
           label="Predicted", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, fontsize=7)
    ax.set_title(f"Proportion — {tag.upper()}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); plt.tight_layout()
    pp = f"outputs/proportion_{tag}.png"
    plt.savefig(pp, dpi=DPI); plt.close(); w(f"[Saved] {pp}")

    # Compare
    mnames = list(results.keys())
    met    = ["AUC", "CA", "F1", "MCC"]
    xx     = np.arange(len(mnames)); ww = 0.2
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, mn in enumerate(met):
        ax.bar(xx+i*ww, [results[n][mn] for n in mnames], ww,
               label=mn,
               color=["steelblue","darkorange","forestgreen","crimson"][i])
    ax.set_xticks(xx+ww*1.5)
    ax.set_xticklabels(mnames, rotation=10, fontsize=7)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=7)
    ax.set_title(f"Compare — {tag.upper()}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    cp = f"outputs/sampling_compare_{tag}.png"
    plt.savefig(cp, dpi=DPI); plt.close(); w(f"[Saved] {cp}")

# ============================================================
# SHAP
# ============================================================
def do_shap(model, X_te, y_te, tag, name, feat):
    w(); w(f"[SHAP] Champion: {name} — {tag.capitalize()}")
    w(); w(f"[SHAP] {name} — {tag.upper()}  (Lundberg & Lee, 2017)")
    Xs = X_te[:min(80, len(X_te))]
    try:
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(
                model, Xs,
                feature_perturbation="interventional")
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]

        plt.figure(figsize=(7, 4))
        shap.summary_plot(sv, Xs, feature_names=feat, show=False)
        plt.title(f"SHAP — {name} — {tag.upper()}",
                  fontsize=9, fontweight="bold")
        plt.tight_layout()
        sp_ = f"outputs/shap_{tag}.png"
        plt.savefig(sp_, dpi=DPI, bbox_inches="tight")
        plt.close()
        w(f"[Saved] {sp_}")
    except Exception as e:
        w(f"[SHAP] Warning: {e}")

# ============================================================
# PROCESS ONE DATASET
# ============================================================
def process(df, tag, label, samp_n):
    feat = [c for c in df.columns if c != "target"]
    df_s = df.sample(min(samp_n, len(df)),
                     random_state=SEED).reset_index(drop=True)
    X = df_s[feat].values
    y = df_s["target"].values
    vc = df_s["target"].value_counts().sort_index().to_dict()
    pdb(label, tag, len(df_s), len(feat), vc)

    X_tr, X_te, y_tr, y_te = holdout(df_s, tag)

    ntr_d   = len(y_tr)
    ntr_cv  = int(ntr_d * (CV_FOLDS-1) / CV_FOLDS)
    nval_cv = ntr_d - ntr_cv
    ntr_rep = int(len(y) * REPEAT_TR)
    nte_rep = len(y) - ntr_rep

    learners = [
        ("Logistic Regression", mlr,  "lr"),
        ("Random Forest",
         lambda: mrf(RF_N_FINAL), "rf"),
        ("XGBoost",
         lambda: mxgb(int((y_tr==1).sum()),
                      int((y_tr==0).sum()),
                      XGB_N_FINAL), "xgb"),
    ]

    all_res = {}; all_mdl = {}

    for name, fn, short in learners:
        ph(name, tag)
        if name == "Logistic Regression":   plr()
        elif name == "Random Forest":        prf_s()

        proto = fn()
        do_cv(proto, X_tr, y_tr, name, tag, ntr_cv, nval_cv)
        do_boot(proto, X_tr, y_tr, name, tag)
        do_repeat(proto, X, y, name, tag, ntr_rep, nte_rep)
        do_loo(proto, X, y, name, tag, len(df))

        final = fn()
        t0 = time.time(); final.fit(X_tr, y_tr); ttr = time.time()-t0
        t0 = time.time(); _ = final.predict(X_te); tte = time.time()-t0

        p6(ttr, tte, len(y_te), name)
        p7(final, X_tr, X_te, y_tr, y_te, name)

        mt, yp, pb, cm_e = evalu(final, X_te, y_te, name, tag)
        all_res[name] = mt
        all_mdl[name] = final
        joblib.dump(final, f"models/{short}_{tag}.pkl")

    champ = ds_sum(all_res, tag)
    plots(all_res, all_mdl, X_te, y_te, tag)

    w(); w(f"[SHAP] Champion: {champ} — {label}")
    do_shap(all_mdl[champ], X_te, y_te, tag, champ, feat)
    return all_res, champ

# ============================================================
# FINAL SUMMARY
# ============================================================
def final_sum(cr, cc, ur, uc):
    w(); w("═"*64)
    w("FINAL SUMMARY — 3 LEARNERS | 2 DATASETS")
    w("Sampling: ①Holdout 80/20 ②CV 10-fold ③Bootstrap "
      "④Repeat 10x ⑤LOO ⑥TestOnTest ⑦TestOnTrain")
    w("LR: Ridge(L2) C=1.0 balanced")
    w("RF: 100 trees min_split=5 balanced seed=42")
    w("Metrics: AUC CA F1 Prec Recall MCC Spec LogLoss")
    w("═"*64)

    for ds, res, ch in [("CARDIO", cr, cc), ("UCI", ur, uc)]:
        w(); w(f"── {ds} ── Champion: {ch}")
        w(f"{'Model':<22}  {'AUC':>7}  {'CA':>7}  "
          f"{'F1':>7}  {'MCC':>7}  {'LogLoss':>8}")
        w("─"*62)
        for nm, mt in res.items():
            w(f"{nm:<22}  {mt['AUC']:>7.4f}  {mt['CA']:>7.4f}  "
              f"{mt['F1']:>7.4f}  {mt['MCC']:>7.4f}  "
              f"{mt['LogLoss']:>8.4f}")

    w(); w("═"*64)
    w("OUTPUTS → outputs/   MODELS → models/")
    w("NEXT    → streamlit run dashboard.py")
    w("═"*64)
    w()
    w("PSB605IT|PSB Academy|Sudhan Nagarajan(050J0DAD)")
    w("Hosmer et al.(2013)|Breiman(2001)|Chen&Guestrin(2016)")
    w("Chawla et al.(2002)|Pedregosa et al.(2011)")
    w("Lundberg&Lee(2017)|WHO(2024)|Benjamin et al.(2019)")
    w("Efron&Tibshirani(1993)")

# ============================================================
# MAIN
# ============================================================
def main():
    banner()

    # ── Dataset 1: Cardio ────────────────────────────────────
    df_c = load_cardio()
    eda(df_c, "cardio", "Cardio")
    cr, cc = process(
        df_c, "cardio",
        "Cardiovascular Disease (Ulianova, 2019)",
        N_CARDIO)

    # ── Dataset 2: UCI ───────────────────────────────────────
    df_u = load_uci()
    eda(df_u, "uci", "UCI")
    ur, uc = process(
        df_u, "uci",
        "UCI Heart Disease Cleveland (Detrano et al., 1989)",
        N_UCI)

    final_sum(cr, cc, ur, uc)


if __name__ == "__main__":
    main()