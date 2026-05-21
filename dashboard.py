# ================================================================
# dashboard.py — CVD Risk Prediction | PSB605IT
# Sudhan Nagarajan (050J0DAD) | PSB Academy 2025/2026
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CVD Risk Prediction | PSB605IT",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box;}
body{background:#050a1a;}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#020817 0%,#0a1628 50%,#020817 100%);
  border-right:1px solid #1e3a5f;
}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}

/* Main background */
.stApp{background:linear-gradient(135deg,#050a1a 0%,#0a1628 50%,#050a1a 100%);}

/* Glassmorphism cards */
.glass{
  background:rgba(255,255,255,0.04);
  backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:16px;padding:20px;margin-bottom:12px;
  box-shadow:0 8px 32px rgba(0,0,0,0.4);
}

/* Hero banner */
.hero{
  background:linear-gradient(135deg,#0a1628,#1a0a2e,#0a1628);
  border:1px solid rgba(124,58,237,0.3);
  border-radius:20px;padding:40px 30px;text-align:center;
  margin-bottom:24px;position:relative;overflow:hidden;
  box-shadow:0 0 60px rgba(124,58,237,0.2);
}
.hero::before{
  content:'';position:absolute;top:-50%;left:-50%;
  width:200%;height:200%;
  background:radial-gradient(ellipse at center,
    rgba(124,58,237,0.1) 0%,transparent 60%);
  animation:pulse 4s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:.5;transform:scale(1);}50%{opacity:1;transform:scale(1.05);}}

/* Stat cards */
.stat-card{
  background:linear-gradient(135deg,rgba(30,41,59,0.8),rgba(15,23,42,0.9));
  border:1px solid rgba(255,255,255,0.1);border-radius:14px;
  padding:20px 14px;text-align:center;
  box-shadow:0 4px 24px rgba(0,0,0,0.5);
  transition:transform .2s,box-shadow .2s;
}
.stat-card:hover{transform:translateY(-4px);box-shadow:0 8px 32px rgba(0,0,0,0.6);}
.stat-num{font-size:2.2rem;font-weight:900;line-height:1;}
.stat-lbl{font-size:.72rem;color:#64748b;margin-top:6px;}

/* Section header */
.sec-hdr{
  background:linear-gradient(90deg,#1e40af,#7c3aed,#db2777);
  border-radius:10px;padding:10px 18px;color:#fff !important;
  font-weight:700;font-size:.95rem;margin:16px 0 10px;
  box-shadow:0 4px 15px rgba(124,58,237,0.4);
}

/* Risk banners */
.risk-high{
  background:linear-gradient(135deg,rgba(127,29,29,0.9),rgba(153,27,27,0.9));
  border:2px solid #ef4444;border-radius:18px;
  padding:28px;text-align:center;color:#fff;
  box-shadow:0 0 40px rgba(239,68,68,0.4);
}
.risk-low{
  background:linear-gradient(135deg,rgba(20,83,45,0.9),rgba(22,101,52,0.9));
  border:2px solid #22c55e;border-radius:18px;
  padding:28px;text-align:center;color:#fff;
  box-shadow:0 0 40px rgba(34,197,94,0.4);
}
.risk-med{
  background:linear-gradient(135deg,rgba(120,53,15,0.9),rgba(146,64,14,0.9));
  border:2px solid #f59e0b;border-radius:18px;
  padding:28px;text-align:center;color:#fff;
  box-shadow:0 0 40px rgba(245,158,11,0.4);
}

/* Model cards */
.model-card{
  background:rgba(30,41,59,0.6);
  border:2px solid transparent;border-radius:14px;
  padding:18px;text-align:center;
  transition:all .3s;
}
.model-card:hover{transform:translateY(-3px);}

/* Metric pill */
.pill{
  display:inline-block;background:rgba(124,58,237,0.2);
  border:1px solid rgba(124,58,237,0.4);border-radius:20px;
  padding:4px 12px;font-size:.75rem;color:#a78bfa;margin:2px;
}

/* Table */
.stDataFrame{border-radius:10px;overflow:hidden;}

/* Buttons */
.stButton>button{
  background:linear-gradient(90deg,#1e40af,#7c3aed);
  color:#fff;border:none;border-radius:12px;
  padding:14px 28px;font-weight:700;font-size:1rem;
  width:100%;transition:all .3s;
  box-shadow:0 4px 15px rgba(124,58,237,0.4);
}
.stButton>button:hover{
  background:linear-gradient(90deg,#1d4ed8,#6d28d9);
  transform:translateY(-2px);
  box-shadow:0 8px 25px rgba(124,58,237,0.6);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(30,41,59,0.6);border-radius:12px;
  padding:4px;gap:4px;
  border:1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px;color:#94a3b8;font-weight:500;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(90deg,#1e40af,#7c3aed) !important;
  color:#fff !important;
}

/* Info box */
.ibox{
  background:rgba(30,41,59,0.5);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:10px;padding:14px 16px;margin-bottom:8px;
  color:#e2e8f0;
}

/* KV row */
.kvr{
  display:flex;justify-content:space-between;align-items:center;
  padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.06);
}
.kvr span:first-child{color:#64748b;font-size:.8rem;}
.kvr span:last-child{font-weight:600;font-size:.82rem;}

/* Footer */
.foot{
  background:rgba(15,23,42,0.8);
  border:1px solid rgba(255,255,255,0.06);
  border-radius:12px;padding:16px;text-align:center;
  color:#334155;font-size:.7rem;margin-top:24px;
}

/* Progress bars */
.stProgress>div>div>div{
  background:linear-gradient(90deg,#1e40af,#7c3aed);
  border-radius:10px;
}

/* Metric */
[data-testid="metric-container"]{
  background:rgba(30,41,59,0.5);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"

# ================================================================
# HELPER FUNCTIONS
# ================================================================
@st.cache_resource
def load_model(path):
    try: return joblib.load(path)
    except: return None

@st.cache_data
def load_csv(path):
    try: return pd.read_csv(path)
    except: return None

def mpath(name, tag):
    return os.path.join(MODELS_DIR, f"{name}_{tag}.pkl")

def safe_img(path, cap=""):
    if os.path.exists(path):
        st.image(path, caption=cap, use_container_width=True)
    else:
        st.markdown(f"""
        <div class='ibox' style='text-align:center;padding:30px;color:#475569'>
          📊 Image not found:<br><code>{path}</code><br>
          Run <code>python main.py</code> first
        </div>""", unsafe_allow_html=True)

def sec(title, emoji=""):
    st.markdown(f"<div class='sec-hdr'>{emoji} {title}</div>",
                unsafe_allow_html=True)

def kv(label, value, color="#e2e8f0"):
    st.markdown(
        f"<div class='kvr'><span>{label}</span>"
        f"<span style='color:{color}'>{value}</span></div>",
        unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class='foot'>
      ❤️ CVD Risk Prediction · PSB605IT ·
      Sudhan Nagarajan (050J0DAD) · PSB Academy 2025/2026 ·
      Supervisor: Mr Raymond Ching Chi Man<br>
      References: Hosmer et al.(2013) · Breiman(2001) ·
      Chen&Guestrin(2016) · Lundberg&Lee(2017) ·
      Pedregosa et al.(2011) · Efron&Tibshirani(1993) ·
      WHO(2024) · Benjamin et al.(2019)
    </div>""", unsafe_allow_html=True)

def load_and_predict(features, tag, thresh=0.5):
    results = {}
    for mkey, mname in [
        ("logistic_regression","Logistic Regression"),
        ("random_forest","Random Forest"),
        ("xgboost","XGBoost"),
    ]:
        mdl = load_model(mpath(mkey, tag))
        if mdl is not None:
            proba = mdl.predict_proba(features)[0]
            results[mname] = {
                "proba": proba,
                "pred":  1 if proba[1] >= thresh else 0
            }
    return results

def show_risk_banner(results, thresh=0.5):
    if not results:
        st.error("No models loaded. Run python main.py first.")
        return
    avg_p = float(np.mean([v["proba"][1] for v in results.values()]))
    votes = sum(v["pred"] for v in results.values())
    pct   = avg_p * 100

    if votes >= 2:
        banner,icon,lbl = "risk-high","⚠️","HIGH CVD RISK DETECTED"
        advice = f"{votes}/3 models predict CVD — consult a cardiologist immediately"
    elif avg_p >= 0.35:
        banner,icon,lbl = "risk-med","🟡","MODERATE CVD RISK"
        advice = "Some risk factors present — lifestyle review recommended"
    else:
        banner,icon,lbl = "risk-low","✅","LOW CVD RISK"
        advice = f"{3-votes}/3 models predict No CVD — maintain healthy lifestyle"

    st.markdown(f"""
    <div class='{banner}'>
      <div style='font-size:3.5rem;margin-bottom:8px'>{icon}</div>
      <h2 style='font-size:1.8rem;font-weight:900;margin:0'>{lbl}</h2>
      <p style='font-size:1.1rem;margin:10px 0 4px'>
        Ensemble CVD Probability:
        <strong style='font-size:1.5rem'>{pct:.1f}%</strong>
      </p>
      <p style='font-size:.85rem;opacity:.85;margin:0'>
        Threshold: {thresh:.0%} · {advice}
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Per-model cards
    sec("Individual Model Predictions", "🤖")
    cols = st.columns(len(results))
    for col,(mname,res) in zip(cols,results.items()):
        p_cvd   = float(res["proba"][1])
        p_nocvd = float(res["proba"][0])
        pred    = res["pred"]
        clr     = "#ef4444" if pred==1 else "#22c55e"
        lbl2    = "⚠️ CVD" if pred==1 else "✅ No CVD"
        with col:
            st.markdown(f"""
            <div class='model-card' style='border-color:{clr}'>
              <div style='color:#64748b;font-size:.72rem;margin-bottom:6px'>
                {mname}</div>
              <div style='color:{clr};font-size:1.6rem;font-weight:900'>
                {lbl2}</div>
              <div style='color:#e2e8f0;font-size:.9rem;margin:8px 0 4px'>
                P(CVD) = <strong>{p_cvd:.4f}</strong></div>
              <div style='color:#64748b;font-size:.78rem'>
                P(No CVD) = {p_nocvd:.4f}</div>
              <div style='color:#94a3b8;font-size:.72rem;margin-top:4px'>
                Confidence: {max(p_cvd,p_nocvd)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            st.progress(float(p_cvd))

    # Probability chart
    st.markdown("---")
    sec("Probability Comparison — All Models", "📊")
    names   = list(results.keys())
    p_cvds  = [float(results[n]["proba"][1]) for n in names]
    p_nocvds= [float(results[n]["proba"][0]) for n in names]

    fig,ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor("#050a1a")
    ax.set_facecolor("#0a1628")
    x=np.arange(len(names)); w=0.35
    bars1=ax.bar(x-w/2,p_nocvds,w,label="P(No CVD)",
                 color="#22c55e",alpha=0.85,
                 edgecolor="rgba(0,0,0,0)")
    bars2=ax.bar(x+w/2,p_cvds,w,label="P(CVD)",
                 color="#ef4444",alpha=0.85)
    ax.axhline(thresh,color="#f59e0b",lw=2,ls="--",
               label=f"Threshold = {thresh:.0%}",alpha=0.8)
    for bar,val in zip(bars1,p_nocvds):
        ax.text(bar.get_x()+bar.get_width()/2,val+.01,
                f"{val:.3f}",ha="center",va="bottom",
                color="#e2e8f0",fontsize=9)
    for bar,val in zip(bars2,p_cvds):
        ax.text(bar.get_x()+bar.get_width()/2,val+.01,
                f"{val:.3f}",ha="center",va="bottom",
                color="#e2e8f0",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names,color="#94a3b8",fontsize=10)
    ax.set_ylim(0,1.15); ax.set_ylabel("Probability",color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("rgba(255,255,255,0.1)")
    ax.legend(facecolor="#0a1628",labelcolor="#e2e8f0",fontsize=9)
    ax.set_title("Model Probability Breakdown",color="#e2e8f0",
                 fontsize=12,fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

def mpl_dark_fig(w=10,h=5):
    fig,ax = plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor("#050a1a")
    ax.set_facecolor("#0a1628")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("rgba(255,255,255,0.1)")
    return fig,ax

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-size:3rem;
           filter:drop-shadow(0 0 20px rgba(239,68,68,0.8))'>❤️</div>
      <div style='font-size:1.1rem;font-weight:800;color:#e2e8f0;
           background:linear-gradient(90deg,#60a5fa,#a78bfa);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        CVD Risk AI</div>
      <div style='font-size:.7rem;color:#334155'>
        PSB605IT · PSB Academy</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.08);margin:8px 0'>
    """, unsafe_allow_html=True)

    PAGE = st.radio("Navigation", [
        "🏠 Home",
        "🩺 Patient Predictor",
        "📊 Dataset Explorer",
        "🤖 Model Performance",
        "📈 Sampling Methods",
        "🔍 SHAP Explainability",
        "📋 Predictions Viewer",
        "⚖️ Model Comparison",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.08);margin:8px 0'>
    """, unsafe_allow_html=True)

    DATASET = st.selectbox("🗄️ Active Dataset", [
        "CARDIO — Ulianova (2019)",
        "UCI — Detrano (1989)"
    ])
    TAG = "cardio" if "CARDIO" in DATASET else "uci"

    MODEL_SEL = st.selectbox("🤖 Active Model", [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ])
    MODEL_SLUG = {
        "Logistic Regression":"logistic_regression",
        "Random Forest":"random_forest",
        "XGBoost":"xgboost"
    }[MODEL_SEL]
    MODEL_SHORT = {
        "Logistic Regression":"lr",
        "Random Forest":"rf",
        "XGBoost":"xgb"
    }[MODEL_SEL]

    THRESH = st.slider("🎯 Decision Threshold",
        0.10,0.90,0.50,0.01)

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.08);margin:8px 0'>
    <div style='font-size:.65rem;color:#1e293b;text-align:center'>
      Sudhan Nagarajan · 050J0DAD<br>
      Mr Raymond Ching Chi Man<br>PSB Academy 2025/2026
    </div>""", unsafe_allow_html=True)

# ================================================================
# PAGE — HOME
# ================================================================
if PAGE == "🏠 Home":
    st.markdown("""
    <div class='hero'>
      <div style='font-size:4rem;margin-bottom:12px;
           filter:drop-shadow(0 0 30px rgba(239,68,68,0.8))'>
        ❤️ 🤖 🔬
      </div>
      <h1 style='color:#e2e8f0;font-size:2.5rem;font-weight:900;
           margin:0;background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        CVD Risk Prediction
      </h1>
      <h2 style='color:#a78bfa;font-size:1.1rem;font-weight:600;margin:8px 0'>
        Using Machine Learning · PSB605IT
      </h2>
      <p style='color:#64748b;font-size:.9rem;max-width:600px;margin:10px auto'>
        Compare Logistic Regression, Random Forest and XGBoost for
        cardiovascular disease risk prediction using 7 sampling strategies
        across 2 clinical datasets.
      </p>
      <div style='margin-top:14px'>
        <span class='pill'>🩺 2 Datasets</span>
        <span class='pill'>🤖 3 ML Models</span>
        <span class='pill'>🔬 7 Sampling Methods</span>
        <span class='pill'>📊 8 Metrics</span>
        <span class='pill'>🧠 SHAP Explainability</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl,clr) in zip(
        [c1,c2,c3,c4,c5],[
        ("70,297","Total Patients","#3b82f6"),
        ("3","ML Models","#7c3aed"),
        ("7","Sampling Methods","#06b6d4"),
        ("8","Eval Metrics","#10b981"),
        ("2","Datasets","#f59e0b"),
    ]):
        col.markdown(f"""
        <div class='stat-card'>
          <div class='stat-num' style='color:{clr}'>{val}</div>
          <div class='stat-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sec("Champion Results from main.py", "🏆")

    r1,r2 = st.columns(2)
    with r1:
        st.markdown("""
        <div class='glass'>
          <div style='color:#64748b;font-size:.8rem;margin-bottom:8px'>
            🫀 CARDIO Dataset (Ulianova, 2019)</div>
        """, unsafe_allow_html=True)
        df_c = pd.DataFrame({
            "Model":["Logistic Regression","Random Forest","⭐ XGBoost"],
            "AUC":[0.7489,0.7776,0.7754],
            "F1":[0.7002,0.6985,0.6996],
            "MCC":[0.4060,0.3999,0.3994],
            "Champion":["★","",""],
        })
        st.dataframe(df_c.style.highlight_max(
            subset=["AUC","F1","MCC"],color="#14532d"),
            use_container_width=True,hide_index=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class='glass'>
          <div style='color:#64748b;font-size:.8rem;margin-bottom:8px'>
            🔬 UCI Dataset (Detrano et al., 1989)</div>
        """, unsafe_allow_html=True)
        df_u = pd.DataFrame({
            "Model":["⭐ Logistic Regression","Random Forest","XGBoost"],
            "AUC":[0.9989,0.9408,0.9609],
            "F1":[0.9665,0.8490,0.8825],
            "MCC":[0.9330,0.6984,0.7655],
            "Champion":["★","",""],
        })
        st.dataframe(df_u.style.highlight_max(
            subset=["AUC","F1","MCC"],color="#14532d"),
            use_container_width=True,hide_index=True)
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("---")
    sec("ML Pipeline Overview", "⚙️")
    p1,p2,p3,p4 = st.columns(4)
    for col,(n,icon,title,body) in zip([p1,p2,p3,p4],[
        ("01","📥","Data Ingestion",
         "2 datasets\n70,297 patients\n27 features"),
        ("02","🔬","EDA & Preprocess",
         "Missing check\nOutlier removal\nStratified split"),
        ("03","🤖","Model Training",
         "LR · RF · XGBoost\n7 sampling strategies\nSeed=42"),
        ("04","📊","Evaluation",
         "AUC·F1·MCC\nSHAP explainability\nConfusion Matrix"),
    ]):
        col.markdown(f"""
        <div class='glass' style='text-align:center;min-height:160px'>
          <div style='color:rgba(124,58,237,0.6);font-size:.7rem;
               font-weight:700;letter-spacing:2px'>{n}</div>
          <div style='font-size:2rem;margin:6px 0'>{icon}</div>
          <div style='color:#e2e8f0;font-weight:700;font-size:.9rem'>
            {title}</div>
          <div style='color:#475569;font-size:.73rem;margin-top:6px;
               white-space:pre-line'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sec("Sampling Methods Overview", "🔬")
    sm_cols = st.columns(4)
    methods = [
        ("①","Simple Holdout","80/20 Fixed proportion\n1 sample · Stratified","#3b82f6"),
        ("②","Cross-Validation","10 subsets · 1 unused\nStratified · seed=42","#7c3aed"),
        ("③","Bootstrap","1 sample · With replacement\nOOB≈36.8%","#06b6d4"),
        ("④","Random Repeat","10× · 66/34 split\nStratified","#10b981"),
    ]
    for col,(n,m,d,c) in zip(sm_cols,methods):
        col.markdown(f"""
        <div class='glass' style='border-left:3px solid {c};
             min-height:120px'>
          <div style='color:{c};font-weight:900;font-size:1.1rem'>{n}</div>
          <div style='color:#e2e8f0;font-weight:600;font-size:.85rem;
               margin:4px 0'>{m}</div>
          <div style='color:#475569;font-size:.73rem;
               white-space:pre-line'>{d}</div>
        </div>""", unsafe_allow_html=True)

    sm2 = st.columns(3)
    methods2 = [
        ("⑤","Leave-One-Out","N folds (150 max)\nMax data usage","#f59e0b"),
        ("⑥","Test on Test Data","Final holdout evaluation\nTrue generalisation","#ef4444"),
        ("⑦","Test on Train Data","Train eval\nOverfitting detection","#ec4899"),
    ]
    for col,(n,m,d,c) in zip(sm2,methods2):
        col.markdown(f"""
        <div class='glass' style='border-left:3px solid {c};
             min-height:110px'>
          <div style='color:{c};font-weight:900;font-size:1.1rem'>{n}</div>
          <div style='color:#e2e8f0;font-weight:600;font-size:.85rem;
               margin:4px 0'>{m}</div>
          <div style='color:#475569;font-size:.73rem;
               white-space:pre-line'>{d}</div>
        </div>""", unsafe_allow_html=True)

    footer()

# ================================================================
# PAGE — PATIENT PREDICTOR
# ================================================================
elif PAGE == "🩺 Patient Predictor":
    st.markdown("""
    <div class='hero' style='padding:28px 24px'>
      <h2 style='color:#e2e8f0;font-size:1.8rem;font-weight:800;margin:0'>
        🩺 Patient CVD Risk Predictor
      </h2>
      <p style='color:#64748b;margin:8px 0 0'>
        Enter patient vitals below — all 3 trained models
        predict CVD risk instantly with ensemble voting
      </p>
    </div>""", unsafe_allow_html=True)

    ds = st.radio("Feature Set",
        ["🫀 CARDIO (Ulianova, 2019)",
         "🔬 UCI Heart Disease (Detrano, 1989)"],
        horizontal=True)
    pred_tag = "cardio" if "CARDIO" in ds else "uci"

    st.markdown(f"""
    <div class='ibox'>
      <b style='color:#7c3aed'>Decision Threshold:</b>
      <span style='color:#e2e8f0'> {THRESH:.0%}</span>
      <span style='color:#475569;font-size:.8rem'>
       · Adjustable in sidebar</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if pred_tag == "cardio":
        sec("CARDIO — Patient Health Metrics", "🫀")
        with st.form("cardio_pred"):
            st.markdown("**👤 Demographics & Anthropometry**")
            d1,d2,d3,d4 = st.columns(4)
            with d1:
                age_yr = st.number_input("Age (years)",1,100,50,1)
            with d2:
                gender = st.selectbox("Gender",[1,2],
                    format_func=lambda x:"♀ Female" if x==1 else "♂ Male")
            with d3:
                height = st.number_input("Height (cm)",100,250,165,1)
            with d4:
                weight = st.number_input("Weight (kg)",30.0,200.0,70.0,.5)

            bmi = weight/((height/100)**2)
            bmi_cat=("🟢 Normal" if 18.5<=bmi<25 else
                     "🟡 Overweight" if 25<=bmi<30 else
                     "🔴 Obese" if bmi>=30 else "🔵 Underweight")
            st.markdown(f"""
            <div class='ibox'>
              <b style='color:#7c3aed'>BMI:</b>
              <span style='color:#e2e8f0;font-size:1.1rem;font-weight:700'>
                {bmi:.1f} kg/m²</span>
              <span style='color:#94a3b8;font-size:.85rem;margin-left:10px'>
                {bmi_cat}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown("**💓 Blood Pressure**")
            bp1,bp2 = st.columns(2)
            with bp1:
                ap_hi = st.slider("Systolic BP (mmHg)",60,240,120,1)
            with bp2:
                ap_lo = st.slider("Diastolic BP (mmHg)",40,160,80,1)
            bp_s=("🟢 Normal" if ap_hi<120 and ap_lo<80 else
                  "🟡 Elevated" if 120<=ap_hi<130 else
                  "🟠 High Stage 1" if 130<=ap_hi<140 else "🔴 High Stage 2")
            st.info(f"BP Status: {bp_s} — {ap_hi}/{ap_lo} mmHg")

            st.markdown("**🧪 Blood Tests**")
            bl1,bl2 = st.columns(2)
            with bl1:
                cholesterol = st.selectbox("Cholesterol",[1,2,3],
                    format_func=lambda x:{1:"1 — Normal",
                        2:"2 — Above Normal",3:"3 — Well Above"}[x])
            with bl2:
                gluc = st.selectbox("Glucose",[1,2,3],
                    format_func=lambda x:{1:"1 — Normal",
                        2:"2 — Above Normal",3:"3 — Well Above"}[x])

            st.markdown("**🏃 Lifestyle**")
            ls1,ls2,ls3 = st.columns(3)
            with ls1:
                smoke=st.radio("🚬 Smoking",[0,1],
                    format_func=lambda x:"Non-Smoker" if x==0 else "Smoker",
                    horizontal=True)
            with ls2:
                alco=st.radio("🍺 Alcohol",[0,1],
                    format_func=lambda x:"No" if x==0 else "Yes",
                    horizontal=True)
            with ls3:
                active=st.radio("🏋️ Active",[0,1],
                    format_func=lambda x:"Inactive" if x==0 else "Active",
                    horizontal=True)

            submitted = st.form_submit_button(
                "🔮 PREDICT CVD RISK — ALL 3 MODELS",
                use_container_width=True)

        if submitted:
            features = np.array([[age_yr*365,gender,height,weight,
                                   ap_hi,ap_lo,cholesterol,gluc,
                                   smoke,alco,active]])
            sec("Prediction Results", "🎯")
            res = load_and_predict(features,"cardio",THRESH)
            show_risk_banner(res,THRESH)

            sec("Patient Summary", "📋")
            st.dataframe(pd.DataFrame({
                "Feature":["Age","Gender","Height","Weight","BMI",
                           "Systolic BP","Diastolic BP","BP Status",
                           "Cholesterol","Glucose",
                           "Smoking","Alcohol","Active"],
                "Value":[
                    f"{age_yr} years",
                    "Female" if gender==1 else "Male",
                    f"{height} cm",f"{weight} kg",
                    f"{bmi:.1f} kg/m² ({bmi_cat})",
                    f"{ap_hi} mmHg",f"{ap_lo} mmHg",bp_s,
                    ["Normal","Above Normal","Well Above"][cholesterol-1],
                    ["Normal","Above Normal","Well Above"][gluc-1],
                    "Smoker" if smoke else "Non-Smoker",
                    "Yes" if alco else "No",
                    "Active" if active else "Inactive",
                ]
            }),use_container_width=True,hide_index=True)

    else:
        sec("UCI Heart Disease — Patient Metrics","🔬")
        with st.form("uci_pred"):
            st.markdown("**👤 Demographics**")
            u1,u2,u3 = st.columns(3)
            with u1: u_age=st.number_input("Age",20,100,55,1)
            with u2:
                u_sex=st.selectbox("Sex",[0,1],
                    format_func=lambda x:"♀ Female" if x==0 else "♂ Male")
            with u3:
                u_cp=st.selectbox("Chest Pain Type",[0,1,2,3],
                    format_func=lambda x:{0:"Typical Angina",
                        1:"Atypical Angina",2:"Non-Anginal",
                        3:"Asymptomatic"}[x])

            st.markdown("**💓 Cardiovascular**")
            cv1,cv2,cv3=st.columns(3)
            with cv1: u_bp=st.slider("Resting BP (mmHg)",80,220,130)
            with cv2: u_ch=st.slider("Cholesterol (mg/dL)",100,600,240)
            with cv3: u_hr=st.slider("Max Heart Rate (bpm)",60,220,150)

            st.markdown("**🧪 Clinical Tests**")
            cl1,cl2,cl3=st.columns(3)
            with cl1:
                u_fbs=st.radio("Fasting BS>120",[0,1],
                    format_func=lambda x:"No" if x==0 else "Yes",
                    horizontal=True)
            with cl2:
                u_ecg=st.selectbox("Resting ECG",[0,1,2],
                    format_func=lambda x:{0:"Normal",
                        1:"ST-T Abnormal",2:"LV Hypertrophy"}[x])
            with cl3:
                u_ex=st.radio("Exercise Angina",[0,1],
                    format_func=lambda x:"No" if x==0 else "Yes",
                    horizontal=True)

            st.markdown("**📉 ST Segment & Vessels**")
            s1,s2,s3,s4=st.columns(4)
            with s1: u_op=st.number_input("ST Depression",0.0,10.0,1.0,.1)
            with s2:
                u_sl=st.selectbox("ST Slope",[0,1,2],
                    format_func=lambda x:{0:"Upsloping",
                        1:"Flat",2:"Downsloping"}[x])
            with s3: u_ca=st.selectbox("Major Vessels",[0,1,2,3])
            with s4:
                u_th=st.selectbox("Thalassemia",[1,2,3],
                    format_func=lambda x:{1:"Normal",
                        2:"Fixed Defect",3:"Reversible"}[x])

            sub_u=st.form_submit_button(
                "🔮 PREDICT CVD RISK — ALL 3 MODELS",
                use_container_width=True)

        if sub_u:
            fu=np.array([[u_age,u_sex,u_cp,u_bp,u_ch,
                          u_fbs,u_ecg,u_hr,u_ex,
                          u_op,u_sl,u_ca,u_th]])
            sec("Prediction Results","🎯")
            res_u=load_and_predict(fu,"uci",THRESH)
            show_risk_banner(res_u,THRESH)

            sec("Patient Summary","📋")
            st.dataframe(pd.DataFrame({
                "Feature":["Age","Sex","Chest Pain","Resting BP",
                           "Cholesterol","Fasting BS>120","ECG",
                           "Max HR","Exercise Angina",
                           "ST Depression","ST Slope",
                           "Major Vessels","Thalassemia"],
                "Value":[
                    f"{u_age} yrs",
                    "Female" if u_sex==0 else "Male",
                    ["Typical","Atypical","Non-Anginal","Asymptomatic"][u_cp],
                    f"{u_bp} mmHg",f"{u_ch} mg/dL",
                    "Yes" if u_fbs else "No",
                    ["Normal","ST-T Abnormal","LV Hypertrophy"][u_ecg],
                    f"{u_hr} bpm","Yes" if u_ex else "No",
                    str(u_op),
                    ["Upsloping","Flat","Downsloping"][u_sl],
                    str(u_ca),
                    ["Normal","Fixed Defect","Reversible"][u_th-1],
                ]
            }),use_container_width=True,hide_index=True)

    footer()

# ================================================================
# PAGE — DATASET EXPLORER
# ================================================================
elif PAGE == "📊 Dataset Explorer":
    sec(f"Dataset Explorer — {DATASET}","📊")

    t1,t2,t3 = st.tabs(["📈 EDA Charts","🌡️ Correlation","📋 Info"])
    with t1:
        safe_img(f"outputs/eda_{TAG}.png","EDA Analysis")
    with t2:
        p=f"outputs/correlation_heatmap_{TAG}.png"
        if os.path.exists(p):
            safe_img(p,"Correlation Heatmap")
        else:
            st.info("Heatmap available for CARDIO dataset only.")
    with t3:
        info=({
            "Source":"Ulianova (2019) — Kaggle",
            "Total Instances":"70,000",
            "After Cleaning":"68,723",
            "Features":"11 input + 1 target",
            "Balance":"50.5% No-CVD / 49.5% CVD",
            "Missing Values":"0",
            "Train Split":"80% (stratified)",
            "Test Split":"20% (stratified)",
            "Seed":"42 (replicable)",
        } if TAG=="cardio" else {
            "Source":"Detrano et al. (1989) — UCI ML Repo",
            "Total Instances":"297",
            "Features":"13 input + 1 target",
            "Balance":"53.5% No-CVD / 46.5% CVD",
            "Missing Values":"0",
            "Train Split":"80% (stratified)",
            "Test Split":"20% (stratified)",
            "Seed":"42 (replicable)",
        })
        st.markdown("<div class='ibox'>",unsafe_allow_html=True)
        for k,v in info.items(): kv(k,v,"#a78bfa")
        st.markdown("</div>",unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — MODEL PERFORMANCE
# ================================================================
elif PAGE == "🤖 Model Performance":
    sec(f"Model Performance — {DATASET} — {MODEL_SEL}","🤖")

    t1,t2,t3,t4,t5 = st.tabs([
        "📊 Heatmap","📉 ROC Curves",
        "🔥 Confusion Matrices",
        "📋 Metrics Table","📈 Repeat 10×"])

    with t1:
        safe_img(f"outputs/eval_heatmap_{TAG}.png","Evaluation Heatmap")

    with t2:
        safe_img(f"outputs/roc_{TAG}.png","ROC Curves — All Models")
        st.markdown("""
        <div class='ibox'>
          <b style='color:#7c3aed'>AUC Interpretation Guide</b><br>
          <span class='pill' style='background:rgba(34,197,94,.2);
            border-color:rgba(34,197,94,.4);color:#4ade80'>
            0.9–1.0 Excellent</span>
          <span class='pill' style='background:rgba(59,130,246,.2);
            border-color:rgba(59,130,246,.4);color:#60a5fa'>
            0.8–0.9 Good</span>
          <span class='pill' style='background:rgba(245,158,11,.2);
            border-color:rgba(245,158,11,.4);color:#fbbf24'>
            0.7–0.8 Fair</span>
          <span class='pill' style='background:rgba(239,68,68,.2);
            border-color:rgba(239,68,68,.4);color:#f87171'>
            0.6–0.7 Poor</span>
          <span class='pill'>0.5 Random</span>
        </div>""",unsafe_allow_html=True)

    with t3:
        cm1,cm2,cm3=st.columns(3)
        with cm1:
            st.markdown("**Logistic Regression**")
            safe_img(f"outputs/cm_logistic_regression_{TAG}.png","LR")
        with cm2:
            st.markdown("**Random Forest**")
            safe_img(f"outputs/cm_random_forest_{TAG}.png","RF")
        with cm3:
            st.markdown("**XGBoost**")
            safe_img(f"outputs/cm_xgboost_{TAG}.png","XGB")

    with t4:
        df_met=load_csv(f"outputs/metrics_{TAG}.csv")
        if df_met is not None:
            st.dataframe(df_met.style.highlight_max(
                subset=[c for c in df_met.columns
                        if c not in ["Model","LogLoss"]],
                color="#14532d").highlight_min(
                subset=["LogLoss"] if "LogLoss" in df_met.columns
                else [],color="#14532d"),
                use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download Metrics CSV",
                df_met.to_csv(index=False),
                f"metrics_{TAG}.csv","text/csv")

        if TAG=="cardio":
            rows={"Model":["Logistic Regression","Random Forest","XGBoost"],
                  "AUC":[0.7489,0.7776,0.7754],
                  "CA":[0.7025,0.7000,0.7000],
                  "F1":[0.7002,0.6985,0.6996],
                  "Prec":[0.7052,0.7012,0.6998],
                  "Recall":[0.7008,0.6987,0.6996],
                  "MCC":[0.4060,0.3999,0.3994],
                  "Spec":[0.7707,0.7512,0.7171],
                  "LogLoss":[0.6061,0.5700,0.5695]}
        else:
            rows={"Model":["Logistic Regression","Random Forest","XGBoost"],
                  "AUC":[0.9989,0.9408,0.9609],
                  "CA":[0.9667,0.8500,0.8833],
                  "F1":[0.9665,0.8490,0.8825],
                  "Prec":[0.9665,0.8502,0.8838],
                  "Recall":[0.9665,0.8482,0.8817],
                  "MCC":[0.9330,0.6984,0.7655],
                  "Spec":[0.9688,0.8750,0.9062],
                  "LogLoss":[0.0943,0.3525,0.2586]}

        st.markdown("**Hardcoded Results from main.py**")
        st.dataframe(pd.DataFrame(rows).style.highlight_max(
            subset=["AUC","CA","F1","Prec","Recall","MCC","Spec"],
            color="#14532d").highlight_min(
            subset=["LogLoss"],color="#14532d"),
            use_container_width=True,hide_index=True)

    with t5:
        r1c,r2c,r3c=st.columns(3)
        with r1c:
            safe_img(f"outputs/repeat_logistic_regression_{TAG}.png","LR")
        with r2c:
            safe_img(f"outputs/repeat_random_forest_{TAG}.png","RF")
        with r3c:
            safe_img(f"outputs/repeat_xgboost_{TAG}.png","XGB")

    footer()

# ================================================================
# PAGE — SAMPLING METHODS
# ================================================================
elif PAGE == "📈 Sampling Methods":
    sec(f"Sampling Methods — {DATASET}","📈")

    t1,t2,t3 = st.tabs([
        "📊 Comparison & Proportions",
        "ℹ️ Method Descriptions",
        "📈 Sampling Settings"])

    with t1:
        sa,sb=st.columns(2)
        with sa:
            safe_img(f"outputs/sampling_compare_{TAG}.png","Model Compare")
        with sb:
            safe_img(f"outputs/proportion_{TAG}.png","Proportions")

    with t2:
        methods=[
            ("①","Simple Holdout",
             "80/20 · Fixed proportion · 1 sample\nStratified · Replicable seed=42",
             "Fast reproducible split. No data leakage.",
             "Pedregosa et al. (2011)","#3b82f6"),
            ("②","Cross-Validation (10-fold)",
             "10 subsets · 1 unused per round · Stratified\nReplicable seed=42",
             "10-fold CV reduces variance. Each fold acts as test once.",
             "Pedregosa et al. (2011)","#7c3aed"),
            ("③","Bootstrap",
             "1 sample · With replacement\nOOB ≈ 36.8% (1-1/e)",
             "Stratified per-class bootstrap. OOB used as validation.",
             "Efron & Tibshirani (1993)","#06b6d4"),
            ("④","Random Repeat",
             "10 rounds · 66% train / 34% test · Stratified\nseed=42+rep",
             "10 independent random splits. Average gives stable estimate.",
             "Pedregosa et al. (2011)","#10b981"),
            ("⑤","Leave-One-Out",
             "N folds (max 150) · train=N-1 · test=1\nSubsample for speed",
             "Maximum data usage. Best for very small datasets.",
             "Pedregosa et al. (2011)","#f59e0b"),
            ("⑥","Test on Test Data",
             "Held-out test set · Final evaluation\nSampling ① split",
             "True generalisation. Never seen during training.",
             "Hosmer et al. (2013)","#ef4444"),
            ("⑦","Test on Train Data",
             "Training set evaluation\nGap analysis",
             "Gap > 5% = potential overfitting detected.",
             "Breiman (2001)","#ec4899"),
        ]
        for n,m,d,desc,ref,c in methods:
            st.markdown(f"""
            <div class='glass' style='border-left:4px solid {c};
                 border-radius:0 12px 12px 0;margin-bottom:8px'>
              <div style='display:flex;align-items:center;gap:12px'>
                <div style='color:{c};font-size:1.5rem;font-weight:900;
                     min-width:36px'>{n}</div>
                <div>
                  <div style='color:#e2e8f0;font-weight:700;font-size:.9rem'>
                    {m}</div>
                  <div style='color:#64748b;font-size:.77rem;
                       white-space:pre-line;margin:3px 0'>{d}</div>
                  <div style='color:#475569;font-size:.73rem'>{desc}</div>
                  <div style='color:rgba(124,58,237,0.7);font-size:.7rem;
                       margin-top:3px'>📚 {ref}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t3:
        sec("Evaluation Metrics","📏")
        metrics_info=[
            ("AUC","Area Under ROC Curve",
             "Discriminative ability. 1.0 = perfect, 0.5 = random","#3b82f6"),
            ("CA","Classification Accuracy",
             "Overall correct predictions / total","#7c3aed"),
            ("F1","F1-Score (Macro)",
             "Harmonic mean of Precision and Recall","#06b6d4"),
            ("Precision","Positive Predictive Value",
             "TP / (TP + FP)","#10b981"),
            ("Recall","Sensitivity / True Positive Rate",
             "TP / (TP + FN)","#f59e0b"),
            ("MCC","Matthews Correlation Coefficient",
             "Balanced metric for imbalanced classes. -1 to +1","#ef4444"),
            ("Spec","Specificity / True Negative Rate",
             "TN / (TN + FP)","#ec4899"),
            ("LogLoss","Logarithmic Loss",
             "Penalty for confident wrong predictions. Lower = better","#8b5cf6"),
        ]
        mc1,mc2=st.columns(2)
        for i,(abbr,full,desc,c) in enumerate(metrics_info):
            col=mc1 if i%2==0 else mc2
            col.markdown(f"""
            <div class='glass' style='border-left:3px solid {c};
                 margin-bottom:6px'>
              <span style='color:{c};font-weight:800;font-size:.95rem'>
                {abbr}</span>
              <span style='color:#94a3b8;font-size:.8rem'> — {full}</span><br>
              <span style='color:#475569;font-size:.73rem'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    footer()

# ================================================================
# PAGE — SHAP
# ================================================================
elif PAGE == "🔍 SHAP Explainability":
    sec(f"SHAP Explainability — {DATASET}","🔍")

    st.markdown("""
    <div class='glass'>
      <b style='color:#7c3aed;font-size:1rem'>
        SHAP — SHapley Additive exPlanations</b>
      <span style='color:#64748b'> (Lundberg & Lee, 2017)</span><br>
      <p style='color:#94a3b8;margin:8px 0 4px;font-size:.85rem'>
        SHAP values explain individual predictions by quantifying
        each feature's contribution. Based on cooperative game theory
        (Shapley values from Lloyd Shapley, 1953).
      </p>
      <div style='margin-top:8px'>
        <span class='pill' style='background:rgba(239,68,68,.2);
          border-color:rgba(239,68,68,.4);color:#f87171'>
          🔴 Positive SHAP → pushes toward CVD</span>
        <span class='pill' style='background:rgba(59,130,246,.2);
          border-color:rgba(59,130,246,.4);color:#60a5fa'>
          🔵 Negative SHAP → pushes toward No CVD</span>
      </div>
    </div>""", unsafe_allow_html=True)

    safe_img(f"outputs/shap_{TAG}.png","SHAP Summary Plot")

    sec("Feature Importance Interpretation","📊")
    if TAG=="cardio":
        feats=[
            ("ap_hi","Systolic BP","#ef4444",
             "Strongest predictor. High systolic pressure raises CVD risk significantly."),
            ("age","Age (days)","#f59e0b",
             "Older patients consistently show higher CVD risk."),
            ("ap_lo","Diastolic BP","#f97316",
             "High diastolic pressure compounds systolic effect."),
            ("cholesterol","Cholesterol Level","#8b5cf6",
             "Above-normal cholesterol (level 2,3) linked to CVD."),
            ("weight","Weight (kg)","#06b6d4",
             "Higher weight increases cardiovascular strain."),
            ("gluc","Glucose Level","#3b82f6",
             "Elevated glucose indicates metabolic syndrome risk."),
            ("active","Physical Activity","#22c55e",
             "Activity is PROTECTIVE — negative SHAP reduces CVD risk."),
            ("smoke","Smoking Status","#ec4899",
             "Smoking significantly increases CVD probability."),
        ]
    else:
        feats=[
            ("ca","Major Vessels (0-3)","#ef4444",
             "Strongest UCI predictor. More blocked vessels = higher risk."),
            ("thal","Thalassemia Type","#f59e0b",
             "Reversible defect (type 3) strongly indicates CVD."),
            ("cp","Chest Pain Type","#f97316",
             "Asymptomatic type (3) paradoxically indicates highest risk."),
            ("oldpeak","ST Depression","#8b5cf6",
             "ST depression induced by exercise indicates ischaemia."),
            ("thalach","Max Heart Rate","#06b6d4",
             "Lower max HR achieved = higher CVD risk."),
            ("exang","Exercise Angina","#3b82f6",
             "Chest pain on exercise strongly indicates CVD."),
            ("age","Age (years)","#10b981",
             "CVD risk increases with age."),
            ("sex","Sex (Male=1)","#22c55e",
             "Males show higher CVD risk in the UCI Cleveland dataset."),
        ]

    fc1,fc2=st.columns(2)
    for i,(feat,name,c,desc) in enumerate(feats):
        col=fc1 if i%2==0 else fc2
        col.markdown(f"""
        <div class='glass' style='border-left:3px solid {c};
             margin-bottom:6px;display:flex;align-items:flex-start;gap:10px'>
          <div style='background:{c};border-radius:4px;
               width:10px;height:10px;margin-top:4px;flex-shrink:0'></div>
          <div>
            <span style='color:{c};font-weight:700;font-size:.88rem'>
              {feat}</span>
            <span style='color:#94a3b8;font-size:.82rem'> — {name}</span><br>
            <span style='color:#475569;font-size:.75rem'>{desc}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    footer()

# ================================================================
# PAGE — PREDICTIONS VIEWER
# ================================================================
elif PAGE == "📋 Predictions Viewer":
    sec(f"Predictions Viewer — {MODEL_SEL} | {DATASET}","📋")

    st.markdown(f"""
    <div class='ibox'>
      <b style='color:#7c3aed'>Viewing predictions for:</b>
      <span style='color:#e2e8f0'> {MODEL_SEL}</span>
      <span style='color:#475569'> on </span>
      <span style='color:#e2e8f0'>{DATASET}</span><br>
      <span style='color:#64748b;font-size:.8rem'>
        Change model/dataset in sidebar · Threshold: {THRESH:.0%}
      </span>
    </div>""", unsafe_allow_html=True)

    df_p=load_csv(f"outputs/predictions_{MODEL_SLUG}_{TAG}.csv")

    if df_p is not None:
        n=len(df_p)
        result_col=next((c for c in df_p.columns
                         if "result" in c.lower()),None)

        # Summary metrics
        sc1,sc2,sc3,sc4=st.columns(4)
        sc1.metric("Total Instances",f"{n:,}")
        if result_col:
            correct=int(df_p[result_col].str.contains("CORRECT",na=False).sum())
            wrong=n-correct
            sc2.metric("✅ Correct",f"{correct:,}")
            sc3.metric("❌ Misclassified",f"{wrong:,}")
            sc4.metric("🎯 Accuracy",f"{correct/n*100:.2f}%")

        st.markdown("---")
        sec("Filter & View Predictions","🔍")

        f1,f2,f3,f4=st.columns(4)
        with f1:
            filt_r=st.selectbox("Result Filter",
                ["All","CORRECT","MISCLASSIFIED"])
        with f2:
            if "Actual" in df_p.columns:
                filt_a=st.selectbox("Actual Class",
                    ["All"]+sorted(df_p["Actual"].unique().tolist()))
            else:
                filt_a="All"
        with f3:
            if "Predicted" in df_p.columns:
                filt_p=st.selectbox("Predicted Class",
                    ["All"]+sorted(df_p["Predicted"].unique().tolist()))
            else:
                filt_p="All"
        with f4:
            n_rows=st.slider("Max Rows",
                10,min(2000,n),min(200,n))

        df_d=df_p.copy()
        if filt_r!="All" and result_col:
            df_d=df_d[df_d[result_col].str.contains(filt_r,na=False)]
        if filt_a!="All" and "Actual" in df_d.columns:
            df_d=df_d[df_d["Actual"]==filt_a]
        if filt_p!="All" and "Predicted" in df_d.columns:
            df_d=df_d[df_d["Predicted"]==filt_p]

        st.markdown(f"""
        <div class='ibox'>
          Showing <b style='color:#7c3aed'>{len(df_d):,}</b>
          of <b>{n:,}</b> instances
        </div>""", unsafe_allow_html=True)

        # Color result column
        def color_result(val):
            if "CORRECT" in str(val):
                return "background-color:#14532d;color:#86efac"
            elif "MISCLASSIFIED" in str(val):
                return "background-color:#7f1d1d;color:#fca5a5"
            return ""

        if result_col and result_col in df_d.columns:
            styled=df_d.head(n_rows).style.applymap(
                color_result,subset=[result_col])
            if "P_CVD" in df_d.columns:
                styled=styled.background_gradient(
                    subset=["P_CVD"],cmap="RdYlGn_r")
            st.dataframe(styled,use_container_width=True,height=400)
        else:
            st.dataframe(df_d.head(n_rows),
                         use_container_width=True,height=400)

        c1,c2=st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download Filtered CSV",
                df_d.to_csv(index=False).encode("utf-8"),
                f"predictions_{MODEL_SLUG}_{TAG}_filtered.csv",
                "text/csv")
        with c2:
            st.download_button(
                "⬇️ Download All Predictions",
                df_p.to_csv(index=False).encode("utf-8"),
                f"predictions_{MODEL_SLUG}_{TAG}_all.csv",
                "text/csv")

        if result_col:
            st.markdown("---")
            sec("Probability Distribution Analysis","📊")
            if "P_CVD" in df_p.columns:
                fig,axes=plt.subplots(1,2,figsize=(12,4))
                fig.patch.set_facecolor("#050a1a")
                for ax in axes:
                    ax.set_facecolor("#0a1628")
                    ax.tick_params(colors="#94a3b8")
                    ax.spines[:].set_color("rgba(255,255,255,0.1)")

                corr_df=df_p[df_p[result_col].str.contains("CORRECT",na=False)]
                miss_df=df_p[df_p[result_col].str.contains("MISCLASSIFIED",na=False)]

                axes[0].hist(corr_df["P_CVD"],bins=20,
                             color="#22c55e",alpha=0.7,
                             label=f"Correct ({len(corr_df)})",
                             edgecolor="none")
                axes[0].hist(miss_df["P_CVD"],bins=20,
                             color="#ef4444",alpha=0.7,
                             label=f"Misclassified ({len(miss_df)})",
                             edgecolor="none")
                axes[0].axvline(THRESH,color="#f59e0b",lw=2,ls="--",
                                label=f"Threshold={THRESH:.0%}")
                axes[0].set_xlabel("P(CVD)",color="#94a3b8")
                axes[0].set_ylabel("Count",color="#94a3b8")
                axes[0].set_title("P(CVD) Distribution",
                                  color="#e2e8f0",fontweight="bold")
                axes[0].legend(facecolor="#0a1628",
                               labelcolor="#e2e8f0",fontsize=8)

                if "Actual" in df_p.columns:
                    cvd_p=df_p[df_p["Actual"]=="CVD (1)"]["P_CVD"]
                    no_p=df_p[df_p["Actual"]=="No CVD (0)"]["P_CVD"]
                    axes[1].hist(no_p,bins=20,color="#22c55e",
                                 alpha=0.7,label="Actual No CVD",
                                 edgecolor="none")
                    axes[1].hist(cvd_p,bins=20,color="#ef4444",
                                 alpha=0.7,label="Actual CVD",
                                 edgecolor="none")
                    axes[1].axvline(THRESH,color="#f59e0b",lw=2,ls="--")
                    axes[1].set_xlabel("P(CVD)",color="#94a3b8")
                    axes[1].set_ylabel("Count",color="#94a3b8")
                    axes[1].set_title("P(CVD) by True Class",
                                      color="#e2e8f0",fontweight="bold")
                    axes[1].legend(facecolor="#0a1628",
                                   labelcolor="#e2e8f0",fontsize=8)

                plt.tight_layout()
                st.pyplot(fig); plt.close()
    else:
        st.warning("Prediction file not found. Run python main.py first.")

    footer()

# ================================================================
# PAGE — MODEL COMPARISON
# ================================================================
elif PAGE == "⚖️ Model Comparison":
    sec("Model Comparison — All Models | Both Datasets","⚖️")

    t1,t2,t3,t4=st.tabs([
        "📊 Side-by-Side","📈 Bar Charts",
        "🕸️ Radar View","🔁 Overfitting"])

    with t1:
        a,b=st.columns(2)
        with a:
            st.markdown("""
            <div class='glass'>
              <div style='color:#3b82f6;font-weight:700;margin-bottom:8px'>
                🫀 CARDIO Dataset</div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Model":["Logistic Regression","Random Forest","XGBoost"],
                "AUC":[0.7489,0.7776,0.7754],
                "CA":[0.7025,0.7000,0.7000],
                "F1":[0.7002,0.6985,0.6996],
                "MCC":[0.4060,0.3999,0.3994],
                "LogLoss":[0.6061,0.5700,0.5695],
                "Champion":["★ F1","",""],
            }).style.highlight_max(
                subset=["AUC","CA","F1","MCC"],color="#14532d").highlight_min(
                subset=["LogLoss"],color="#14532d"),
                use_container_width=True,hide_index=True)
            st.markdown("</div>",unsafe_allow_html=True)
        with b:
            st.markdown("""
            <div class='glass'>
              <div style='color:#7c3aed;font-weight:700;margin-bottom:8px'>
                🔬 UCI Dataset</div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Model":["Logistic Regression","Random Forest","XGBoost"],
                "AUC":[0.9989,0.9408,0.9609],
                "CA":[0.9667,0.8500,0.8833],
                "F1":[0.9665,0.8490,0.8825],
                "MCC":[0.9330,0.6984,0.7655],
                "LogLoss":[0.0943,0.3525,0.2586],
                "Champion":["★ F1","",""],
            }).style.highlight_max(
                subset=["AUC","CA","F1","MCC"],color="#14532d").highlight_min(
                subset=["LogLoss"],color="#14532d"),
                use_container_width=True,hide_index=True)
            st.markdown("</div>",unsafe_allow_html=True)

    with t2:
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        fig.patch.set_facecolor("#050a1a")
        datasets=[
            ({"Metric":["AUC","CA","F1","MCC"],
              "LR":[0.7489,0.7025,0.7002,0.4060],
              "RF":[0.7776,0.7000,0.6985,0.3999],
              "XGB":[0.7754,0.7000,0.6996,0.3994]},"CARDIO"),
            ({"Metric":["AUC","CA","F1","MCC"],
              "LR":[0.9989,0.9667,0.9665,0.9330],
              "RF":[0.9408,0.8500,0.8490,0.6984],
              "XGB":[0.9609,0.8833,0.8825,0.7655]},"UCI"),
        ]
        for ax,(data,title) in zip(axes,datasets):
            ax.set_facecolor("#0a1628")
            x=np.arange(len(data["Metric"])); w=0.25
            for i,(col,c) in enumerate(zip(
                ["LR","RF","XGB"],
                ["#3b82f6","#10b981","#7c3aed"]
            )):
                bars=ax.bar(x+i*w,data[col],w,label=col,
                            color=c,alpha=0.85)
                for bar,val in zip(bars,data[col]):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            val+.005,f"{val:.3f}",ha="center",
                            va="bottom",color="#e2e8f0",fontsize=7)
            ax.set_xticks(x+w)
            ax.set_xticklabels(data["Metric"],color="#94a3b8",fontsize=9)
            ax.set_title(title,color="#e2e8f0",fontweight="bold",fontsize=12)
            ax.tick_params(colors="#94a3b8")
            ax.spines[:].set_color("rgba(255,255,255,0.1)")
            ax.set_ylim(0,1.15); ax.set_facecolor("#0a1628")
            ax.legend(facecolor="#0a1628",labelcolor="#e2e8f0",fontsize=8)
        plt.suptitle("Model Performance Comparison — CARDIO vs UCI",
                     color="#e2e8f0",fontsize=13,fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with t3:
        cats=["AUC","CA","F1","MCC","Spec"]
        N=len(cats)
        angles=[n/float(N)*2*np.pi for n in range(N)]+[0]
        fig2,axes2=plt.subplots(1,2,figsize=(12,5),
                                subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("#050a1a")
        ds_r={
            "CARDIO":{
                "LR":[0.7489,0.7025,0.7002,0.4060,0.7707],
                "RF":[0.7776,0.7000,0.6985,0.3999,0.7512],
                "XGB":[0.7754,0.7000,0.6996,0.3994,0.7171],
            },
            "UCI":{
                "LR":[0.9989,0.9667,0.9665,0.9330,0.9688],
                "RF":[0.9408,0.8500,0.8490,0.6984,0.8750],
                "XGB":[0.9609,0.8833,0.8825,0.7655,0.9062],
            }
        }
        clrs_r={"LR":"#3b82f6","RF":"#10b981","XGB":"#7c3aed"}
        for ax2,(dn,dv) in zip(axes2,ds_r.items()):
            ax2.set_facecolor("#0a1628")
            ax2.set_title(dn,color="#e2e8f0",fontweight="bold",pad=15)
            ax2.set_thetagrids(np.degrees(angles[:-1]),cats,
                               color="#94a3b8",fontsize=8)
            ax2.set_ylim(0,1)
            ax2.spines["polar"].set_color("rgba(255,255,255,0.1)")
            ax2.tick_params(colors="#64748b")
            for mn,vals in dv.items():
                v=vals+vals[:1]
                ax2.plot(angles,v,"o-",lw=2,label=mn,
                         color=clrs_r[mn])
                ax2.fill(angles,v,alpha=.15,color=clrs_r[mn])
            ax2.legend(loc="upper right",
                       bbox_to_anchor=(1.35,1.15),
                       facecolor="#0a1628",
                       labelcolor="#e2e8f0",fontsize=8)
        plt.suptitle("Radar Chart — Model Performance Profile",
                     color="#e2e8f0",fontsize=12,fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    with t4:
        sec("Overfitting Analysis — Train vs Test Gap","🔁")
        ov=pd.DataFrame({
            "Model + Dataset":[
                "LR — CARDIO","RF — CARDIO","XGB — CARDIO",
                "LR — UCI","RF — UCI","XGB — UCI"],
            "Train Acc":[0.6981,0.7669,0.7731,
                         0.9662,0.9958,1.0000],
            "Test Acc": [0.7025,0.7000,0.7000,
                         0.9667,0.8500,0.8833],
            "Train F1": [0.6970,0.7644,0.7727,
                         0.9661,0.9958,1.0000],
            "Test F1":  [0.7002,0.6985,0.6996,
                         0.9665,0.8490,0.8825],
            "Acc Gap":  [-0.0044,+0.0669,+0.0731,
                         -0.0004,+0.1458,+0.1167],
            "Status":   ["✓ Good","✗ Overfit","✗ Overfit",
                         "✓ Good","✗ Overfit","✗ Overfit"],
        })
        def col_st(v):
            if "✓" in str(v):
                return "background:#14532d;color:#86efac"
            return "background:#7f1d1d;color:#fca5a5"
        st.dataframe(ov.style.applymap(col_st,subset=["Status"]),
                     use_container_width=True,hide_index=True)
        st.markdown("""
        <div class='ibox'>
          <b style='color:#f59e0b'>Overfitting Note:</b>
          <span style='color:#94a3b8;font-size:.85rem'>
           Random Forest shows the highest train-test gap on both datasets
           (+0.23 on CARDIO, +0.15 on UCI) due to deep tree memorisation.
           XGBoost also overfits on UCI (1.00 train vs 0.88 test).
           Logistic Regression generalises best with near-zero gap.
          </span>
        </div>""", unsafe_allow_html=True)

    footer()

# ================================================================
# PAGE — ABOUT
# ================================================================
elif PAGE == "ℹ️ About":
    sec("About This Project","ℹ️")

    t1,t2,t3=st.tabs(["👨‍🎓 Project Info","📚 References","⚙️ Configuration"])

    with t1:
        st.markdown("""
        <div class='hero' style='padding:30px'>
          <div style='font-size:3rem'>🎓</div>
          <h2 style='color:#e2e8f0;font-weight:800;margin:8px 0'>
            PSB605IT — Individual Computing Science Project</h2>
          <p style='color:#7c3aed;font-size:1rem'>
            CVD Risk Prediction Using Machine Learning</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='ibox'>",unsafe_allow_html=True)
        for k,v in [
            ("Project Title","CVD Risk Prediction Using Machine Learning"),
            ("Module","PSB605IT — Individual Computing Science Project"),
            ("Student","Sudhan Nagarajan"),
            ("Student ID","050J0DAD"),
            ("Supervisor","Mr Raymond Ching Chi Man"),
            ("Institution","PSB Academy"),
            ("Academic Year","2025 / 2026"),
        ]: kv(k,v,"#a78bfa")
        st.markdown("</div>",unsafe_allow_html=True)

        sec("Technical Stack","🛠️")
        tech=[
            ("Python 3.x","Core programming language",
             "#3b82f6"),
            ("Streamlit 1.57","Interactive dashboard framework",
             "#7c3aed"),
            ("scikit-learn 1.3","LR · RF · CV · Metrics · Preprocessing",
             "#06b6d4"),
            ("XGBoost 1.7","Gradient Boosting classifier",
             "#10b981"),
            ("SHAP 0.42","Model explainability (Shapley values)",
             "#f59e0b"),
            ("Pandas 2.x","Data manipulation and analysis",
             "#ef4444"),
            ("NumPy 1.x","Numerical computing",
             "#ec4899"),
            ("Matplotlib 3.7 / Seaborn 0.12","Visualisation",
             "#8b5cf6"),
            ("Joblib 1.3","Model serialisation",
             "#06b6d4"),
        ]
        tc1,tc2=st.columns(2)
        for i,(lib,use,c) in enumerate(tech):
            col=tc1 if i%2==0 else tc2
            col.markdown(f"""
            <div class='glass' style='border-left:3px solid {c};
                 margin-bottom:6px'>
              <span style='color:{c};font-weight:700'>{lib}</span><br>
              <span style='color:#475569;font-size:.78rem'>{use}</span>
            </div>""", unsafe_allow_html=True)

    with t2:
        refs=[
            ("Breiman (2001)","Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.","Random Forest"),
            ("Chen & Guestrin (2016)","Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.","XGBoost"),
            ("Chawla et al. (2002)","Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Oversampling. JAIR, 16, 321–357.","Class Imbalance"),
            ("Detrano et al. (1989)","Detrano, R. et al. (1989). International application of a new probability algorithm. AJC, 64(5), 304–310.","UCI Dataset"),
            ("Efron & Tibshirani (1993)","Efron, B., & Tibshirani, R.J. (1993). An Introduction to the Bootstrap. Chapman & Hall.","Bootstrap"),
            ("Hosmer et al. (2013)","Hosmer, D.W. et al. (2013). Applied Logistic Regression, 3rd ed. Wiley.","Logistic Regression"),
            ("Lundberg & Lee (2017)","Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.","SHAP"),
            ("Pedregosa et al. (2011)","Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.","scikit-learn"),
            ("Ulianova (2019)","Ulianova, S. (2019). Cardiovascular Disease Dataset. Kaggle.","CARDIO Dataset"),
            ("WHO (2024)","World Health Organization (2024). Cardiovascular Diseases Fact Sheet.","Context"),
            ("Benjamin et al. (2019)","Benjamin, E.J. et al. (2019). Heart Disease and Stroke Statistics. Circulation, 139(10).","Epidemiology"),
        ]
        for auth,full,use in refs:
            st.markdown(f"""
            <div class='glass' style='border-left:3px solid #7c3aed;
                 border-radius:0 12px 12px 0;margin-bottom:6px'>
              <b style='color:#a78bfa;font-size:.88rem'>{auth}</b>
              <span class='pill'>{use}</span><br>
              <span style='color:#64748b;font-size:.78rem'>{full}</span>
            </div>""", unsafe_allow_html=True)

    with t3:
        st.markdown("<div class='ibox'>",unsafe_allow_html=True)
        for k,v in [
            ("Seed","42 — replicable and deterministic"),
            ("Holdout","80% train / 20% test · Stratified"),
            ("CV Folds","10 · Stratified · seed=42"),
            ("Bootstrap","1 sample · With replacement · OOB≈36.8%"),
            ("Repeat Rounds","10 × 66% train / 34% test"),
            ("LOO Folds","150 subsample (stratified)"),
            ("RF Trees","100 · min_samples_split=5"),
            ("LR","Ridge L2 · C=1.0 · lbfgs · max_iter=1000"),
            ("XGBoost","scale_pos_weight=auto · verbosity=0"),
            ("Class Weight","balanced — handles class imbalance"),
            ("CARDIO Champion","Logistic Regression (highest F1=0.7002)"),
            ("UCI Champion","Logistic Regression (highest F1=0.9665)"),
            ("Dashboard","Streamlit 1.57 · Python 3.x"),
        ]: kv(k,v,"#a78bfa")
        st.markdown("</div>",unsafe_allow_html=True)

    footer()