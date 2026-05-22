# ================================================================
# dashboard.py — CVD Risk Prediction | PSB605IT
# Sudhan Nagarajan (050J0DAD) | PSB Academy 2025/2026
# Supervisor: Mr Raymond Ching Chi Man
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CVD Risk AI | PSB605IT",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box;}
.stApp{
  background:
    radial-gradient(ellipse at 20% 50%,rgba(30,64,175,.08) 0%,transparent 50%),
    radial-gradient(ellipse at 80% 20%,rgba(124,58,237,.08) 0%,transparent 50%),
    linear-gradient(135deg,#020617 0%,#0a0f1e 50%,#020617 100%);
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(2,6,23,.98) 0%,
    rgba(10,15,40,.98) 50%,rgba(2,6,23,.98) 100%) !important;
  border-right:1px solid rgba(30,64,175,.3) !important;
}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
.glass{
  background:rgba(255,255,255,.03);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,.08);border-radius:16px;
  padding:20px;margin-bottom:12px;
  box-shadow:0 8px 32px rgba(0,0,0,.4);
  transition:all .3s ease;
}
.glass:hover{
  border-color:rgba(124,58,237,.3);
  box-shadow:0 12px 40px rgba(0,0,0,.5),0 0 20px rgba(124,58,237,.1);
  transform:translateY(-2px);
}
.hero{
  background:linear-gradient(135deg,rgba(10,15,40,.95) 0%,
    rgba(20,10,50,.95) 50%,rgba(10,15,40,.95) 100%);
  border:1px solid rgba(124,58,237,.4);border-radius:20px;
  padding:44px 32px;text-align:center;margin-bottom:24px;
  position:relative;overflow:hidden;
  box-shadow:0 0 60px rgba(124,58,237,.15);
}
.stat-card{
  background:linear-gradient(135deg,rgba(30,41,59,.6),rgba(15,23,42,.8));
  border:1px solid rgba(255,255,255,.06);border-radius:16px;
  padding:22px 14px;text-align:center;transition:all .3s ease;
  position:relative;overflow:hidden;
}
.stat-card::after{
  content:'';position:absolute;top:0;left:0;width:100%;height:2px;
  background:var(--accent,linear-gradient(90deg,#1e40af,#7c3aed));
}
.stat-card:hover{transform:translateY(-5px);}
.stat-num{font-size:2.2rem;font-weight:900;line-height:1;}
.stat-lbl{font-size:.72rem;color:#475569;margin-top:8px;
  text-transform:uppercase;letter-spacing:1.5px;}
.sec-hdr{
  background:linear-gradient(90deg,#1e40af,#7c3aed,#db2777);
  border-radius:10px;padding:11px 18px;color:#fff !important;
  font-weight:700;font-size:.9rem;margin:18px 0 12px;
  box-shadow:0 4px 20px rgba(124,58,237,.3);
}
.risk-high{
  background:linear-gradient(135deg,rgba(127,29,29,.9),rgba(153,27,27,.95));
  border:2px solid #ef4444;border-radius:20px;padding:32px;
  text-align:center;color:#fff;
  box-shadow:0 0 60px rgba(239,68,68,.3);
  animation:pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red{
  0%,100%{box-shadow:0 0 60px rgba(239,68,68,.3);}
  50%{box-shadow:0 0 80px rgba(239,68,68,.5);}
}
.risk-low{
  background:linear-gradient(135deg,rgba(20,83,45,.9),rgba(22,101,52,.95));
  border:2px solid #22c55e;border-radius:20px;padding:32px;
  text-align:center;color:#fff;box-shadow:0 0 60px rgba(34,197,94,.3);
}
.risk-med{
  background:linear-gradient(135deg,rgba(120,53,15,.9),rgba(146,64,14,.95));
  border:2px solid #f59e0b;border-radius:20px;padding:32px;
  text-align:center;color:#fff;box-shadow:0 0 60px rgba(245,158,11,.3);
}
.model-card{
  background:rgba(15,23,42,.6);border:2px solid transparent;
  border-radius:16px;padding:20px;text-align:center;transition:all .3s;
}
.model-card:hover{transform:translateY(-4px);}
.pill{
  display:inline-block;background:rgba(124,58,237,.15);
  border:1px solid rgba(124,58,237,.3);border-radius:20px;
  padding:4px 12px;font-size:.73rem;color:#a78bfa;margin:3px;
}
.infograph{
  background:rgba(15,23,42,.6);border:1px solid rgba(255,255,255,.06);
  border-radius:14px;padding:18px;margin-bottom:8px;color:#e2e8f0;
}
.kvr{
  display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);
}
.kvr:last-child{border-bottom:none;}
.kvr span:first-child{color:#475569;font-size:.8rem;}
.kvr span:last-child{font-weight:600;font-size:.82rem;
  font-family:'JetBrains Mono',monospace;}
.stTabs [data-baseweb="tab-list"]{
  background:rgba(15,23,42,.6);border-radius:12px;padding:4px;gap:4px;
  border:1px solid rgba(255,255,255,.06);
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px;color:#64748b;font-weight:500;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(90deg,#1e40af,#7c3aed) !important;
  color:#fff !important;
}
.stButton>button{
  background:linear-gradient(90deg,#1e40af,#7c3aed) !important;
  color:#fff !important;border:none !important;border-radius:12px !important;
  padding:14px 28px !important;font-weight:700 !important;
  width:100% !important;transition:all .3s !important;
  box-shadow:0 4px 20px rgba(124,58,237,.4) !important;
}
.stButton>button:hover{
  transform:translateY(-3px) !important;
  box-shadow:0 8px 30px rgba(124,58,237,.6) !important;
}
.foot{
  background:rgba(2,6,23,.8);border:1px solid rgba(255,255,255,.04);
  border-radius:12px;padding:18px;text-align:center;
  color:#334155;font-size:.7rem;margin-top:28px;
}
.timeline-item{
  border-left:3px solid var(--tc,#7c3aed);
  padding:12px 16px;margin-bottom:8px;
  background:rgba(15,23,42,.5);border-radius:0 10px 10px 0;
}
.ilo-card{
  background:linear-gradient(135deg,rgba(30,41,59,.6),rgba(15,23,42,.8));
  border:1px solid rgba(255,255,255,.06);border-radius:12px;
  padding:16px;margin-bottom:8px;
}
@keyframes heartbeat{
  0%,100%{transform:scale(1);}14%{transform:scale(1.1);}
  28%{transform:scale(1);}42%{transform:scale(1.1);}70%{transform:scale(1);}
}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#020617;}
::-webkit-scrollbar-thumb{
  background:linear-gradient(180deg,#1e40af,#7c3aed);border-radius:3px;
}
img{border-radius:12px !important;box-shadow:0 4px 20px rgba(0,0,0,.4) !important;}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f'Model load failed: {path}')
        return None

@st.cache_data
def load_csv(path):
    try: return pd.read_csv(path)
    except: return None

def mpath(name, tag):
    return os.path.join("models", f"{name}_{tag}.pkl")

def safe_img(path, cap=""):
    if os.path.exists(path):
        st.image(path, caption=cap, use_container_width=True)
    else:
        st.markdown(f"""
        <div class='infograph' style='text-align:center;padding:28px'>
          <div style='font-size:2rem'>📊</div>
          <div style='color:#475569;font-size:.85rem;margin-top:8px'>
            Run <code style='color:#7c3aed'>python main.py</code> to generate:<br>
            <code style='color:#94a3b8'>{path}</code>
          </div>
        </div>""", unsafe_allow_html=True)

def sec(title, emoji=""):
    t = f"{emoji} {title}" if emoji else title
    st.markdown(f"<div class='sec-hdr'>{t}</div>", unsafe_allow_html=True)

def kv(label, value, color="#e2e8f0"):
    st.markdown(
        f"<div class='kvr'><span>{label}</span>"
        f"<span style='color:{color}'>{value}</span></div>",
        unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class='foot'>
      ❤️ CVD Risk Prediction Using Machine Learning &nbsp;·&nbsp;
      PSB605IT &nbsp;·&nbsp; Sudhan Nagarajan (050J0DAD) &nbsp;·&nbsp;
      PSB Academy 2025/2026 &nbsp;·&nbsp;
      Supervisor: Mr Raymond Ching Chi Man<br>
      Hosmer et al.(2013) · Breiman(2001) · Chen&Guestrin(2016) ·
      Lundberg&Lee(2017) · Pedregosa et al.(2011) ·
      Efron&Tibshirani(1993) · WHO(2024) · Benjamin et al.(2019)
    </div>""", unsafe_allow_html=True)

def mpl_dark(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#0a0f1e")
    ax.tick_params(colors="#475569")
    for sp in ax.spines.values():
        sp.set_color("#1a1f2e")
    return fig, ax

def load_all_predict(features, tag, thresh=0.5):
    results = {}
    for mkey, mname in [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
    ]:
        mdl = load_model(mpath(mkey, tag))
        if mdl is not None:
            proba = mdl.predict_proba(features)[0]
            results[mname] = {
                "proba": proba,
                "pred": 1 if proba[1] >= thresh else 0
            }
    return results

def show_ensemble(results, thresh=0.5):
    if not results:
        st.error("No models found. Run python main.py first.")
        return
    avg_p = float(np.mean([v["proba"][1] for v in results.values()]))
    votes = sum(v["pred"] for v in results.values())
    pct = avg_p * 100
    if votes >= 2:
        banner, icon, lbl = "risk-high", "⚠️", "HIGH CVD RISK DETECTED"
        adv = f"{votes}/3 models predict CVD — consult a cardiologist immediately"
    elif avg_p >= 0.35:
        banner, icon, lbl = "risk-med", "🟡", "MODERATE CVD RISK"
        adv = "Risk factors detected — lifestyle modification recommended"
    else:
        banner, icon, lbl = "risk-low", "✅", "LOW CVD RISK"
        adv = f"{3-votes}/3 models predict No CVD — maintain healthy habits"
    st.markdown(f"""
    <div class='{banner}'>
      <div style='font-size:4rem;margin-bottom:10px'>{icon}</div>
      <h2 style='font-size:2rem;font-weight:900;margin:0'>{lbl}</h2>
      <p style='font-size:1.2rem;margin:12px 0 6px'>
        Ensemble CVD Probability:
        <strong style='font-size:1.8rem'>{pct:.1f}%</strong></p>
      <p style='font-size:.88rem;opacity:.8;margin:0'>
        Threshold: {thresh:.0%} &nbsp;·&nbsp; {adv}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    sec("Individual Model Predictions", "🤖")
    cols = st.columns(3)
    for col, (mname, res) in zip(cols, results.items()):
        p_cvd = float(res["proba"][1])
        p_nocvd = float(res["proba"][0])
        pred = res["pred"]
        clr = "#ef4444" if pred == 1 else "#22c55e"
        lbl2 = "⚠️ CVD" if pred == 1 else "✅ No CVD"
        with col:
            st.markdown(f"""
            <div class='model-card' style='border-color:{clr}'>
              <div style='color:#475569;font-size:.7rem;
                   text-transform:uppercase;letter-spacing:1.5px;
                   margin-bottom:8px'>{mname}</div>
              <div style='color:{clr};font-size:1.8rem;font-weight:900'>
                {lbl2}</div>
              <div style='color:#e2e8f0;font-size:.95rem;margin:10px 0 4px'>
                P(CVD) = {p_cvd:.4f}</div>
              <div style='color:#475569;font-size:.78rem'>
                P(No CVD) = {p_nocvd:.4f}</div>
              <div style='color:#64748b;font-size:.72rem;margin-top:6px'>
                Confidence: {max(p_cvd,p_nocvd)*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            st.progress(float(p_cvd))
    st.markdown("---")
    sec("Probability Comparison", "📊")
    names = list(results.keys())
    p_cvds = [float(results[n]["proba"][1]) for n in names]
    p_nocvds = [float(results[n]["proba"][0]) for n in names]
    fig, ax = mpl_dark(10, 4)
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, p_nocvds, w, label="P(No CVD)", color="#22c55e", alpha=0.85)
    b2 = ax.bar(x + w/2, p_cvds, w, label="P(CVD)", color="#ef4444", alpha=0.85)
    for bar, val in zip(b1, p_nocvds):
        ax.text(bar.get_x()+bar.get_width()/2, val+.012, f"{val:.4f}",
                ha="center", va="bottom", color="#22c55e", fontsize=9)
    for bar, val in zip(b2, p_cvds):
        ax.text(bar.get_x()+bar.get_width()/2, val+.012, f"{val:.4f}",
                ha="center", va="bottom", color="#ef4444", fontsize=9)
    ax.axhline(thresh, color="#f59e0b", lw=2, ls="--",
               label=f"Threshold={thresh:.0%}", alpha=.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="#94a3b8", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Probability", color="#64748b")
    ax.set_title("Model Probability Comparison", color="#e2e8f0",
                 fontsize=12, fontweight="bold")
    ax.legend(facecolor="#0a0f1e", labelcolor="#e2e8f0", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:22px 0 12px'>
      <div style='font-size:3.5rem;
           filter:drop-shadow(0 0 25px rgba(239,68,68,.8));
           animation:heartbeat 1.5s ease-in-out infinite'>❤️</div>
      <div style='font-size:1.1rem;font-weight:800;
           background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           margin-top:8px'>CVD Risk AI</div>
      <div style='font-size:.65rem;color:#1e293b;
           text-transform:uppercase;letter-spacing:2px;margin-top:4px'>
        PSB605IT · PSB Academy</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(30,64,175,.3);margin:10px 0'>
    """, unsafe_allow_html=True)

    PAGE = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🩺  Patient Predictor",
            "📂  Dataset Sources",
            "📊  Dataset Explorer",
            "🤖  Model Performance",
            "📈  Sampling Methods",
            "🔍  SHAP Explainability",
            "📋  Predictions Viewer",
            "⚖️  Model Comparison",
            "🎓  PSB605IT Module",
            "ℹ️  About",
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(30,64,175,.2);margin:10px 0'>
    """, unsafe_allow_html=True)

    DATASET = st.selectbox("🗄️ Dataset", [
        "CARDIO — Ulianova (2019)",
        "UCI — Detrano (1989)"
    ])
    TAG = "cardio" if "CARDIO" in DATASET else "uci"

    MODEL_CHOICE = st.selectbox("🤖 Model", [
        "Logistic Regression", "Random Forest", "XGBoost"
    ])
    MODEL_SLUG = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }[MODEL_CHOICE]

    THRESH = st.slider("🎯 Decision Threshold", 0.10, 0.90, 0.50, 0.01)

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(30,64,175,.2);margin:10px 0'>
    <div style='font-size:.65rem;color:#0f172a;text-align:center;line-height:1.6'>
      Sudhan Nagarajan · 050J0DAD<br>
      Mr Raymond Ching Chi Man<br>
      PSB Academy · 2025/2026
    </div>""", unsafe_allow_html=True)

# ================================================================
# PAGE — HOME
# ================================================================
if PAGE == "🏠  Home":
    st.markdown("""
    <div class='hero'>
      <div style='position:relative;z-index:1'>
        <div style='font-size:4rem;margin-bottom:14px;
             filter:drop-shadow(0 0 30px rgba(239,68,68,.9))'>
          ❤️ 🤖 🔬</div>
        <h1 style='
          background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 50%,#f472b6 100%);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          font-size:2.8rem;font-weight:900;margin:0;line-height:1.1'>
          CVD Risk Prediction</h1>
        <h2 style='color:#7c3aed;font-size:1.1rem;font-weight:600;
             margin:10px 0;letter-spacing:1px'>
          Using Machine Learning · PSB605IT</h2>
        <p style='color:#475569;font-size:.95rem;
             max-width:680px;margin:14px auto;line-height:1.7'>
          Predicting cardiovascular disease risk using
          <strong style='color:#60a5fa'>Logistic Regression</strong>,
          <strong style='color:#a78bfa'>Random Forest</strong> &amp;
          <strong style='color:#f472b6'>XGBoost</strong>
          across <strong style='color:#34d399'>7 sampling strategies</strong>
          on <strong style='color:#fbbf24'>2 clinical datasets</strong>.
        </p>
        <div style='margin-top:16px'>
          <div style='color:#334155;font-size:.75rem;margin-bottom:6px;
               text-transform:uppercase;letter-spacing:2px'>
            PSB605IT · BSc (Hons) Computing Science · Level 6 · FHEQ Level
          </div>
          <div style='display:flex;flex-wrap:wrap;justify-content:center;gap:8px'>
            <span class='pill'>🩺 2 Clinical Datasets</span>
            <span class='pill'>🤖 3 ML Models</span>
            <span class='pill'>🔬 7 Sampling Methods</span>
            <span class='pill'>📊 8 Evaluation Metrics</span>
            <span class='pill'>🧠 SHAP Explainability</span>
            <span class='pill'>🎯 Live Prediction</span>
            <span class='pill'>📱 Mobile Friendly</span>
            <span class='pill'>🌐 Open Source</span>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, lbl, c1c, c2c) in zip(
        [c1, c2, c3, c4, c5], [
        ("70,297", "Total Patients", "#3b82f6", "#60a5fa"),
        ("3", "ML Models", "#7c3aed", "#a78bfa"),
        ("7", "Sampling Methods", "#06b6d4", "#67e8f9"),
        ("8", "Eval Metrics", "#10b981", "#34d399"),
        ("2", "Datasets", "#f59e0b", "#fcd34d"),
    ]):
        col.markdown(f"""
        <div class='stat-card'
             style='--accent:linear-gradient(90deg,{c1c},{c2c})'>
          <div class='stat-num'
               style='background:linear-gradient(135deg,{c1c},{c2c});
               -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            {val}</div>
          <div class='stat-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sec("Champion Results from main.py", "🏆")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        <div class='glass'>
          <div style='color:#3b82f6;font-size:.75rem;
               text-transform:uppercase;letter-spacing:2px;margin-bottom:8px'>
            🫀 CARDIO — Ulianova (2019)</div>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model": ["★ Logistic Regression", "Random Forest", "XGBoost"],
            "AUC":   [0.7489, 0.7776, 0.7754],
            "CA":    [0.7025, 0.7000, 0.7000],
            "F1":    [0.7002, 0.6985, 0.6996],
            "MCC":   [0.4060, 0.3999, 0.3994],
            "LogLoss": [0.6061, 0.5700, 0.5695],
        }).style.highlight_max(
            subset=["AUC", "CA", "F1", "MCC"], color="#14532d"
        ).highlight_min(subset=["LogLoss"], color="#14532d"),
            use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class='glass'>
          <div style='color:#7c3aed;font-size:.75rem;
               text-transform:uppercase;letter-spacing:2px;margin-bottom:8px'>
            🔬 UCI — Detrano et al. (1989)</div>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model": ["★ Logistic Regression", "Random Forest", "XGBoost"],
            "AUC":   [0.9989, 0.9408, 0.9609],
            "CA":    [0.9667, 0.8500, 0.8833],
            "F1":    [0.9665, 0.8490, 0.8825],
            "MCC":   [0.9330, 0.6984, 0.7655],
            "LogLoss": [0.0943, 0.3525, 0.2586],
        }).style.highlight_max(
            subset=["AUC", "CA", "F1", "MCC"], color="#14532d"
        ).highlight_min(subset=["LogLoss"], color="#14532d"),
            use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    sec("7 Sampling Methods", "🔬")
    sm = [
        ("①", "Simple Holdout", "80/20 · 1 sample\nStratified · seed=42", "#3b82f6"),
        ("②", "Cross-Validation", "10 subsets · 1 unused\nStratified · seed=42", "#7c3aed"),
        ("③", "Bootstrap", "1 sample · With replacement\nOOB≈36.8%", "#06b6d4"),
        ("④", "Random Repeat", "10× · 66/34 split\nStratified", "#10b981"),
        ("⑤", "Leave-One-Out", "N folds (max 150)\ntrain=N-1 · test=1", "#f59e0b"),
        ("⑥", "Test on Test", "Final holdout\nTrue generalisation", "#ef4444"),
        ("⑦", "Test on Train", "Train evaluation\nOverfitting detection", "#ec4899"),
    ]
    row1 = st.columns(4)
    row2 = st.columns(3)
    for col, (n, m, d, c) in zip(row1 + row2, sm):
        col.markdown(f"""
        <div class='infograph' style='border-left:3px solid {c};min-height:130px'>
          <div style='color:{c};font-size:1.4rem;font-weight:900'>{n}</div>
          <div style='color:#e2e8f0;font-weight:700;font-size:.85rem;
               margin:4px 0'>{m}</div>
          <div style='color:#475569;font-size:.72rem;white-space:pre-line'>{d}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sec("ML Pipeline", "⚙️")
    p1, p2, p3, p4 = st.columns(4)
    for col, (n, icon, title, body, c) in zip([p1, p2, p3, p4], [
        ("01", "📥", "Data Ingestion", "2 datasets · 70,297 pts", "#3b82f6"),
        ("02", "🔬", "EDA & Preprocess", "Missing check\nOutlier removal", "#7c3aed"),
        ("03", "🤖", "Model Training", "LR · RF · XGBoost\n7 strategies", "#06b6d4"),
        ("04", "📊", "Evaluation", "8 metrics\nSHAP · CM", "#10b981"),
    ]):
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(15,23,42,.8),
             rgba(2,6,23,.9));border:1px solid rgba(255,255,255,.06);
             border-radius:16px;padding:20px;text-align:center;
             min-height:170px;border-top:2px solid {c}'>
          <div style='color:{c};font-size:.65rem;font-weight:700;
               letter-spacing:3px'>STEP {n}</div>
          <div style='font-size:2rem;margin:6px 0'>{icon}</div>
          <div style='color:#e2e8f0;font-weight:700;font-size:.88rem;
               margin:6px 0'>{title}</div>
          <div style='color:#334155;font-size:.72rem;white-space:pre-line'>{body}</div>
        </div>""", unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — PATIENT PREDICTOR
# ================================================================
elif PAGE == "🩺  Patient Predictor":
    st.markdown("""
    <div class='hero' style='padding:28px 24px'>
      <div style='position:relative;z-index:1'>
        <div style='font-size:2.5rem;margin-bottom:10px'>🩺</div>
        <h2 style='background:linear-gradient(135deg,#60a5fa,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             font-size:1.8rem;font-weight:900;margin:0'>
          Patient CVD Risk Predictor</h2>
        <p style='color:#475569;margin:8px 0 0;font-size:.9rem'>
          Enter patient vitals — all 3 trained models predict CVD risk
          with ensemble voting</p>
      </div>
    </div>""", unsafe_allow_html=True)

    ds_r = st.radio("Feature Set", [
        "🫀 CARDIO (Ulianova, 2019)",
        "🔬 UCI Heart Disease (Detrano, 1989)"
    ], horizontal=True)
    pred_tag = "cardio" if "CARDIO" in ds_r else "uci"

    st.markdown(f"""
    <div class='infograph'>
      <span style='color:#7c3aed;font-weight:600'>Threshold:</span>
      <span style='color:#e2e8f0'> {THRESH:.0%}</span>
      <span style='color:#334155;font-size:.8rem'> · Adjustable in sidebar</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    if pred_tag == "cardio":
        sec("CARDIO — Patient Health Metrics", "🫀")
        with st.form("cardio_f"):
            d1, d2, d3, d4 = st.columns(4)
            with d1: age_yr = st.number_input("Age (years)", 1, 100, 50, 1)
            with d2:
                gender = st.selectbox("Gender", [1, 2],
                    format_func=lambda x: "♀ Female" if x == 1 else "♂ Male")
            with d3: height = st.number_input("Height (cm)", 100, 250, 165, 1)
            with d4: weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, .5)
            bmi = weight / ((height / 100) ** 2)
            bmi_cat = ("🟢 Normal" if 18.5 <= bmi < 25 else
                       "🟡 Overweight" if 25 <= bmi < 30 else
                       "🔴 Obese" if bmi >= 30 else "🔵 Underweight")
            st.markdown(f"""
            <div class='infograph'>
              <b style='color:#7c3aed'>BMI:</b>
              <span style='color:#e2e8f0;font-size:1.1rem;font-weight:800;margin:0 12px'>
                {bmi:.1f} kg/m²</span>
              <span>{bmi_cat}</span>
            </div>""", unsafe_allow_html=True)
            bp1, bp2 = st.columns(2)
            with bp1: ap_hi = st.slider("Systolic BP (mmHg)", 60, 240, 120)
            with bp2: ap_lo = st.slider("Diastolic BP (mmHg)", 40, 160, 80)
            bp_s = ("🟢 Normal" if ap_hi < 120 and ap_lo < 80 else
                    "🟡 Elevated" if 120 <= ap_hi < 130 else
                    "🟠 High Stage 1" if 130 <= ap_hi < 140 else "🔴 High Stage 2")
            st.info(f"BP: **{bp_s}** — {ap_hi}/{ap_lo} mmHg")
            bl1, bl2 = st.columns(2)
            with bl1:
                chol = st.selectbox("Cholesterol", [1, 2, 3],
                    format_func=lambda x: {1:"1—Normal",2:"2—Above Normal",3:"3—Well Above"}[x])
            with bl2:
                gluc = st.selectbox("Glucose", [1, 2, 3],
                    format_func=lambda x: {1:"1—Normal",2:"2—Above Normal",3:"3—Well Above"}[x])
            ls1, ls2, ls3 = st.columns(3)
            with ls1:
                smoke = st.radio("🚬 Smoking", [0, 1],
                    format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker",
                    horizontal=True)
            with ls2:
                alco = st.radio("🍺 Alcohol", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    horizontal=True)
            with ls3:
                active = st.radio("🏋️ Active", [0, 1],
                    format_func=lambda x: "Inactive" if x == 0 else "Active",
                    horizontal=True)
            st.markdown("---")
            sub = st.form_submit_button(
                "🔮 PREDICT CVD RISK — ENSEMBLE OF 3 MODELS",
                use_container_width=True)
        if sub:
            features = np.array([[age_yr*365, gender, height, weight,
                                   ap_hi, ap_lo, chol, gluc, smoke, alco, active]])
            sec("Ensemble Prediction Results", "🎯")
            res = load_all_predict(features, "cardio", THRESH)
            show_ensemble(res, THRESH)
    else:
        sub_u = False
        sec("UCI Heart Disease — Patient Metrics", "🔬")
        with st.form("uci_f"):
            u1, u2, u3 = st.columns(3)
            with u1: u_age = st.number_input("Age", 20, 100, 55, 1)
            with u2:
                u_sex = st.selectbox("Sex", [0, 1],
                    format_func=lambda x: "♀ Female" if x == 0 else "♂ Male")
            with u3:
                u_cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                    format_func=lambda x: {0:"Typical Angina",1:"Atypical",
                        2:"Non-Anginal",3:"Asymptomatic"}[x])
            cv1, cv2, cv3 = st.columns(3)
            with cv1: u_bp = st.slider("Resting BP", 80, 220, 130)
            with cv2: u_ch = st.slider("Cholesterol (mg/dL)", 100, 600, 240)
            with cv3: u_hr = st.slider("Max Heart Rate", 60, 220, 150)
            cl1, cl2, cl3 = st.columns(3)
            with cl1:
                u_fbs = st.radio("Fasting BS>120", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
            with cl2:
                u_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                    format_func=lambda x: {0:"Normal",1:"ST-T Abnormal",
                        2:"LV Hypertrophy"}[x])
            with cl3:
                u_ex = st.radio("Exercise Angina", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1: u_op = st.number_input("ST Depression", 0.0, 10.0, 1.0, .1)
            with s2:
                u_sl = st.selectbox("ST Slope", [0, 1, 2],
                    format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
            with s3: u_ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
            with s4:
                u_th = st.selectbox("Thalassemia", [1, 2, 3],
                    format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible"}[x])
            st.markdown("---")
            sub_u = st.form_submit_button(
                "🔮 PREDICT CVD RISK — ENSEMBLE OF 3 MODELS",
                use_container_width=True)
        if sub_u:
            fu = np.array([[u_age, u_sex, u_cp, u_bp, u_ch,
                            u_fbs, u_ecg, u_hr, u_ex, u_op, u_sl, u_ca, u_th]])
            sec("Ensemble Prediction Results", "🎯")
            res_u = load_all_predict(fu, "uci", THRESH)
            show_ensemble(res_u, THRESH)
    footer()

# ================================================================
# PAGE — DATASET SOURCES
# ================================================================
elif PAGE == "📂  Dataset Sources":
    sec("Dataset Sources — Kaggle Repositories", "📂")
    st.markdown("""
    <div class='glass' style='border-left:4px solid #f59e0b'>
      <div style='color:#fbbf24;font-weight:800;font-size:1rem;margin-bottom:6px'>
        📦 Data Provenance & Citation</div>
      <div style='color:#94a3b8;font-size:.85rem;line-height:1.7'>
        Both datasets are publicly available on Kaggle and referenced in all
        academic outputs as required by PSB605IT guidelines.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='hero' style='padding:24px 28px;margin-bottom:16px;
         border-color:rgba(239,68,68,.4)'>
      <div style='position:relative;z-index:1;text-align:left'>
        <div style='display:flex;align-items:center;gap:16px'>
          <div style='font-size:2.5rem'>🫀</div>
          <div>
            <div style='color:#f87171;font-size:.7rem;
                 text-transform:uppercase;letter-spacing:2px'>
              Dataset 1 · PSB605IT Primary</div>
            <h2 style='background:linear-gradient(90deg,#f87171,#fb923c);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 font-size:1.5rem;font-weight:900;margin:4px 0'>
              Cardiovascular Disease Dataset</h2>
            <div style='color:#475569;font-size:.85rem'>
              Svetlana Ulianova · Kaggle · 2019</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='infograph'>", unsafe_allow_html=True)
        for k, v in {
            "Author": "Svetlana Ulianova",
            "Platform": "Kaggle",
            "File": "cardio_train.csv",
            "File Size": "2.94 MB",
            "Total Records": "70,000",
            "Columns": "13 (ID + 11 features + target)",
            "Target Variable": "cardio (0=No CVD, 1=CVD)",
            "Balance": "50% No-CVD / 50% CVD",
            "Missing Values": "0",
            "Usability Score": "6.47 / 10",
        }.items():
            kv(k, v, "#f87171")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='infograph'>
          <div style='color:#f87171;font-weight:700;margin-bottom:10px'>🔗 Kaggle Link</div>
          <a href='https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset'
             target='_blank'
             style='display:block;background:rgba(239,68,68,.15);
             border:1px solid rgba(239,68,68,.4);border-radius:10px;
             padding:12px 16px;color:#f87171;text-decoration:none;
             font-weight:600;font-size:.85rem;margin-bottom:12px'>
            🔗 kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
          </a>
          <div style='color:#475569;font-size:.78rem;line-height:1.7'>
            <b style='color:#94a3b8'>APA Citation:</b><br>
            Ulianova, S. (2019). Cardiovascular Disease Dataset. Kaggle.
          </div>
        </div>""", unsafe_allow_html=True)

    sec("Feature Dictionary — cardio_train.csv", "📋")
    st.dataframe(pd.DataFrame({
        "Column": ["id","age","gender","height","weight","ap_hi","ap_lo",
                   "cholesterol","gluc","smoke","alco","active","cardio"],
        "Type": ["Integer","Integer","Categorical","Integer","Float",
                 "Integer","Integer","Categorical","Categorical",
                 "Binary","Binary","Binary","Binary"],
        "Description": [
            "Patient ID (not used in model)",
            "Age in days",
            "1=Women, 2=Men",
            "Height in cm",
            "Weight in kg",
            "Systolic blood pressure",
            "Diastolic blood pressure",
            "1=Normal, 2=Above Normal, 3=Well Above Normal",
            "1=Normal, 2=Above Normal, 3=Well Above Normal",
            "0=Non-smoker, 1=Smoker",
            "0=No alcohol, 1=Alcohol",
            "0=Inactive, 1=Active",
            "0=No CVD, 1=CVD (TARGET)",
        ],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div class='hero' style='padding:24px 28px;margin-bottom:16px;
         border-color:rgba(124,58,237,.4)'>
      <div style='position:relative;z-index:1;text-align:left'>
        <div style='display:flex;align-items:center;gap:16px'>
          <div style='font-size:2.5rem'>🔬</div>
          <div>
            <div style='color:#a78bfa;font-size:.7rem;
                 text-transform:uppercase;letter-spacing:2px'>
              Dataset 2 · PSB605IT Secondary</div>
            <h2 style='background:linear-gradient(90deg,#a78bfa,#60a5fa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 font-size:1.5rem;font-weight:900;margin:4px 0'>
              UCI Heart Disease Data Set</h2>
            <div style='color:#475569;font-size:.85rem'>
              Detrano et al. (1989) · Cleveland Clinic Foundation</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    u1, u2 = st.columns(2)
    with u1:
        st.markdown("<div class='infograph'>", unsafe_allow_html=True)
        for k, v in {
            "Original Authors": "Detrano et al. (1989)",
            "File": "processed.cleveland.data",
            "File Size": "18.76 kB",
            "Total Records": "303",
            "After Cleaning": "297 (6 rows with ? removed)",
            "Columns": "14 (13 features + target)",
            "Target": "num (binarised: 0=No CVD, 1=CVD)",
            "Balance": "53.5% No-CVD / 46.5% CVD",
            "Collection": "May 1981 – September 1984",
            "Location": "Cleveland Clinic Foundation, Ohio, USA",
        }.items():
            kv(k, v, "#a78bfa")
        st.markdown("</div>", unsafe_allow_html=True)
    with u2:
        st.markdown("""
        <div class='infograph'>
          <div style='color:#a78bfa;font-weight:700;margin-bottom:10px'>🔗 Links</div>
          <a href='https://www.kaggle.com/datasets/lourenswalters/uci-heart-disease-data-set'
             target='_blank'
             style='display:block;background:rgba(124,58,237,.15);
             border:1px solid rgba(124,58,237,.4);border-radius:10px;
             padding:12px 16px;color:#a78bfa;text-decoration:none;
             font-weight:600;font-size:.85rem;margin-bottom:10px'>
            🔗 Kaggle: UCI Heart Disease Data Set
          </a>
          <a href='http://archive.ics.uci.edu/ml' target='_blank'
             style='display:block;background:rgba(30,41,59,.4);
             border:1px solid rgba(255,255,255,.06);border-radius:8px;
             padding:10px 14px;color:#60a5fa;text-decoration:none;
             font-size:.8rem;margin-bottom:10px'>
            🏛️ UCI ML Repository: archive.ics.uci.edu/ml
          </a>
          <div style='color:#475569;font-size:.75rem;line-height:1.7'>
            <b style='color:#94a3b8'>APA Citation:</b><br>
            Detrano, R., et al. (1989). International application of
            a new probability algorithm. AJC, 64(5), 304–310.
          </div>
        </div>""", unsafe_allow_html=True)

    sec("Feature Dictionary — processed.cleveland.data", "📋")
    st.dataframe(pd.DataFrame({
        "Column": ["age","sex","cp","trestbps","chol","fbs","restecg",
                   "thalach","exang","oldpeak","slope","ca","thal","num"],
        "Description": [
            "Age in years",
            "0=Female, 1=Male",
            "Chest pain: 1=Typical, 2=Atypical, 3=Non-Anginal, 4=Asymptomatic",
            "Resting BP (mmHg on admission)",
            "Serum cholesterol (mg/dl)",
            "Fasting blood sugar >120 mg/dl (1=True, 0=False)",
            "Resting ECG: 0=Normal, 1=ST-T Abnormal, 2=LV Hypertrophy",
            "Max heart rate achieved (bpm)",
            "Exercise induced angina (1=Yes, 0=No)",
            "ST depression induced by exercise vs rest",
            "ST slope: 1=Upsloping, 2=Flat, 3=Downsloping",
            "Major vessels (0–3) coloured by fluoroscopy",
            "Thalassemia: 3=Normal, 6=Fixed Defect, 7=Reversible",
            "Angiographic result (binarised: 0=No CVD, 1–4=CVD)",
        ],
    }), use_container_width=True, hide_index=True)
    footer()

# ================================================================
# PAGE — DATASET EXPLORER
# ================================================================
elif PAGE == "📊  Dataset Explorer":
    sec(f"Dataset Explorer — {DATASET}", "📊")
    t1, t2, t3 = st.tabs(["📈 EDA Charts", "🌡️ Correlation", "📋 Info"])
    with t1:
        safe_img(f"outputs/eda_{TAG}.png", "EDA Analysis")
    with t2:
        p = f"outputs/correlation_heatmap_{TAG}.png"
        if os.path.exists(p):
            safe_img(p, "Correlation Heatmap")
        else:
            st.info("Heatmap available for CARDIO dataset only.")
    with t3:
        info = ({
            "Source": "Ulianova, S. (2019) — Kaggle",
            "Total Instances": "70,000",
            "After Cleaning": "68,723",
            "Features": "11 input + 1 target",
            "Balance": "50.5% No-CVD / 49.5% CVD",
            "Missing Values": "0",
            "Train Split": "80% (stratified)",
            "Test Split": "20% (stratified)",
            "Seed": "42 (replicable)",
        } if TAG == "cardio" else {
            "Source": "Detrano et al. (1989) — UCI ML Repo",
            "Total Instances": "303",
            "After Cleaning": "297",
            "Features": "13 input + 1 target",
            "Balance": "53.5% No-CVD / 46.5% CVD",
            "Missing Values": "6 rows removed",
            "Train Split": "80% (stratified)",
            "Test Split": "20% (stratified)",
            "Seed": "42 (replicable)",
        })
        st.markdown("<div class='infograph'>", unsafe_allow_html=True)
        for k, v in info.items():
            kv(k, v, "#a78bfa")
        st.markdown("</div>", unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — MODEL PERFORMANCE
# ================================================================
elif PAGE == "🤖  Model Performance":
    sec(f"Model Performance — {DATASET} — {MODEL_CHOICE}", "🤖")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Heatmap", "📉 ROC", "🔥 Confusion Matrices",
        "📋 Metrics", "📈 Repeat 10×", "⚙️ Settings"])

    with t1:
        safe_img(f"outputs/eval_heatmap_{TAG}.png", "Heatmap")
        safe_img(f"outputs/sampling_compare_{TAG}.png", "Comparison")
    with t2:
        safe_img(f"outputs/roc_{TAG}.png", "ROC Curves")
        st.markdown("""
        <div class='infograph'>
          <b style='color:#e2e8f0'>AUC Guide:</b>&nbsp;
          <span class='pill' style='background:rgba(34,197,94,.2);
            border-color:rgba(34,197,94,.4);color:#4ade80'>
            🟢 0.9–1.0 Excellent</span>
          <span class='pill' style='background:rgba(59,130,246,.2);
            border-color:rgba(59,130,246,.4);color:#60a5fa'>
            🔵 0.8–0.9 Good</span>
          <span class='pill' style='background:rgba(245,158,11,.2);
            border-color:rgba(245,158,11,.4);color:#fbbf24'>
            🟡 0.7–0.8 Fair</span>
          <span class='pill' style='background:rgba(239,68,68,.2);
            border-color:rgba(239,68,68,.4);color:#f87171'>
            🔴 0.6–0.7 Poor</span>
          <span class='pill'>⚫ 0.5 Random</span>
        </div>""", unsafe_allow_html=True)
    with t3:
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            st.markdown("**Logistic Regression**")
            safe_img(f"outputs/cm_logistic_regression_{TAG}.png")
        with cm2:
            st.markdown("**Random Forest**")
            safe_img(f"outputs/cm_random_forest_{TAG}.png")
        with cm3:
            st.markdown("**XGBoost**")
            safe_img(f"outputs/cm_xgboost_{TAG}.png")
    with t4:
        df_met = load_csv(f"outputs/metrics_{TAG}.csv")
        if df_met is not None:
            st.dataframe(df_met.style.highlight_max(
                subset=[c for c in df_met.columns if c not in ["Model", "LogLoss"]],
                color="#14532d").highlight_min(
                subset=["LogLoss"] if "LogLoss" in df_met.columns else [],
                color="#14532d"),
                use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download CSV",
                df_met.to_csv(index=False), f"metrics_{TAG}.csv", "text/csv")
        if TAG == "cardio":
            rows = {"Model": ["★ LR", "RF", "XGB"],
                    "AUC": [0.7489, 0.7776, 0.7754],
                    "CA":  [0.7025, 0.7000, 0.7000],
                    "F1":  [0.7002, 0.6985, 0.6996],
                    "Prec":[0.7052, 0.7012, 0.6998],
                    "Recall":[0.7008, 0.6987, 0.6996],
                    "MCC": [0.4060, 0.3999, 0.3994],
                    "Spec":[0.7707, 0.7512, 0.7171],
                    "LogLoss":[0.6061, 0.5700, 0.5695]}
        else:
            rows = {"Model": ["★ LR", "RF", "XGB"],
                    "AUC": [0.9989, 0.9408, 0.9609],
                    "CA":  [0.9667, 0.8500, 0.8833],
                    "F1":  [0.9665, 0.8490, 0.8825],
                    "Prec":[0.9665, 0.8502, 0.8838],
                    "Recall":[0.9665, 0.8482, 0.8817],
                    "MCC": [0.9330, 0.6984, 0.7655],
                    "Spec":[0.9688, 0.8750, 0.9062],
                    "LogLoss":[0.0943, 0.3525, 0.2586]}
        sec("Holdout 80/20 Results", "")
        st.dataframe(pd.DataFrame(rows).style.highlight_max(
            subset=["AUC","CA","F1","Prec","Recall","MCC","Spec"],
            color="#14532d").highlight_min(
            subset=["LogLoss"], color="#14532d"),
            use_container_width=True, hide_index=True)
    with t5:
        r1c, r2c, r3c = st.columns(3)
        with r1c: safe_img(f"outputs/repeat_logistic_regression_{TAG}.png", "LR")
        with r2c: safe_img(f"outputs/repeat_random_forest_{TAG}.png", "RF")
        with r3c: safe_img(f"outputs/repeat_xgboost_{TAG}.png", "XGB")
    with t6:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='infograph'>", unsafe_allow_html=True)
            st.markdown("<div style='color:#3b82f6;font-weight:700;margin-bottom:8px'>Logistic Regression</div>",
                        unsafe_allow_html=True)
            for k, v in [
                ("Regularization Type", "Ridge (L2)"),
                ("Strength", "C=1.0"),
                ("Balance", "class_weight=balanced"),
                ("Solver", "lbfgs · max_iter=1000"),
                ("Replicable", "random_state=42"),
            ]: kv(k, v, "#60a5fa")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='infograph'>", unsafe_allow_html=True)
            st.markdown("<div style='color:#7c3aed;font-weight:700;margin-bottom:8px'>Random Forest</div>",
                        unsafe_allow_html=True)
            for k, v in [
                ("Number of Trees", "100"),
                ("Replicable", "random_state=42"),
                ("Balance", "class_weight=balanced"),
                ("Growth Control", "min_samples_split=5"),
                ("n_jobs", "-1 (all cores)"),
            ]: kv(k, v, "#a78bfa")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='infograph'>", unsafe_allow_html=True)
            st.markdown("<div style='color:#06b6d4;font-weight:700;margin-bottom:8px'>XGBoost</div>",
                        unsafe_allow_html=True)
            for k, v in [
                ("n_estimators", "100"),
                ("scale_pos_weight", "auto (n_neg/n_pos)"),
                ("eval_metric", "logloss"),
                ("Replicable", "random_state=42"),
                ("verbosity", "0 (silent)"),
            ]: kv(k, v, "#67e8f9")
            st.markdown("</div>", unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — SAMPLING METHODS
# ================================================================
elif PAGE == "📈  Sampling Methods":
    sec(f"Sampling Methods — {DATASET}", "📈")
    t1, t2, t3, t4 = st.tabs([
        "📊 Compare", "ℹ️ Method Details",
        "📏 Metric Definitions", "🔁 Overfitting"])

    with t1:
        sa, sb = st.columns(2)
        with sa: safe_img(f"outputs/sampling_compare_{TAG}.png")
        with sb: safe_img(f"outputs/proportion_{TAG}.png")

    with t2:
        methods = [
            ("①", "Simple Holdout",
             "Fixed proportion: 80/20 · 1 sample\nWith Replacement: No\nReplicable: Yes seed=42\nStratified: Yes",
             "#3b82f6", "Pedregosa et al. (2011)"),
            ("②", "Cross-Validation (10-Fold)",
             "Subsets: 10 · Unused: 1 per round\nStratified: Yes · seed=42",
             "#7c3aed", "Pedregosa et al. (2011)"),
            ("③", "Bootstrap",
             "Samples: 1 · With Replacement: Yes\nReplicable: seed=42 · OOB≈36.8%",
             "#06b6d4", "Efron & Tibshirani (1993)"),
            ("④", "Random Repeat Train/Test",
             "Repeats: 10 · 66/34 split\nStratified: Yes · seed=42+rep",
             "#10b981", "Pedregosa et al. (2011)"),
            ("⑤", "Leave-One-Out",
             "Folds: N (max 150)\nTrain: N-1 · Test: 1",
             "#f59e0b", "Pedregosa et al. (2011)"),
            ("⑥", "Test on Test Data",
             "Final holdout evaluation\nNever seen in training",
             "#ef4444", "Hosmer et al. (2013)"),
            ("⑦", "Test on Train Data",
             "Training set evaluation\nOverfitting detection",
             "#ec4899", "Breiman (2001)"),
        ]
        for n, m, d, c, ref in methods:
            st.markdown(f"""
            <div class='glass' style='border-left:4px solid {c};
                 border-radius:0 16px 16px 0;margin-bottom:10px'>
              <div style='display:flex;align-items:flex-start;gap:16px'>
                <div style='color:{c};font-size:1.8rem;font-weight:900;
                     min-width:44px;text-align:center'>{n}</div>
                <div>
                  <div style='color:#e2e8f0;font-weight:700;
                       font-size:.95rem;margin-bottom:6px'>{m}</div>
                  <div style='color:#475569;font-size:.78rem;
                       white-space:pre-line;line-height:1.7'>{d}</div>
                  <div style='color:rgba(124,58,237,.6);font-size:.7rem;
                       margin-top:6px'>📚 {ref}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t3:
        for abbr, full, desc, c in [
            ("AUC", "Area Under ROC Curve", "1.0=Perfect · 0.5=Random", "#3b82f6"),
            ("CA", "Classification Accuracy", "(TP+TN)/(Total)", "#7c3aed"),
            ("F1", "F1-Score (Macro)", "Harmonic mean Precision & Recall", "#06b6d4"),
            ("Prec", "Precision", "TP/(TP+FP)", "#10b981"),
            ("Recall", "Recall / Sensitivity", "TP/(TP+FN)", "#f59e0b"),
            ("MCC", "Matthews Correlation", "−1 to +1 · balanced metric", "#ef4444"),
            ("Spec", "Specificity", "TN/(TN+FP)", "#ec4899"),
            ("LogLoss", "Logarithmic Loss", "Lower=better · penalises confidence", "#8b5cf6"),
        ]:
            st.markdown(f"""
            <div class='glass' style='border-left:3px solid {c};
                 padding:14px;margin-bottom:6px'>
              <div style='display:flex;align-items:center;gap:12px'>
                <div style='background:{c}22;border:1px solid {c}66;
                     border-radius:8px;padding:6px 10px;font-weight:900;
                     font-size:.9rem;color:{c};min-width:70px;text-align:center'>
                  {abbr}</div>
                <div>
                  <div style='color:#e2e8f0;font-weight:600;font-size:.85rem'>
                    {full}</div>
                  <div style='color:#475569;font-size:.73rem;margin-top:3px'>
                    {desc}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t4:
        sec("Overfitting Detection — Train vs Test", "🔁")
        ov = pd.DataFrame({
            "Model + Dataset": [
                "LR — CARDIO", "RF — CARDIO", "XGB — CARDIO",
                "LR — UCI", "RF — UCI", "XGB — UCI"],
            "Train Acc": [0.6981, 0.7669, 0.7731, 0.9662, 0.9958, 1.0000],
            "Test Acc":  [0.7025, 0.7000, 0.7000, 0.9667, 0.8500, 0.8833],
            "Acc Gap":   [-0.0044, +0.0669, +0.0731, -0.0004, +0.1458, +0.1167],
            "Train F1":  [0.6970, 0.7644, 0.7727, 0.9661, 0.9958, 1.0000],
            "Test F1":   [0.7002, 0.6985, 0.6996, 0.9665, 0.8490, 0.8825],
            "Status":    ["✓ Good", "✗ Overfit", "✗ Overfit",
                          "✓ Good", "✗ Overfit", "✗ Overfit"],
        })
        def col_st(v):
            if "✓" in str(v): return "background:#14532d;color:#86efac"
            return "background:#7f1d1d;color:#fca5a5"
        st.dataframe(ov.style.map(col_st, subset=["Status"]),
                     use_container_width=True, hide_index=True)
    footer()

# ================================================================
# PAGE — SHAP EXPLAINABILITY
# ================================================================
elif PAGE == "🔍  SHAP Explainability":
    sec(f"SHAP Explainability — {DATASET}", "🔍")
    st.markdown("""
    <div class='glass'>
      <div style='display:flex;align-items:center;gap:16px'>
        <div style='font-size:2.5rem'>🧠</div>
        <div>
          <div style='color:#a78bfa;font-weight:800;font-size:1.1rem'>
            SHAP — SHapley Additive exPlanations</div>
          <div style='color:#475569;font-size:.82rem;margin-top:4px'>
            Lundberg & Lee (2017) · NeurIPS</div>
          <div style='margin-top:10px;display:flex;gap:8px;flex-wrap:wrap'>
            <span class='pill' style='background:rgba(239,68,68,.15);
              border-color:rgba(239,68,68,.3);color:#f87171'>
              🔴 Positive SHAP → pushes toward CVD</span>
            <span class='pill' style='background:rgba(59,130,246,.15);
              border-color:rgba(59,130,246,.3);color:#60a5fa'>
              🔵 Negative SHAP → pushes toward No CVD</span>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    safe_img(f"outputs/shap_{TAG}.png", "SHAP Summary Plot")
    sec("Feature Importance Interpretation", "📊")

    feats = (
        [("ap_hi", "Systolic BP", "#ef4444",
          "STRONGEST predictor. High systolic BP greatly increases CVD risk.", "Very High"),
         ("age", "Age (days)", "#f59e0b",
          "Older patients consistently show higher CVD risk.", "High"),
         ("ap_lo", "Diastolic BP", "#f97316",
          "High diastolic compounds systolic effect.", "High"),
         ("cholesterol", "Cholesterol Level", "#8b5cf6",
          "Level 2 & 3 linked to CVD.", "Medium-High"),
         ("weight", "Body Weight", "#06b6d4",
          "Higher weight increases cardiovascular strain.", "Medium"),
         ("gluc", "Blood Glucose", "#3b82f6",
          "Elevated glucose indicates metabolic syndrome risk.", "Medium"),
         ("active", "Physical Activity", "#22c55e",
          "Activity is PROTECTIVE — reduces CVD probability.", "Protective"),
         ("smoke", "Smoking Status", "#ec4899",
          "Smoking damages blood vessels.", "Medium"),
         ] if TAG == "cardio" else
        [("ca", "Major Vessels (0–3)", "#ef4444",
          "STRONGEST UCI predictor. More blocked vessels = higher risk.", "Very High"),
         ("thal", "Thalassemia Type", "#f59e0b",
          "Reversible defect (type 3) strongly predicts CVD.", "Very High"),
         ("cp", "Chest Pain Type", "#f97316",
          "Asymptomatic (type 3) paradoxically indicates highest risk.", "High"),
         ("oldpeak", "ST Depression", "#8b5cf6",
          "Indicates myocardial ischaemia.", "High"),
         ("thalach", "Max Heart Rate", "#06b6d4",
          "Lower max HR = higher CVD risk.", "Medium-High"),
         ("exang", "Exercise Angina", "#3b82f6",
          "Chest pain on exercise = strong CVD indicator.", "Medium-High"),
         ("age", "Age (years)", "#10b981",
          "CVD risk increases with age.", "Medium"),
         ("sex", "Sex (Male=1)", "#22c55e",
          "Males show higher CVD risk in UCI dataset.", "Medium"),
         ]
    )
    fc1, fc2 = st.columns(2)
    for i, (feat, name, c, desc, impact) in enumerate(feats):
        col = fc1 if i % 2 == 0 else fc2
        col.markdown(f"""
        <div class='infograph' style='border-left:3px solid {c};margin-bottom:8px'>
          <div style='display:flex;justify-content:space-between'>
            <span style='color:{c};font-weight:800'>{feat}</span>
            <span style='background:{c}22;border:1px solid {c}44;
                 border-radius:6px;padding:2px 8px;
                 font-size:.68rem;color:{c}'>{impact}</span>
          </div>
          <div style='color:#64748b;font-size:.78rem;margin-top:2px'>{name}</div>
          <div style='color:#475569;font-size:.73rem;margin-top:5px'>{desc}</div>
        </div>""", unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — PREDICTIONS VIEWER
# ================================================================
elif PAGE == "📋  Predictions Viewer":
    sec(f"Predictions Viewer — {MODEL_CHOICE} | {DATASET}", "📋")
    st.markdown(f"""
    <div class='infograph'>
      <div style='display:flex;gap:20px;flex-wrap:wrap;align-items:center'>
        <div><span style='color:#475569;font-size:.75rem'>Model</span><br>
          <span style='color:#a78bfa;font-weight:700'>{MODEL_CHOICE}</span></div>
        <div><span style='color:#475569;font-size:.75rem'>Dataset</span><br>
          <span style='color:#e2e8f0;font-weight:700'>{DATASET}</span></div>
        <div><span style='color:#475569;font-size:.75rem'>Threshold</span><br>
          <span style='color:#34d399;font-weight:700'>{THRESH:.0%}</span></div>
        <div><span style='color:#475569;font-size:.75rem'>Classes</span><br>
          <span style='color:#e2e8f0;font-weight:700'>[0, 1] → No CVD · CVD</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    df_p = load_csv(f"outputs/predictions_{MODEL_SLUG}_{TAG}.csv")

    if df_p is not None:
        n = len(df_p)
        result_col = next((c for c in df_p.columns
                           if "result" in c.lower()), None)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Instances", f"{n:,}")
        if result_col:
            correct = int(df_p[result_col].str.contains("CORRECT", na=False).sum())
            wrong = n - correct
            sc2.metric("✅ Correct", f"{correct:,}")
            sc3.metric("❌ Misclassified", f"{wrong:,}")
            sc4.metric("🎯 Accuracy", f"{correct/n*100:.2f}%")

        if "Actual" in df_p.columns:
            n0a = int((df_p["Actual"] == "No CVD (0)").sum())
            n1a = int((df_p["Actual"] == "CVD (1)").sum())
            n0p = (int((df_p["Predicted"] == "No CVD (0)").sum())
                   if "Predicted" in df_p.columns else 0)
            n1p = (int((df_p["Predicted"] == "CVD (1)").sum())
                   if "Predicted" in df_p.columns else 0)
            st.markdown(f"""
            <div class='infograph' style='margin-top:10px'>
              <div style='color:#e2e8f0;font-weight:700;margin-bottom:8px'>
                Proportion of Actual / Predicted</div>
              <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                   gap:10px;text-align:center'>
                <div><div style='color:#475569;font-size:.72rem;
                     text-transform:uppercase'>N Actual No CVD</div>
                  <div style='color:#22c55e;font-size:1.3rem;font-weight:800'>{n0a:,}</div>
                  <div style='color:#334155;font-size:.72rem'>Prop: {n0a/n:.4f}</div></div>
                <div><div style='color:#475569;font-size:.72rem;
                     text-transform:uppercase'>N Actual CVD</div>
                  <div style='color:#ef4444;font-size:1.3rem;font-weight:800'>{n1a:,}</div>
                  <div style='color:#334155;font-size:.72rem'>Prop: {n1a/n:.4f}</div></div>
                <div><div style='color:#475569;font-size:.72rem;
                     text-transform:uppercase'>N Pred No CVD</div>
                  <div style='color:#22c55e;font-size:1.3rem;font-weight:800'>{n0p:,}</div>
                  <div style='color:#334155;font-size:.72rem'>Prop: {n0p/n:.4f}</div></div>
                <div><div style='color:#475569;font-size:.72rem;
                     text-transform:uppercase'>N Pred CVD</div>
                  <div style='color:#ef4444;font-size:1.3rem;font-weight:800'>{n1p:,}</div>
                  <div style='color:#334155;font-size:.72rem'>Prop: {n1p/n:.4f}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        if "P_CVD" in df_p.columns and "P_NoCVD" in df_p.columns:
            s0 = df_p["P_NoCVD"].sum()
            s1 = df_p["P_CVD"].sum()
            st.markdown(f"""
            <div class='infograph'>
              <div style='color:#e2e8f0;font-weight:700;margin-bottom:6px'>
                Sum of Probabilities</div>
              <div style='display:flex;gap:24px;flex-wrap:wrap'>
                <div><span style='color:#475569;font-size:.8rem'>Sum P(No CVD):</span>
                  <span style='color:#22c55e;font-weight:700;margin-left:8px'>
                    {s0:.4f}</span></div>
                <div><span style='color:#475569;font-size:.8rem'>Sum P(CVD):</span>
                  <span style='color:#ef4444;font-weight:700;margin-left:8px'>
                    {s1:.4f}</span></div>
                <div><span style='color:#475569;font-size:.8rem'>Sum P total:</span>
                  <span style='color:#a78bfa;font-weight:700;margin-left:8px'>
                    {s0+s1:.4f} (= {n:,} × 1.0000)</span></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        sec("Filter Predictions", "🔍")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            filt_r = st.selectbox("Result Filter", [
                "All — Clear Selection",
                "✅ Select CORRECT Only",
                "❌ Select MISCLASSIFIED Only"])
        with f2:
            filt_a = "All"
            if "Actual" in df_p.columns:
                filt_a = st.selectbox("Actual Class Filter",
                    ["All"] + sorted(df_p["Actual"].unique().tolist()))
        with f3:
            filt_pred = "All"
            if "Predicted" in df_p.columns:
                filt_pred = st.selectbox("Predicted Class Filter",
                    ["All"] + sorted(df_p["Predicted"].unique().tolist()))
        with f4:
            n_rows = st.slider("Rows to show", 10, min(2000, n), min(100, n))

        df_d = df_p.copy()
        if "CORRECT Only" in filt_r and result_col:
            df_d = df_d[df_d[result_col].str.contains("CORRECT", na=False) &
                        ~df_d[result_col].str.contains("MISCLASSIFIED", na=False)]
        elif "MISCLASSIFIED Only" in filt_r and result_col:
            df_d = df_d[df_d[result_col].str.contains("MISCLASSIFIED", na=False)]
        if filt_a != "All" and "Actual" in df_d.columns:
            df_d = df_d[df_d["Actual"] == filt_a]
        if filt_pred != "All" and "Predicted" in df_d.columns:
            df_d = df_d[df_d["Predicted"] == filt_pred]

        n_filt = len(df_d)
        n_corr_f = (int(df_d[result_col].str.contains("CORRECT", na=False).sum())
                    if result_col else 0)
        st.markdown(f"""
        <div class='infograph'>
          Showing <strong style='color:#7c3aed'>{n_filt:,}</strong>
          of <strong>{n:,}</strong> &nbsp;·&nbsp;
          <span style='color:#22c55e'>✅ {n_corr_f:,} correct</span>
          &nbsp;·&nbsp;
          <span style='color:#ef4444'>❌ {n_filt-n_corr_f:,} misclassified</span>
        </div>""", unsafe_allow_html=True)

        def color_result(val):
            if "CORRECT" in str(val) and "MIS" not in str(val):
                return "background:#14532d;color:#86efac;font-weight:600"
            elif "MISCLASSIFIED" in str(val):
                return "background:#7f1d1d;color:#fca5a5;font-weight:600"
            return ""

        df_show = df_d.head(n_rows)
        if result_col and result_col in df_show.columns:
            styled = df_show.style.map(color_result, subset=[result_col])
            if "P_CVD" in df_show.columns:
                styled = styled.background_gradient(
                    subset=["P_CVD"], cmap="RdYlGn_r", vmin=0, vmax=1)
            st.dataframe(styled, use_container_width=True, height=420)
        else:
            st.dataframe(df_show, use_container_width=True, height=420)

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("⬇️ Download Filtered",
                df_d.to_csv(index=False).encode("utf-8"),
                f"predictions_{MODEL_SLUG}_{TAG}_filtered.csv", "text/csv")
        with dl2:
            if result_col:
                c_df = df_p[df_p[result_col].str.contains("CORRECT", na=False) &
                            ~df_p[result_col].str.contains("MISCLASSIFIED", na=False)]
                st.download_button("⬇️ Download Correct Only",
                    c_df.to_csv(index=False).encode("utf-8"),
                    f"correct_{MODEL_SLUG}_{TAG}.csv", "text/csv")
        with dl3:
            if result_col:
                m_df = df_p[df_p[result_col].str.contains("MISCLASSIFIED", na=False)]
                st.download_button("⬇️ Download Misclassified",
                    m_df.to_csv(index=False).encode("utf-8"),
                    f"misclassified_{MODEL_SLUG}_{TAG}.csv", "text/csv")

        if "P_CVD" in df_p.columns:
            st.markdown("---")
            sec("Probability Distribution", "📊")
            fig_d, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig_d.patch.set_facecolor("#020617")
            for ax in axes:
                ax.set_facecolor("#0a0f1e")
                ax.tick_params(colors="#475569")
                for sp in ax.spines.values(): sp.set_color("#1a1f2e")
            if result_col:
                cp_ = df_p[df_p[result_col].str.contains("CORRECT", na=False) &
                           ~df_p[result_col].str.contains("MISCLASSIFIED", na=False)]["P_CVD"]
                mp_ = df_p[df_p[result_col].str.contains("MISCLASSIFIED", na=False)]["P_CVD"]
                axes[0].hist(cp_, bins=25, color="#22c55e", alpha=0.75,
                             label=f"Correct ({len(cp_)})")
                axes[0].hist(mp_, bins=25, color="#ef4444", alpha=0.75,
                             label=f"Misclassified ({len(mp_)})")
                axes[0].axvline(THRESH, color="#f59e0b", lw=2.5, ls="--",
                                label=f"Threshold={THRESH:.0%}")
                axes[0].set_xlabel("P(CVD)", color="#64748b")
                axes[0].set_ylabel("Count", color="#64748b")
                axes[0].set_title("P(CVD) by Result",
                                  color="#e2e8f0", fontweight="bold")
                axes[0].legend(facecolor="#0a0f1e", labelcolor="#e2e8f0", fontsize=8)
            if "Actual" in df_p.columns:
                cvd_p = df_p[df_p["Actual"] == "CVD (1)"]["P_CVD"]
                no_p  = df_p[df_p["Actual"] == "No CVD (0)"]["P_CVD"]
                axes[1].hist(no_p, bins=25, color="#22c55e", alpha=0.75,
                             label=f"Actual No CVD ({len(no_p)})")
                axes[1].hist(cvd_p, bins=25, color="#ef4444", alpha=0.75,
                             label=f"Actual CVD ({len(cvd_p)})")
                axes[1].axvline(THRESH, color="#f59e0b", lw=2.5, ls="--")
                axes[1].set_xlabel("P(CVD)", color="#64748b")
                axes[1].set_ylabel("Count", color="#64748b")
                axes[1].set_title("P(CVD) by True Class",
                                  color="#e2e8f0", fontweight="bold")
                axes[1].legend(facecolor="#0a0f1e", labelcolor="#e2e8f0", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close()
    else:
        st.markdown("""
        <div class='infograph' style='text-align:center;padding:40px'>
          <div style='font-size:3rem;margin-bottom:12px'>📋</div>
          <div style='color:#e2e8f0;font-weight:600'>No predictions found</div>
          <div style='color:#475569;margin-top:8px;font-size:.85rem'>
            Run <code style='color:#7c3aed'>python main.py</code>
            to generate prediction files
          </div>
        </div>""", unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — MODEL COMPARISON
# ================================================================
elif PAGE == "⚖️  Model Comparison":
    sec("Model Comparison — All Models | Both Datasets", "⚖️")
    t1, t2, t3, t4 = st.tabs([
        "📊 Side-by-Side", "📈 Bar Charts",
        "🕸️ Radar Chart", "🔁 Overfitting"])

    with t1:
        a, b = st.columns(2)
        with a:
            st.markdown("""
            <div class='glass'>
              <div style='color:#3b82f6;font-size:.75rem;
                   text-transform:uppercase;letter-spacing:2px;margin-bottom:8px'>
                🫀 CARDIO Dataset</div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Model": ["★ LR", "RF", "XGB"],
                "AUC":   [0.7489, 0.7776, 0.7754],
                "CA":    [0.7025, 0.7000, 0.7000],
                "F1":    [0.7002, 0.6985, 0.6996],
                "MCC":   [0.4060, 0.3999, 0.3994],
                "LogLoss": [0.6061, 0.5700, 0.5695],
            }).style.highlight_max(
                subset=["AUC","CA","F1","MCC"], color="#14532d"
            ).highlight_min(subset=["LogLoss"], color="#14532d"),
                use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.markdown("""
            <div class='glass'>
              <div style='color:#7c3aed;font-size:.75rem;
                   text-transform:uppercase;letter-spacing:2px;margin-bottom:8px'>
                🔬 UCI Dataset</div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Model": ["★ LR", "RF", "XGB"],
                "AUC":   [0.9989, 0.9408, 0.9609],
                "CA":    [0.9667, 0.8500, 0.8833],
                "F1":    [0.9665, 0.8490, 0.8825],
                "MCC":   [0.9330, 0.6984, 0.7655],
                "LogLoss": [0.0943, 0.3525, 0.2586],
            }).style.highlight_max(
                subset=["AUC","CA","F1","MCC"], color="#14532d"
            ).highlight_min(subset=["LogLoss"], color="#14532d"),
                use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#020617")
        for ax, (data, title) in zip(axes, [
            ({"Metric": ["AUC","CA","F1","MCC"],
              "LR": [0.7489,0.7025,0.7002,0.4060],
              "RF": [0.7776,0.7000,0.6985,0.3999],
              "XGB":[0.7754,0.7000,0.6996,0.3994]}, "CARDIO"),
            ({"Metric": ["AUC","CA","F1","MCC"],
              "LR": [0.9989,0.9667,0.9665,0.9330],
              "RF": [0.9408,0.8500,0.8490,0.6984],
              "XGB":[0.9609,0.8833,0.8825,0.7655]}, "UCI"),
        ]):
            ax.set_facecolor("#0a0f1e")
            x = np.arange(len(data["Metric"]))
            w = 0.25
            for i, (col, c) in enumerate(zip(
                ["LR","RF","XGB"], ["#3b82f6","#10b981","#a78bfa"])):
                bars = ax.bar(x+i*w, data[col], w, label=col, color=c, alpha=0.85)
                for bar, val in zip(bars, data[col]):
                    ax.text(bar.get_x()+bar.get_width()/2, val+.005,
                            f"{val:.3f}", ha="center", va="bottom",
                            color="#94a3b8", fontsize=8)
            ax.set_xticks(x+w)
            ax.set_xticklabels(data["Metric"], color="#64748b", fontsize=9)
            ax.set_title(title, color="#e2e8f0", fontweight="bold", fontsize=12)
            ax.tick_params(colors="#475569")
            for sp in ax.spines.values(): sp.set_color("#1a1f2e")
            ax.set_ylim(0, 1.15)
            ax.legend(facecolor="#0a0f1e", labelcolor="#e2e8f0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with t3:
        cats = ["AUC","CA","F1","MCC","Spec"]
        N = len(cats)
        angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5),
                                   subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("#020617")
        ds_r = {
            "CARDIO": {
                "LR":  [0.7489,0.7025,0.7002,0.4060,0.7707],
                "RF":  [0.7776,0.7000,0.6985,0.3999,0.7512],
                "XGB": [0.7754,0.7000,0.6996,0.3994,0.7171],
            },
            "UCI": {
                "LR":  [0.9989,0.9667,0.9665,0.9330,0.9688],
                "RF":  [0.9408,0.8500,0.8490,0.6984,0.8750],
                "XGB": [0.9609,0.8833,0.8825,0.7655,0.9062],
            }
        }
        clrs_r = {"LR":"#3b82f6","RF":"#10b981","XGB":"#a78bfa"}
        for ax2, (dn, dv) in zip(axes2, ds_r.items()):
            ax2.set_facecolor("#0a0f1e")
            ax2.set_title(dn, color="#e2e8f0", fontweight="bold", pad=18)
            ax2.set_thetagrids(np.degrees(angles[:-1]), cats,
                               color="#64748b", fontsize=8)
            ax2.set_ylim(0, 1)
            ax2.spines["polar"].set_color("#1a1f2e")
            ax2.tick_params(colors="#334155")
            for mn, vals in dv.items():
                v = vals + vals[:1]
                ax2.plot(angles, v, "o-", lw=2.5, label=mn,
                         color=clrs_r[mn], markersize=5)
                ax2.fill(angles, v, alpha=.12, color=clrs_r[mn])
            ax2.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
                       facecolor="#0a0f1e", labelcolor="#e2e8f0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with t4:
        ov = pd.DataFrame({
            "Model + Dataset": [
                "LR — CARDIO","RF — CARDIO","XGB — CARDIO",
                "LR — UCI","RF — UCI","XGB — UCI"],
            "Train Acc": [0.6981,0.7669,0.7731,0.9662,0.9958,1.0000],
            "Test Acc":  [0.7025,0.7000,0.7000,0.9667,0.8500,0.8833],
            "Acc Gap":   [-0.0044,+0.0669,+0.0731,-0.0004,+0.1458,+0.1167],
            "Status":    ["✓ Good","✗ Overfit","✗ Overfit",
                          "✓ Good","✗ Overfit","✗ Overfit"],
        })
        def col_st(v):
            if "✓" in str(v): return "background:#14532d;color:#86efac"
            return "background:#7f1d1d;color:#fca5a5"
        st.dataframe(ov.style.map(col_st, subset=["Status"]),
                     use_container_width=True, hide_index=True)
    footer()

# ================================================================
# PAGE — PSB605IT MODULE
# ================================================================
elif PAGE == "🎓  PSB605IT Module":
    st.markdown("""
    <div class='hero'>
      <div style='position:relative;z-index:1'>
        <div style='font-size:3rem;margin-bottom:12px'>🎓</div>
        <h1 style='background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             font-size:2rem;font-weight:900;margin:0'>PSB605IT</h1>
        <h2 style='color:#a78bfa;font-size:1.1rem;font-weight:600;margin:8px 0'>
          Individual Computing Science Project</h2>
        <p style='color:#475569;font-size:.9rem;margin:8px 0'>
          BSc (Hons) Computing Science · Level 6 · FHEQ Level ·
          Mandatory (M) · 30 Credits · ECTS 15 · PSB Academy
        </p>
        <div style='margin-top:12px;display:flex;flex-wrap:wrap;
             justify-content:center;gap:8px'>
          <span class='pill'>📅 Module Start: 30 MAR 2026</span>
          <span class='pill'>📋 Proposal: 3 MAY 2026</span>
          <span class='pill'>📄 Final Report: 26 JUN 2026</span>
          <span class='pill'>🎤 Viva: 29 JUN – 3 JUL 2026</span>
          <span class='pill'>💻 Slides & Code: 3 JUL 2026</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📋 Module Info", "🎯 ILOs & Assessment",
        "📅 Important Dates", "👨‍🏫 Supervisors",
        "📚 Resources", "💡 Project Guidance"])

    with t1:
        sec("Module Summary", "📋")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='infograph'>", unsafe_allow_html=True)
            for k, v in [
                ("Module Title", "Individual Computing Science Project"),
                ("Module Code", "PSB605IT"),
                ("Level", "6 (FHEQ Level)"),
                ("Applicable Course", "BSc (Hons) Computing Science"),
                ("Mandatory or Optional", "Mandatory (M)"),
                ("Learning Credits", "30"),
                ("ECTS Credits", "15"),
                ("Availability", "On campus only"),
                ("Total Study Hours", "300"),
                ("Number of Weeks", "6"),
                ("School Responsible", "PSB Academy"),
                ("Academic Year", "2025/2026"),
            ]: kv(k, v, "#60a5fa")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='infograph'>
              <div style='color:#a78bfa;font-weight:700;margin-bottom:12px'>
                📝 Module Aims</div>
              <div style='color:#94a3b8;font-size:.82rem;line-height:1.8'>
                This module provides students with the opportunity to explore a
                defined Computing Science issue from both an
                <strong style='color:#e2e8f0'>academic</strong> and a
                <strong style='color:#e2e8f0'>practical</strong> perspective.
                <br><br>
                The project draws upon and further develops:
                <ul style='color:#64748b;margin-top:8px;padding-left:18px'>
                  <li>Subject-related theory and technology</li>
                  <li>Development methods and tools</li>
                  <li>Professional and intellectual skills</li>
                  <li>Research methods and ethical considerations</li>
                  <li>Report writing and critical analysis</li>
                </ul>
              </div>
            </div>
            <div class='infograph' style='margin-top:8px'>
              <div style='color:#a78bfa;font-weight:700;margin-bottom:10px'>
                ⏱️ Teaching Hours Breakdown</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                <div style='background:rgba(30,64,175,.1);border-radius:8px;
                     padding:10px;text-align:center'>
                  <div style='color:#60a5fa;font-size:.7rem;
                       text-transform:uppercase'>Lecture</div>
                  <div style='color:#e2e8f0;font-size:1.3rem;font-weight:800'>60 hrs</div>
                  <div style='color:#475569;font-size:.72rem'>20%</div></div>
                <div style='background:rgba(124,58,237,.1);border-radius:8px;
                     padding:10px;text-align:center'>
                  <div style='color:#a78bfa;font-size:.7rem;
                       text-transform:uppercase'>Lab / Tutorial</div>
                  <div style='color:#e2e8f0;font-size:1.3rem;font-weight:800'>60 hrs</div>
                  <div style='color:#475569;font-size:.72rem'>20%</div></div>
                <div style='background:rgba(6,182,212,.1);border-radius:8px;
                     padding:10px;text-align:center'>
                  <div style='color:#67e8f9;font-size:.7rem;
                       text-transform:uppercase'>Self-Guided</div>
                  <div style='color:#e2e8f0;font-size:1.3rem;font-weight:800'>180 hrs</div>
                  <div style='color:#475569;font-size:.72rem'>60%</div></div>
                <div style='background:rgba(16,185,129,.1);border-radius:8px;
                     padding:10px;text-align:center'>
                  <div style='color:#34d399;font-size:.7rem;
                       text-transform:uppercase'>Total</div>
                  <div style='color:#e2e8f0;font-size:1.3rem;font-weight:800'>300 hrs</div>
                  <div style='color:#475569;font-size:.72rem'>100%</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t2:
        sec("Intended Learning Outcomes (ILOs)", "🎯")
        for code, desc, c, align in [
            ("ILO 1", "Critically review the background related to an area of study.",
             "#3b82f6", "LO1, LO2, LO4"),
            ("ILO 2", "Develop a system or system artefact to enable a defined Cyber Security issue to be investigated via a practical project framework.",
             "#7c3aed", "LO1, LO3, LO4"),
            ("ILO 3", "Deploy appropriate design and development methods, tools and technologies.",
             "#06b6d4", "LO2, LO3"),
            ("ILO 4", "Evaluate project findings and communicate these along with process and professional issues reflections in report form.",
             "#10b981", "LO3, LO4"),
        ]:
            st.markdown(f"""
            <div class='ilo-card' style='border-left:4px solid {c}'>
              <div style='display:flex;justify-content:space-between;
                   align-items:flex-start;gap:12px'>
                <div>
                  <span style='color:{c};font-weight:800;font-size:.95rem'>{code}</span><br>
                  <span style='color:#e2e8f0;font-size:.85rem;line-height:1.5'>{desc}</span>
                </div>
                <span style='background:{c}22;border:1px solid {c}44;
                     border-radius:8px;padding:4px 10px;font-size:.72rem;
                     color:{c};white-space:nowrap;font-weight:600'>{align}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        sec("Assessment Components", "📊")
        for comp, weight, ilos_c, type_, desc, c in [
            ("Component 1", "10 Credits (33.3%)", "ILO 1, 2, 3",
             "Initial Report",
             "Initial report including introduction, background, project plan, design, ethical and technical considerations.\nICSP Proposal — Expected: 30–40 pages.",
             "#3b82f6"),
            ("Component 2", "20 Credits (66.7%)", "ILO 1, 2, 3, 4",
             "Final Report + Viva",
             "Final report including implementation summary, evaluation, critical reflection and concluding remarks.\nICSP Final Report: 40–70 pages.\nViva: 15 min presentation & demo + 5 min Q&A.",
             "#7c3aed"),
        ]:
            st.markdown(f"""
            <div class='glass' style='border-left:4px solid {c}'>
              <div style='display:flex;justify-content:space-between;
                   align-items:flex-start;gap:12px;margin-bottom:10px'>
                <div>
                  <span style='color:{c};font-weight:800;font-size:1rem'>{comp}</span>
                  <span style='color:#94a3b8;font-size:.85rem'> — {type_}</span>
                </div>
                <div style='text-align:right'>
                  <span style='color:#fbbf24;font-weight:700;font-size:.9rem'>{weight}</span><br>
                  <span style='color:#475569;font-size:.75rem'>{ilos_c}</span>
                </div>
              </div>
              <div style='color:#64748b;font-size:.78rem;line-height:1.7;
                   white-space:pre-line'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    with t3:
        sec("Important Dates — Academic Year 2025/2026", "📅")
        for icon, date, title, desc, c in [
            ("🟢", "30 MAR 2026", "Module Start Date",
             "Begin project work and meet your supervisor", "#22c55e"),
            ("🟡", "3 MAY 2026", "Project Proposal Submission",
             "Component 1 — Initial report (30–40 pages expected)", "#f59e0b"),
            ("🔵", "26 JUN 2026", "Final Project Report Submission",
             "Component 2 — Final report (40–70 pages expected)", "#3b82f6"),
            ("🟣", "3 JUL 2026", "Viva Slides & Code Submission",
             "Submit presentation slides and all code", "#7c3aed"),
            ("🔴", "29 JUN – 3 JUL 2026", "Viva Presentation",
             "20 min total: 15 min presentation & demo + 5 min Q&A", "#ef4444"),
        ]:
            st.markdown(f"""
            <div class='timeline-item' style='--tc:{c}'>
              <div style='display:flex;align-items:center;gap:12px'>
                <div style='font-size:1.5rem'>{icon}</div>
                <div>
                  <div style='color:{c};font-weight:800;font-size:.9rem'>{date}</div>
                  <div style='color:#e2e8f0;font-weight:700;font-size:.85rem'>{title}</div>
                  <div style='color:#64748b;font-size:.75rem;margin-top:2px'>{desc}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px'>
          <div class='stat-card' style='--accent:linear-gradient(90deg,#3b82f6,#60a5fa)'>
            <div class='stat-num' style='background:linear-gradient(135deg,#3b82f6,#60a5fa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>30</div>
            <div class='stat-lbl'>Learning Credits</div></div>
          <div class='stat-card' style='--accent:linear-gradient(90deg,#7c3aed,#a78bfa)'>
            <div class='stat-num' style='background:linear-gradient(135deg,#7c3aed,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>15</div>
            <div class='stat-lbl'>ECTS Credits</div></div>
          <div class='stat-card' style='--accent:linear-gradient(90deg,#10b981,#34d399)'>
            <div class='stat-num' style='background:linear-gradient(135deg,#10b981,#34d399);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>300</div>
            <div class='stat-lbl'>Study Hours</div></div>
          <div class='stat-card' style='--accent:linear-gradient(90deg,#f59e0b,#fbbf24)'>
            <div class='stat-num' style='background:linear-gradient(135deg,#f59e0b,#fbbf24);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>6</div>
            <div class='stat-lbl'>Weeks</div></div>
        </div>""", unsafe_allow_html=True)

    with t4:
        sec("Project Supervisors", "👨‍🏫")
        sup1, sup2 = st.columns(2)
        for i, (name, title, c, init) in enumerate([
            ("Dr. Matthew Teow", "PhD · Computing Science", "#3b82f6", "MT"),
            ("Dr. Jason Tan", "PhD · Computing Science", "#7c3aed", "JT"),
            ("Dr. Md Saifullah Bin Razali", "PhD · Computing Science", "#06b6d4", "MR"),
            ("Mr. Suryanto (Ryan)", "MSc · Computing Science", "#10b981", "SR"),
            ("Mr. Raymond Ching Chi Man",
             "MSc · Computing Science · ★ Your Supervisor",
             "#f59e0b", "RC"),
        ]):
            col = sup1 if i % 2 == 0 else sup2
            col.markdown(f"""
            <div class='glass' style='border-left:3px solid {c};margin-bottom:8px'>
              <div style='display:flex;align-items:center;gap:12px'>
                <div style='background:{c}33;border:2px solid {c};
                     border-radius:50%;width:44px;height:44px;
                     display:flex;align-items:center;justify-content:center;
                     font-weight:900;color:{c};font-size:.85rem;flex-shrink:0'>
                  {init}</div>
                <div>
                  <div style='color:#e2e8f0;font-weight:700;font-size:.88rem'>{name}</div>
                  <div style='color:#475569;font-size:.75rem'>{title}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t5:
        sec("Essential Reading", "📚")
        for auth, title, pub, cat in [
            ("Dawson, C. (2015)",
             "Projects in Computing and Information Systems: a Student's Guide.",
             "3rd edn. Harlow: Pearson Education", "Essential"),
            ("Cottrell, S. (2014)",
             "Dissertations and Project Reports: a Step by Step Guide.",
             "Basingstoke: Palgrave Macmillan", "Essential"),
            ("Edgar, T. and Manz, D. (2017)",
             "Research Methods for Cyber Security.",
             "Cambridge, MA: Syngress", "Recommended"),
            ("Ridley, D. (2012)",
             "The Literature Review: a Step-by-Step Guide for Students.",
             "2nd edn. London: Sage", "Recommended"),
            ("Neuman, W. (2014)",
             "Understanding Research.",
             "2nd edn. Harlow: Pearson Education", "Recommended"),
            ("Bott, F. (2014)",
             "Professional Issues in Information Technology.",
             "2nd edn. Swindon: British Computer Society", "Recommended"),
            ("Quinn, M. (2016)",
             "Ethics for the Information Age.",
             "7th edn. Harlow: Pearson Education", "Recommended"),
        ]:
            c = "#3b82f6" if cat == "Essential" else "#475569"
            badge_bg = "#1e40af" if cat == "Essential" else "rgba(71,85,105,.3)"
            st.markdown(f"""
            <div class='glass' style='border-left:3px solid {c};
                 padding:14px;margin-bottom:6px'>
              <div style='display:flex;justify-content:space-between;
                   align-items:flex-start;gap:10px'>
                <div>
                  <b style='color:{c};font-size:.85rem'>{auth}</b><br>
                  <span style='color:#e2e8f0;font-size:.82rem'>{title}</span><br>
                  <span style='color:#475569;font-size:.75rem'>{pub}</span>
                </div>
                <span style='background:{badge_bg};border-radius:6px;
                     padding:3px 10px;font-size:.7rem;color:#e2e8f0;
                     white-space:nowrap;font-weight:600'>{cat}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='infograph' style='border-left:3px solid #f59e0b;margin-top:12px'>
          <div style='color:#fbbf24;font-weight:700;margin-bottom:6px'>
            APA 7th Edition</div>
          <div style='color:#64748b;font-size:.82rem;line-height:1.7'>
            The ICSP reports use only The American Psychological Association (APA)
            Referencing Style, The 7th Edition.<br><br>
            This project references: Hosmer et al. (2013) · Breiman (2001) ·
            Chen & Guestrin (2016) · Lundberg & Lee (2017) ·
            Pedregosa et al. (2011) · Efron & Tibshirani (1993) ·
            WHO (2024) · Benjamin et al. (2019) ·
            Ulianova (2019) · Detrano et al. (1989)
          </div>
        </div>""", unsafe_allow_html=True)

    with t6:
        sec("Project Guidance", "💡")
        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
          <div class='infograph' style='border-left:3px solid #3b82f6'>
            <div style='color:#60a5fa;font-weight:700;margin-bottom:8px'>
              Chapter 1: Introduction</div>
            <div style='color:#64748b;font-size:.78rem;line-height:1.8'>
              ✓ Provisional project title<br>
              ✓ Project background<br>
              ✓ Project aims and objectives<br>
              ✓ Problem statement (3–5 problems)<br>
              ✓ Suggested solutions (Use Case)<br>
              ✓ Project deliverables<br>
              ✓ Project scope<br>
              ✓ Project plan (Gantt Chart)<br>
              ✓ List of resources
            </div>
          </div>
          <div class='infograph' style='border-left:3px solid #7c3aed'>
            <div style='color:#a78bfa;font-weight:700;margin-bottom:8px'>
              Chapter 2 & 3</div>
            <div style='color:#64748b;font-size:.78rem;line-height:1.8'>
              <b style='color:#e2e8f0'>Chapter 2: Literature Review</b><br>
              ✓ Literature Review<br>
              ✓ Technical Literature Review<br><br>
              <b style='color:#e2e8f0'>Chapter 3: Methodology</b><br>
              ✓ Research Methodology<br>
              ✓ System Development Methodology
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        sec("How to Avoid Failing", "⚠️")
        f1, f2 = st.columns(2)
        for i, (fail, fix) in enumerate([
            ("Not reading the project guide", "Read all briefing materials thoroughly"),
            ("Cheating and Plagiarism", "All work must be your own"),
            ("No regular supervisor contact", "Schedule weekly meetings"),
            ("Including misunderstood material", "Only include what you understand"),
            ("Not being honest in the Report", "Never claim your program does what it does not"),
            ("No conclusion", "Always write a proper conclusion section"),
            ("Rushing the Final Report", "Start writing early and revise"),
            ("Thinking project is just a coursework", "Requires independent research"),
        ]):
            col = f1 if i % 2 == 0 else f2
            col.markdown(f"""
            <div style='background:rgba(239,68,68,.08);
                 border:1px solid rgba(239,68,68,.2);border-radius:10px;
                 padding:10px 14px;margin-bottom:6px'>
              <div style='color:#f87171;font-size:.8rem;font-weight:600'>✗ {fail}</div>
              <div style='color:#22c55e;font-size:.75rem;margin-top:4px'>✓ {fix}</div>
            </div>""", unsafe_allow_html=True)

        sec("Project Types Accepted", "💡")
        st.markdown("""
        <div style='display:flex;flex-wrap:wrap;gap:8px'>
          <span class='pill'>💻 Software Application</span>
          <span class='pill'>🌐 Website</span>
          <span class='pill'>📱 Mobile Application</span>
          <span class='pill'>🔐 Security Applications</span>
          <span class='pill'>🗄️ Database Application</span>
          <span class='pill'>🧠 AI Project</span>
          <span class='pill'>🔬 ML / Data Science</span>
          <span class='pill'>🦺 Cyber Security Tool</span>
          <span class='pill'>📡 IoT System</span>
        </div>""", unsafe_allow_html=True)
    footer()

# ================================================================
# PAGE — ABOUT
# ================================================================
elif PAGE == "ℹ️  About":
    sec("About This Project", "ℹ️")
    t1, t2, t3, t4 = st.tabs([
        "👨‍🎓 Project", "📚 References",
        "⚙️ Configuration", "🛠️ Tech Stack"])

    with t1:
        st.markdown("""
        <div class='hero' style='padding:32px'>
          <div style='position:relative;z-index:1'>
            <div style='font-size:3rem;margin-bottom:12px'>🎓</div>
            <h2 style='background:linear-gradient(135deg,#60a5fa,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 font-size:1.8rem;font-weight:900;margin:0'>
              PSB605IT — Individual Computing Science Project</h2>
            <p style='color:#7c3aed;margin:8px 0 0;font-size:.95rem'>
              CVD Risk Prediction Using Machine Learning</p>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        for k, v in [
            ("Project Title", "CVD Risk Prediction Using Machine Learning"),
            ("Module", "PSB605IT — Individual Computing Science Project"),
            ("Course", "BSc (Hons) Computing Science · Level 6 · FHEQ"),
            ("Student Name", "Sudhan Nagarajan"),
            ("Student ID", "050J0DAD"),
            ("Supervisor", "Mr Raymond Ching Chi Man"),
            ("Institution", "PSB Academy"),
            ("Academic Year", "2025 / 2026"),
            ("Dashboard", "Streamlit"),
            ("Champion CARDIO", "Logistic Regression (F1=0.7002)"),
            ("Champion UCI", "Logistic Regression (F1=0.9665)"),
        ]: kv(k, v, "#a78bfa")
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        for auth, full, use in [
            ("Breiman (2001)", "Random Forests. Machine Learning, 45(1), 5–32.", "RF"),
            ("Chen & Guestrin (2016)", "XGBoost: A Scalable Tree Boosting System. KDD '16.", "XGBoost"),
            ("Chawla et al. (2002)", "SMOTE: Synthetic Minority Oversampling. JAIR, 16.", "Imbalance"),
            ("Detrano et al. (1989)", "UCI Heart Disease. AJC, 64(5), 304–310.", "UCI Dataset"),
            ("Efron & Tibshirani (1993)", "Introduction to the Bootstrap. Chapman & Hall.", "Bootstrap"),
            ("Hosmer et al. (2013)", "Applied Logistic Regression, 3rd Ed. Wiley.", "LR"),
            ("Lundberg & Lee (2017)", "SHAP. NeurIPS.", "Explainability"),
            ("Pedregosa et al. (2011)", "scikit-learn. JMLR, 12.", "sklearn"),
            ("Ulianova (2019)", "Cardiovascular Disease Dataset. Kaggle.", "CARDIO"),
            ("WHO (2024)", "CVD Fact Sheet.", "Context"),
            ("Benjamin et al. (2019)", "Heart Disease Statistics. Circulation 139(10).", "Stats"),
        ]:
            st.markdown(f"""
            <div class='glass' style='border-left:3px solid #7c3aed;
                 margin-bottom:8px;padding:14px 18px'>
              <div style='display:flex;justify-content:space-between;gap:12px'>
                <div>
                  <b style='color:#a78bfa;font-size:.9rem'>{auth}</b><br>
                  <span style='color:#475569;font-size:.78rem'>{full}</span>
                </div>
                <span class='pill'>{use}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    with t3:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        for k, v in [
            ("Random Seed", "42 — fully replicable"),
            ("Holdout Split", "80% / 20% · Stratified"),
            ("CV Folds", "10 · Stratified · seed=42"),
            ("Bootstrap", "1 sample · With replacement · OOB≈36.8%"),
            ("Repeat Rounds", "10 × 66/34 · Stratified"),
            ("LOO Folds", "150 subsample"),
            ("RF Trees", "100 · min_samples_split=5"),
            ("LR", "Ridge L2 · C=1.0 · lbfgs · max_iter=1000"),
            ("XGBoost", "scale_pos_weight=auto · verbosity=0"),
            ("Class Weight", "balanced"),
            ("CARDIO Champion", "Logistic Regression (F1=0.7002)"),
            ("UCI Champion", "Logistic Regression (F1=0.9665)"),
        ]: kv(k, v, "#a78bfa")
        st.markdown("</div>", unsafe_allow_html=True)

    with t4:
        tc1, tc2 = st.columns(2)
        for i, (lib, use, c) in enumerate([
            ("Python 3.x", "Core language", "#3b82f6"),
            ("Streamlit", "Dashboard", "#7c3aed"),
            ("scikit-learn 1.3", "LR · RF · CV · Metrics", "#06b6d4"),
            ("XGBoost 1.7", "Gradient Boosting", "#10b981"),
            ("SHAP 0.42", "Explainability", "#f59e0b"),
            ("Pandas 2.x", "Data manipulation", "#ef4444"),
            ("NumPy 1.x", "Numerical computing", "#ec4899"),
            ("Matplotlib / Seaborn", "Visualisation", "#8b5cf6"),
            ("Joblib 1.3", "Model save/load", "#3b82f6"),
        ]):
            col = tc1 if i % 2 == 0 else tc2
            col.markdown(f"""
            <div class='glass' style='border-left:3px solid {c};
                 padding:14px;margin-bottom:6px'>
              <div style='color:{c};font-weight:700;font-size:.9rem'>{lib}</div>
              <div style='color:#475569;font-size:.75rem;margin-top:4px'>{use}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='glass' style='margin-top:12px;text-align:center'>
          <div style='color:#475569;font-size:.75rem;text-transform:uppercase;
               letter-spacing:2px;margin-bottom:10px'>Compatible Platforms</div>
          <div style='display:flex;gap:8px;flex-wrap:wrap;justify-content:center'>
            <span class='pill'>🌐 Chrome</span>
            <span class='pill'>🦊 Firefox</span>
            <span class='pill'>🔵 Edge</span>
            <span class='pill'>🍎 Safari</span>
            <span class='pill'>💻 Windows</span>
            <span class='pill'>🐧 Linux</span>
            <span class='pill'>🍎 macOS</span>
            <span class='pill'>📱 Android</span>
            <span class='pill'>📱 iPhone / iPad</span>
            <span class='pill'>🖥️ Desktop</span>
            <span class='pill'>💼 Laptop</span>
            <span class='pill'>📱 Tablet</span>
          </div>
        </div>""", unsafe_allow_html=True)
    footer()
