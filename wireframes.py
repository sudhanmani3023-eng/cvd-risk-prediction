# ═══════════════════════════════════════════════════════════════════
#  wireframes.py  —  HCD/UCD Wireframe Generator | PSB605IT
#  Sudhan Nagarajan (050J0DAD) | PSB Academy 2025/2026
#  FIX: Emoji → ASCII symbols (no font warnings)
#  Norman (2013) · ISO 9241-210 · Nielsen (1994)
# ═══════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.rcParams['font.family']      = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus']= False

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

os.makedirs("outputs/wireframes", exist_ok=True)

# ── Colour Palette ────────────────────────────────────────────────
C = {
    "bg":      "#0f172a",
    "panel":   "#1e293b",
    "border":  "#334155",
    "purple":  "#7c3aed",
    "blue":    "#3b82f6",
    "green":   "#22c55e",
    "red":     "#ef4444",
    "amber":   "#f59e0b",
    "text":    "#e2e8f0",
    "muted":   "#94a3b8",
    "dim":     "#475569",
    "sidebar": "#111827",
}

# ── ASCII icon replacements (no emoji — no font warnings) ─────────
ICON = {
    "home":     "[HOME]",
    "patient":  "[PT]",
    "play":     "[LAB]",
    "data":     "[DATA]",
    "model":    "[MDL]",
    "sampling": "[SAMP]",
    "shap":     "[SHAP]",
    "pred":     "[PRED]",
    "compare":  "[CMP]",
    "about":    "[INFO]",
    "star":     "(*)",
    "check":    "[OK]",
    "cross":    "[X]",
    "warn":     "[!]",
    "arrow":    "->",
    "pill":     "[o]",
    "lr":       "LR",
    "rf":       "RF",
    "xgb":      "XGB",
    "cardio":   "CARDIO",
    "uci":      "UCI",
    "high":     "HIGH",
    "low":      "LOW",
    "cvd":      "CVD",
    "nocvd":    "NoCVD",
    "dl":       "[DL]",
    "filter":   "[FLT]",
    "chart":    "[CHT]",
    "hcd":      "[HCD]",
    "ucd":      "[UCD]",
    "nielsen":  "[N]",
    "iso":      "[ISO]",
    "norman":   "[NRM]",
}

# ═══════════════════════════════════════════════════════════════════
#  CORE DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════
def setup_fig(w=16, h=10, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    if title:
        ax.text(w/2, h - 0.22, title,
                ha="center", va="top",
                color=C["purple"], fontsize=9,
                fontweight="bold")
    return fig, ax

def box(ax, x, y, w, h,
        fc=None, ec=None, lw=1.2,
        alpha=1.0, radius=0.12):
    fc = fc or C["panel"]
    ec = ec or C["border"]
    p  = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,"
                 f"rounding_size={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha)
    ax.add_patch(p)

def txt(ax, x, y, text,
        size=7.5, color=None,
        ha="center", va="center",
        bold=False, mono=False):
    color = color or C["text"]
    fw    = "bold" if bold else "normal"
    ff    = "monospace" if mono else "DejaVu Sans"
    ax.text(x, y, str(text),
            ha=ha, va=va,
            color=color, fontsize=size,
            fontweight=fw, fontfamily=ff,
            clip_on=True)

def pill(ax, x, y, w, h, text, fc, size=6.5):
    box(ax, x, y, w, h,
        fc=fc, ec=fc, lw=0, radius=0.08)
    txt(ax, x+w/2, y+h/2, text,
        size=size, color="#fff", bold=True)

def hline(ax, x, y, length,
          color=None, lw=0.8, ls="-"):
    color = color or C["border"]
    ax.plot([x, x+length], [y, y],
            color=color, lw=lw, ls=ls,
            solid_capstyle="round")

def vline(ax, x, y, length,
          color=None, lw=0.8, ls="-"):
    color = color or C["border"]
    ax.plot([x, x], [y, y+length],
            color=color, lw=lw, ls=ls)

def arrow(ax, x1, y1, x2, y2,
          color=None, lw=1.2):
    color = color or C["purple"]
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=color, lw=lw,
            connectionstyle="arc3,rad=0.0"))

def hcd_tag(ax, x, y, text,
            clr=None, w_hint=None):
    """HCD annotation tag — ASCII only"""
    clr    = clr or C["purple"]
    w      = w_hint or (len(text)*0.072 + 0.2)
    h      = 0.22
    box(ax, x, y, w, h,
        fc=clr+"22", ec=clr,
        lw=0.7, radius=0.04)
    txt(ax, x + w/2, y + h/2,
        text, size=5.5, color=clr)

def section_bar(ax, x, y, w, h, text, clr):
    box(ax, x, y, w, h,
        fc=clr+"33", ec=clr, lw=1.5)
    txt(ax, x + w/2, y + h/2,
        text, size=8.5, bold=True, color=clr)


# ═══════════════════════════════════════════════════════════════════
#  WF-00  SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
def wf00_architecture():
    fig, ax = setup_fig(20, 14,
        "WF-00 | System Architecture | HCD/UCD | "
        "PSB605IT | Sudhan Nagarajan (050J0DAD)")

    # ── Title ─────────────────────────────────────────────────────
    box(ax, 0.3, 12.8, 19.4, 1.0,
        fc=C["purple"]+"33", ec=C["purple"], lw=2)
    txt(ax, 10, 13.35,
        "CVD Risk Prediction — Full System Architecture",
        size=13, bold=True, color=C["purple"])
    txt(ax, 10, 12.98,
        "PSB605IT | Sudhan Nagarajan (050J0DAD) | "
        "HCD (Norman 2013) | ISO 9241-210",
        size=7.5, color=C["muted"])

    layers = [
        # (label, y, height, colour, items)
        ("LAYER 1: DATA SOURCES",
         11.3, 1.2, C["blue"],
         [("CARDIO\n(Ulianova 2019)",
           "68,723 pts | 12 features"),
          ("UCI\n(Detrano 1989)",
           "297 pts | 14 features"),
          ("models/ *.pkl",
           "6 trained models saved"),
          ("outputs/ *.csv/*.png",
           "Results, charts, predictions")]),
        ("LAYER 2: main.py — ML PIPELINE",
         9.4, 1.65, C["amber"],
         [("(1) EDA\n& Clean",   "Ulianova\nDetrano"),
          ("(2) Holdout\n80/20", "seed=42\nStratified"),
          ("(3) CV\n10-fold",    "Stratified\nseed=42"),
          ("(4) Bootstrap\nOOB", "1 sample\n36.8% OOB"),
          ("(5) Repeat\n10x",    "66% train\nStratified"),
          ("(6) LOO\n150 folds", "Full\ndataset"),
          ("(7) Train\nvs Test", "Overfit\ndetect"),
          ("LR | RF\n| XGBoost", "3 models\nseed=42"),
          ("SHAP\nExplain",      "Lundberg\n2017"),
          ("Save\nOutputs",      "PNG+CSV\nmodels/")]),
    ]

    for lbl, ly, lh, lclr, items in layers:
        box(ax, 0.3, ly, 19.4, lh,
            fc=lclr+"11", ec=lclr, lw=1.2)
        txt(ax, 1.4, ly+lh-0.22, lbl,
            size=7.5, bold=True,
            color=lclr, ha="left")
        n   = len(items)
        iw  = 19.0 / n
        for i, item in enumerate(items):
            xi = 0.4 + i*iw
            title = item[0] if isinstance(item, tuple) else item
            sub   = item[1] if isinstance(item, tuple) and len(item)>1 else ""
            box(ax, xi+0.05, ly+0.12,
                iw-0.15, lh-0.38,
                fc=lclr+"22", ec=lclr, lw=0.9)
            txt(ax, xi+iw/2, ly+lh/2+0.08,
                title, size=7, bold=True, color=lclr)
            txt(ax, xi+iw/2, ly+0.27,
                sub, size=5.8, color=C["muted"])
            if i < n-1:
                arrow(ax, xi+iw-0.12, ly+lh/2,
                      xi+iw-0.0, ly+lh/2,
                      color=lclr, lw=0.9)

    # Layer 3: Dashboard pages
    box(ax, 0.3, 4.8, 19.4, 4.4,
        fc=C["purple"]+"11", ec=C["purple"], lw=1.2)
    txt(ax, 1.4, 9.0,
        "LAYER 3: dashboard.py — STREAMLIT GUI (HCD/UCD)",
        size=7.5, bold=True, color=C["purple"], ha="left")

    pages = [
        ("[HOME] Home",               "Overview\nNavigate",    C["blue"]),
        ("[PT] Patient\nPredictor",   "Vitals in\nEnsemble",   C["green"]),
        ("[LAB] Classifier\nPlayground","Single mdl\nThreshold",C["amber"]),
        ("[DATA] Dataset\nExplorer",  "EDA\nHeatmap",          C["blue"]),
        ("[MDL] Model\nPerformance",  "ROC CM\nMetrics",       C["purple"]),
        ("[SAMP] Sampling\nMethods",  "7 methods\nCharts",     C["green"]),
        ("[SHAP] SHAP\nExplain",      "Feature\nimport.",      C["amber"]),
        ("[PRED] Predictions\nViewer","Filter\nDownload",      C["blue"]),
        ("[CMP] Model\nCompare",      "Radar\nOverfit",        C["purple"]),
        ("[INFO] About\n& Refs",      "HCD/UCD\nRefs",         C["muted"]),
    ]
    for i, (pg, desc, clr) in enumerate(pages):
        xi = 0.4 + (i % 5)*3.9
        yi = 6.8 if i < 5 else 5.0
        box(ax, xi, yi, 3.65, 1.6,
            fc=clr+"22", ec=clr, lw=1.2)
        txt(ax, xi+1.83, yi+1.05,
            pg, size=7, bold=True, color=clr)
        txt(ax, xi+1.83, yi+0.38,
            desc, size=6.5, color=C["muted"])

    # Layer 4: HCD Principles
    box(ax, 0.3, 2.8, 19.4, 1.8,
        fc=C["green"]+"11", ec=C["green"], lw=1.2)
    txt(ax, 1.4, 4.4,
        "LAYER 4: HCD/UCD PRINCIPLES — Nielsen (1994) "
        "· Norman (2013) · ISO 9241-210:2019 · WCAG 2.1 AA",
        size=7.5, bold=True, color=C["green"], ha="left")
    principles = [
        "#1 Visibility", "#2 Real World",
        "#3 User Control", "#4 Consistency",
        "#5 Error Prev", "#6 Recognition",
        "#7 Flexibility", "#8 Minimalist",
        "#9 Recovery",   "#10 Help & Docs",
    ]
    pcolors = [C["blue"],C["green"],C["amber"],C["purple"],C["red"],
               C["blue"],C["green"],C["amber"],C["purple"],C["muted"]]
    for i, (p, c) in enumerate(zip(principles, pcolors)):
        xi = 0.5 + i*1.93
        box(ax, xi, 3.0, 1.78, 0.65,
            fc=c+"22", ec=c, lw=0.9, radius=0.07)
        txt(ax, xi+0.89, 3.32,
            p, size=6, bold=True, color=c)

    # References block
    box(ax, 0.3, 0.25, 19.4, 2.35,
        fc=C["panel"], ec=C["border"], lw=0.8)
    txt(ax, 0.7, 2.42,
        "REFERENCES",
        size=7, bold=True, color=C["muted"], ha="left")
    refs = [
        "Breiman (2001) Random Forests",
        "Chen & Guestrin (2016) XGBoost",
        "Detrano et al. (1989) UCI Dataset",
        "Efron & Tibshirani (1993) Bootstrap",
        "Hosmer et al. (2013) Logistic Reg.",
        "ISO 9241-210:2019 HCD Standard",
        "Lundberg & Lee (2017) SHAP",
        "Nielsen (1994) 10 Usability Heuristics",
        "Norman (2013) Design Everyday Things",
        "Pedregosa et al. (2011) scikit-learn",
        "Ulianova (2019) CARDIO Dataset",
        "WCAG 2.1 Accessibility Guidelines",
        "WHO (2024) CVD Fact Sheet",
    ]
    for i, r in enumerate(refs):
        xi = 0.7 + (i % 5)*3.9
        yi = 2.1 - (i // 5)*0.55
        txt(ax, xi, yi, f"  {r}",
            size=6, color=C["dim"], ha="left")

    plt.tight_layout()
    _save(fig, "wf00_system_architecture")


# ═══════════════════════════════════════════════════════════════════
#  WF-01  HOME PAGE
# ═══════════════════════════════════════════════════════════════════
def wf01_home():
    fig, ax = setup_fig(18, 12,
        "WF-01 | Home Page | HCD: First Impression & Navigation | "
        "Nielsen #1 #6 | PSB605IT")

    # Sidebar
    box(ax, 0.2, 0.3, 2.8, 11.4,
        fc=C["sidebar"], ec=C["purple"], lw=1.5)
    txt(ax, 1.6, 11.25,
        "CVD Risk AI", size=9,
        bold=True, color=C["purple"])
    txt(ax, 1.6, 10.9,
        "PSB605IT · PSB Academy",
        size=6, color=C["dim"])
    hline(ax, 0.3, 10.72, 2.6,
          color=C["purple"]+"44")
    hcd_tag(ax, 0.3, 10.52,
            "[N]#6 Recognition not Recall",
            C["purple"], 2.6)

    nav = [
        ("[HOME] Home",                True),
        ("[PT] Patient Predictor",     False),
        ("[LAB] Classifier Playground",False),
        ("[DATA] Dataset Explorer",    False),
        ("[MDL] Model Performance",    False),
        ("[SAMP] Sampling Methods",    False),
        ("[SHAP] SHAP Explainability", False),
        ("[PRED] Predictions Viewer",  False),
        ("[CMP] Model Comparison",     False),
        ("[INFO] About & References",  False),
    ]
    for i, (item, active) in enumerate(nav):
        yi = 10.18 - i*0.73
        if active:
            box(ax, 0.3, yi-0.19, 2.6, 0.4,
                fc=C["purple"]+"33",
                ec=C["purple"], lw=1)
        txt(ax, 0.5, yi, item, size=7,
            color=C["purple"] if active else C["muted"],
            ha="left", bold=active)

    hline(ax, 0.3, 2.8, 2.6, color=C["border"])
    txt(ax, 1.6, 2.55,
        "Active Dataset",
        size=6.5, bold=True, color=C["muted"])
    box(ax, 0.4, 1.9, 2.4, 0.5,
        fc=C["panel"], ec=C["border"])
    txt(ax, 1.6, 2.15,
        "CARDIO — Ulianova (2019) v",
        size=6.5, color=C["text"])
    txt(ax, 1.6, 1.6,
        "Model Status",
        size=6, bold=True, color=C["dim"])
    for j, (m, ok) in enumerate([
        ("(o) LR  cardio", True),
        ("(o) RF  cardio", True),
        ("(o) XGB cardio", True),
        ("(o) LR  uci",    True),
        ("(o) RF  uci",    True),
        ("(o) XGB uci",    True),
    ]):
        txt(ax, 0.5, 1.38-j*0.17, m,
            size=6,
            color=C["green"] if ok else C["red"],
            ha="left")
    hcd_tag(ax, 0.3, 0.38,
            "[N]#1 Visibility of Status",
            C["blue"], 2.6)

    # Hero banner
    box(ax, 3.2, 9.6, 14.6, 2.1,
        fc=C["purple"]+"22",
        ec=C["purple"], lw=1.5)
    txt(ax, 10.5, 11.3,
        "CVD Risk Prediction Using Machine Learning",
        size=13, bold=True, color=C["text"])
    txt(ax, 10.5, 10.85,
        "PSB605IT — Individual Computing Science Project",
        size=8, color=C["purple"], bold=True)
    txt(ax, 10.5, 10.45,
        "HCD (Norman, 2013)  |  ISO 9241-210  |  "
        "Nielsen (1994)  |  WCAG 2.1",
        size=7, color=C["muted"])
    txt(ax, 10.5, 9.82,
        "[3 ML Models]  [2 Datasets]  "
        "[7 Sampling Methods]  [CVD Prediction]",
        size=8, color=C["blue"])
    hcd_tag(ax, 3.2, 9.45,
            "[NRM] Value prop clear in 5s (Norman 2013)",
            C["purple"], 9.5)

    # Stat cards
    for i, (v, l, c) in enumerate([
        ("70,297", "Total Patients",   C["blue"]),
        ("3",      "ML Models",        C["purple"]),
        ("7",      "Sampling Methods", C["green"]),
        ("8",      "Eval Metrics",     C["amber"]),
        ("2",      "Datasets",         C["red"]),
    ]):
        xi = 3.2 + i*2.95
        box(ax, xi, 8.05, 2.75, 1.2,
            fc=c+"22", ec=c, lw=1.2)
        txt(ax, xi+1.38, 8.82, v,
            size=14, bold=True, color=c)
        txt(ax, xi+1.38, 8.22, l,
            size=6.5, color=C["muted"])
    hcd_tag(ax, 3.2, 7.9,
            "[N]#1 Key stats immediately visible",
            C["blue"], 8.0)

    # Results tables
    for i, (title, rows, border) in enumerate([
        ("CARDIO — Champion Results",
         [("(*) LR", "0.7489","0.7002", True),
          ("   RF",  "0.7776","0.6985", False),
          ("   XGB", "0.7754","0.6996", False)],
         C["green"]),
        ("UCI — Champion Results",
         [("(*) LR", "0.9989","0.9665", True),
          ("   RF",  "0.9408","0.8490", False),
          ("   XGB", "0.9609","0.8825", False)],
         C["blue"]),
    ]):
        xi = 3.2 + i*7.3
        box(ax, xi, 5.55, 7.0, 2.2,
            fc=C["panel"], ec=border, lw=1.2)
        txt(ax, xi+3.5, 7.55, title,
            size=8, bold=True, color=border)
        for j, (m, auc, f1, champ) in enumerate(rows):
            yi = 7.15 - j*0.52
            if champ:
                box(ax, xi+0.15, yi-0.2,
                    6.7, 0.42,
                    fc=C["amber"]+"22",
                    ec=C["amber"], lw=0.8)
            txt(ax, xi+0.4, yi, m, size=7.5,
                bold=champ,
                color=C["amber"] if champ else C["muted"],
                ha="left")
            txt(ax, xi+4.2, yi, auc,
                size=7.5, color=C["text"])
            txt(ax, xi+6.0, yi, f1,
                size=7.5, color=C["text"])
        txt(ax, xi+0.4, 7.3, "Model",
            size=6.5, color=C["dim"], ha="left")
        txt(ax, xi+4.2, 7.3, "AUC",
            size=6.5, color=C["dim"])
        txt(ax, xi+6.0, 7.3, "F1",
            size=6.5, color=C["dim"])

    # Navigation cards
    txt(ax, 10.5, 5.35,
        "Navigation Guide",
        size=9, bold=True, color=C["text"])
    nav_cards = [
        ("[PT] Patient Predictor","Vitals->Risk",     C["green"]),
        ("[LAB] Playground",      "Model->Classify",  C["blue"]),
        ("[DATA] Dataset",        "EDA->Insights",    C["blue"]),
        ("[MDL] Performance",     "ROC->Metrics",     C["purple"]),
        ("[SAMP] Sampling",       "7 methods",        C["green"]),
        ("[SHAP] Explainability", "Feature importance",C["amber"]),
        ("[PRED] Predictions",    "Filter->Download", C["blue"]),
        ("[CMP] Comparison",      "Radar->Overfit",   C["purple"]),
    ]
    for i, (pg, desc, clr) in enumerate(nav_cards):
        xi = 3.2 + (i % 4)*3.65
        yi = 4.6 - (i // 4)*1.0
        box(ax, xi, yi, 3.45, 0.85,
            fc=C["panel"], ec=clr, lw=1)
        txt(ax, xi+1.73, yi+0.58, pg,
            size=7, bold=True, color=clr)
        txt(ax, xi+1.73, yi+0.22, desc,
            size=6.5, color=C["muted"])
    hcd_tag(ax, 3.2, 2.55,
            "[N]#6 Recognition not Recall — all options visible",
            C["purple"], 9.5)

    # HCD principles strip
    box(ax, 3.2, 0.3, 14.6, 2.0,
        fc=C["panel"], ec=C["border"])
    txt(ax, 10.5, 2.12,
        "HCD / UCD Principles Applied",
        size=8, bold=True, color=C["purple"])
    hcd_p = [
        ("#1 Visibility",   C["blue"]),
        ("#2 Real World",   C["green"]),
        ("#3 Control",      C["amber"]),
        ("#4 Consistency",  C["purple"]),
        ("#5 Error Prev",   C["red"]),
        ("#6 Recognition",  C["blue"]),
        ("#7 Flexibility",  C["green"]),
        ("#8 Minimalist",   C["amber"]),
        ("#9 Recovery",     C["purple"]),
        ("#10 Help",        C["muted"]),
    ]
    for i, (p, c) in enumerate(hcd_p):
        xi = 3.4 + i*1.46
        box(ax, xi, 0.42, 1.35, 0.48,
            fc=c+"22", ec=c, lw=0.8, radius=0.06)
        txt(ax, xi+0.68, 0.66, p,
            size=5.8, bold=True, color=c)

    plt.tight_layout()
    _save(fig, "wf01_home")


# ═══════════════════════════════════════════════════════════════════
#  WF-02  PATIENT RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════
def wf02_patient_predictor():
    fig, ax = setup_fig(20, 14,
        "WF-02 | Patient Risk Predictor | "
        "HCD: Task Analysis · Clinical Workflow | "
        "Nielsen #2 #5 | PSB605IT")

    # Page header
    box(ax, 0.3, 12.5, 19.4, 1.25,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 13.38,
        "[PT] Patient Health Metric Input",
        size=12, bold=True,
        color=C["text"], ha="left")
    txt(ax, 0.7, 12.82,
        "Enter patient vitals — all 3 ML models predict CVD risk "
        "with ensemble voting and probability breakdown",
        size=7.5, color=C["muted"], ha="left")
    hcd_tag(ax, 0.3, 12.38,
            "[NRM] Task Analysis — form maps clinical intake "
            "workflow (Norman 2013)",
            C["purple"], 11.0)

    # Controls row
    box(ax, 0.3, 11.5, 13.0, 0.78,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 11.98,
        "Select Feature Set:",
        size=7.5, bold=True,
        color=C["muted"], ha="left")
    pill(ax, 2.2, 11.62, 4.2, 0.5,
         "(*) CARDIO (Ulianova 2019)", C["green"], 7)
    box(ax, 6.6, 11.62, 3.5, 0.5,
        fc=C["panel"], ec=C["border"])
    txt(ax, 8.35, 11.87,
        "UCI Heart Disease",
        size=7, color=C["muted"])

    box(ax, 13.5, 11.5, 6.2, 0.78,
        fc=C["panel"], ec=C["border"])
    txt(ax, 13.8, 11.98,
        "Decision Threshold",
        size=7.5, bold=True,
        color=C["muted"], ha="left")
    box(ax, 16.2, 11.62, 3.3, 0.48,
        fc=C["border"], ec=C["purple"], lw=1.5)
    txt(ax, 17.85, 11.86,
        "---[O]---  50%",
        size=7.5, color=C["purple"])
    hcd_tag(ax, 13.5, 11.35,
            "[N]#7 Flexibility — threshold adjustable by expert users",
            C["amber"], 6.0)

    hline(ax, 0.3, 11.3, 19.4, color=C["border"])

    # Form sections
    form_sections = [
        ("DEMOGRAPHICS", 9.9,
         [("Age (years)","50","yrs"),
          ("Gender","Male",""),
          ("Height (cm)","170","cm"),
          ("Weight (kg)","75","kg")]),
        ("BLOOD PRESSURE", 8.25,
         [("Systolic BP","---O--- 120","mmHg"),
          ("Diastolic BP","---O--- 80","mmHg"),
          ("","",""),
          ("","","")]),
        ("BLOOD TESTS", 6.6,
         [("Cholesterol","1 - Normal",""),
          ("Blood Glucose","1 - Normal",""),
          ("","",""),
          ("","","")]),
        ("LIFESTYLE", 4.95,
         [("Smoking","Non-Smoker [OK]",""),
          ("Alcohol","No [OK]",""),
          ("Physical Activity","Active [OK]",""),
          ("","","")]),
    ]
    for sect, ys, fields in form_sections:
        box(ax, 0.3, ys-0.05, 19.4, 1.7,
            fc=C["panel"]+"88", ec=C["border"], lw=0.8)
        txt(ax, 0.65, ys+1.45, sect,
            size=8, bold=True,
            color=C["purple"], ha="left")
        for j, (fn, fv, fu) in enumerate(fields):
            if not fn:
                continue
            xi = 0.5 + j*4.88
            box(ax, xi, ys+0.1, 4.6, 1.1,
                fc=C["bg"], ec=C["border"], lw=0.8)
            txt(ax, xi+0.18, ys+1.0, fn,
                size=6.5, color=C["dim"], ha="left")
            txt(ax, xi+2.3, ys+0.55, fv,
                size=8, bold=True, color=C["text"])
            if fu:
                txt(ax, xi+4.3, ys+1.0, fu,
                    size=6, color=C["dim"], ha="right")

    # BMI feedback bar
    box(ax, 0.3, 8.28, 19.4, 0.58,
        fc=C["amber"]+"22", ec=C["amber"], lw=1)
    txt(ax, 0.7, 8.57,
        "Calculated BMI: 25.9 kg/m2  ->  [!] Overweight  |  "
        "Normal: 18.5-24.9  Overweight: 25-29.9  Obese: >=30",
        size=7.5, color=C["amber"], ha="left")
    hcd_tag(ax, 14.0, 8.28,
            "[NRM] Immediate feedback (Norman 2013)",
            C["green"], 5.5)

    hcd_tag(ax, 0.3, 4.78,
            "[N]#2 Real World Match — clinical terminology users know",
            C["blue"], 9.5)
    hcd_tag(ax, 0.3, 4.48,
            "[N]#5 Error Prevention — slider ranges enforce valid input",
            C["red"], 8.0)

    # Submit button
    box(ax, 0.3, 3.15, 19.4, 1.2,
        fc=C["purple"]+"44", ec=C["purple"], lw=2.5)
    txt(ax, 10, 3.77,
        "PREDICT CVD RISK  —  ALL 3 MODELS",
        size=12, bold=True, color=C["text"])
    hcd_tag(ax, 0.3, 2.95,
            "[NRM] Single clear CTA — Affordance (Norman 2013)",
            C["purple"], 7.5)

    # Results banner
    box(ax, 0.3, 0.25, 19.4, 2.6,
        fc=C["red"]+"22", ec=C["red"], lw=2)
    txt(ax, 10, 2.55,
        "[!]  HIGH CVD RISK DETECTED",
        size=14, bold=True, color=C["red"])
    txt(ax, 10, 2.08,
        "Ensemble CVD Probability: 73.2%  |  "
        "Vote: 3/3 Models -> CVD  |  Threshold: 50%",
        size=9, color=C["text"])
    for i, (m, p, clr) in enumerate([
        ("Logistic Regression", "P(CVD)=0.7451", C["red"]),
        ("Random Forest",       "P(CVD)=0.6924", C["red"]),
        ("XGBoost",             "P(CVD)=0.7613", C["red"]),
    ]):
        xi = 0.5 + i*6.5
        box(ax, xi, 0.35, 6.1, 1.35,
            fc=C["panel"], ec=clr, lw=1.5)
        txt(ax, xi+3.05, 1.38, m,
            size=7.5, bold=True, color=clr)
        txt(ax, xi+3.05, 0.97,
            "[!] CVD", size=11,
            bold=True, color=clr)
        txt(ax, xi+3.05, 0.58, p,
            size=7.5, color=C["text"])
    hcd_tag(ax, 0.3, 0.1,
            "[HCD] Ensemble voting — 3 models reduce single-model error",
            C["amber"], 9.0)

    plt.tight_layout()
    _save(fig, "wf02_patient_predictor")


# ═══════════════════════════════════════════════════════════════════
#  WF-03  CLASSIFIER PLAYGROUND
# ═══════════════════════════════════════════════════════════════════
def wf03_classifier_playground():
    fig, ax = setup_fig(20, 13,
        "WF-03 | Classifier Playground | "
        "HCD: Flexibility & Expert Control | "
        "Nielsen #3 #7 | PSB605IT")

    # Header
    box(ax, 0.3, 11.5, 19.4, 1.3,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 12.45,
        "[LAB] Classifier Playground",
        size=12, bold=True, color=C["text"], ha="left")
    txt(ax, 0.7, 11.78,
        "Choose classifier  |  Choose dataset  |  Adjust threshold  "
        "|  Get instant probability breakdown",
        size=7.5, color=C["muted"], ha="left")
    hcd_tag(ax, 0.3, 11.35,
            "[N]#7 Flexibility — experts tune; novices use defaults",
            C["amber"], 8.5)

    # 4 Config controls
    ctrls = [
        ("Dataset",     "CARDIO — Ulianova (2019) v", C["blue"]),
        ("Classifier",  "XGBoost v",                  C["purple"]),
        ("Threshold",   "---[O]---  50%",             C["amber"]),
        ("Pie Mode",    "Both Classes v",             C["green"]),
    ]
    for i, (lbl_t, val, clr) in enumerate(ctrls):
        xi = 0.3 + i*4.88
        box(ax, xi, 10.3, 4.6, 1.0,
            fc=C["panel"], ec=clr, lw=1.2)
        txt(ax, xi+0.18, 11.12, lbl_t,
            size=7.5, bold=True, color=clr, ha="left")
        txt(ax, xi+2.3, 10.74, val,
            size=7.5, color=C["text"])

    # Model info cards
    for i, (v, lbl_t, clr) in enumerate([
        ("XGBoost",  "Classifier",   C["purple"]),
        ("0.7754",   "AUC CARDIO",   C["blue"]),
        ("0.6996",   "F1  CARDIO",   C["green"]),
    ]):
        xi = 0.3 + i*6.55
        box(ax, xi, 9.0, 6.25, 1.1,
            fc=C["panel"], ec=clr+"88", lw=1)
        txt(ax, xi+3.1, 9.75, v,
            size=12, bold=True, color=clr)
        txt(ax, xi+3.1, 9.2, lbl_t,
            size=7, color=C["muted"])

    txt(ax, 0.7, 8.8,
        "About XGBoost: Gradient boosted trees. "
        "Strong AUC. Some overfitting on small UCI dataset. "
        "Ref: Chen & Guestrin (2016)",
        size=7, color=C["dim"], ha="left")
    hline(ax, 0.3, 8.62, 19.4)

    # Input form
    box(ax, 0.3, 5.15, 19.4, 3.3,
        fc=C["panel"]+"55", ec=C["border"], lw=0.8)
    txt(ax, 0.7, 8.25,
        "[PT] Patient Input",
        size=9, bold=True, color=C["text"], ha="left")
    row_data = [
        ("Demographics:",
         ["Age: 55", "Gender: Male",
          "Height: 172cm", "Weight: 80kg"]),
        ("Blood Pressure:",
         ["Systolic: 135 mmHg", "Diastolic: 85",
          "", "-> Stage 1"]),
        ("Blood Tests:",
         ["Chol: 2-Above", "Glucose: 1-Normal",
          "", ""]),
        ("Lifestyle:",
         ["Smoking: No [OK]", "Alcohol: No [OK]",
          "Active: [OK]", ""]),
    ]
    for ri, (rl, fields) in enumerate(row_data):
        yi = 7.72 - ri*0.72
        txt(ax, 0.7, yi, rl,
            size=7.5, bold=True,
            color=C["purple"], ha="left")
        for fi, fv in enumerate(fields):
            if not fv:
                continue
            xi = 1.6 + fi*4.5
            box(ax, xi, yi-0.55, 4.2, 0.48,
                fc=C["bg"], ec=C["border"], lw=0.7)
            txt(ax, xi+2.1, yi-0.31, fv,
                size=7, color=C["text"])

    # Submit
    box(ax, 0.3, 3.8, 19.4, 1.1,
        fc=C["purple"]+"44", ec=C["purple"], lw=2)
    txt(ax, 10, 4.38,
        "CLASSIFY WITH XGBOOST",
        size=12, bold=True, color=C["text"])

    # Result panels
    box(ax, 0.3, 0.3, 9.5, 3.3,
        fc=C["panel"], ec=C["green"], lw=1.5)
    txt(ax, 5.05, 3.4,
        "Probability Distribution (Pie)",
        size=8, bold=True, color=C["text"])
    # Simulated pie
    circle = plt.Circle((5.05, 1.8), 1.3,
                         fc=C["green"]+"33",
                         ec=C["green"], lw=1.5)
    ax.add_patch(circle)
    wedge = patches.Wedge(
        (5.05, 1.8), 1.3, 90, 90+360*0.28,
        fc=C["red"]+"88", ec=C["red"], lw=1.5)
    ax.add_patch(wedge)
    txt(ax, 4.2, 1.8, "No CVD\n72%",
        size=7, color=C["green"], bold=True)
    txt(ax, 5.9, 1.8, "CVD\n28%",
        size=7, color=C["red"], bold=True)

    box(ax, 10.1, 0.3, 9.6, 3.3,
        fc=C["panel"], ec=C["border"])
    txt(ax, 14.9, 3.4,
        "Probability Breakdown",
        size=8, bold=True, color=C["text"])
    bd = [
        ("Classifier",  "XGBoost",   C["purple"]),
        ("Dataset",     "CARDIO",    C["blue"]),
        ("P(CVD)",      "0.27842130",C["red"]),
        ("P(No CVD)",   "0.72157870",C["green"]),
        ("Threshold",   "50%",       C["amber"]),
        ("Decision",    "[OK] NO CVD",C["green"]),
        ("Confidence",  "72.16%",    C["muted"]),
        ("AUC (test)",  "0.7754",    C["purple"]),
    ]
    for bi, (k, v, c) in enumerate(bd):
        yi = 2.95 - bi*0.32
        txt(ax, 10.4, yi, k, size=6.5,
            color=C["dim"], ha="left")
        txt(ax, 19.5, yi, v, size=6.5,
            bold=True, color=c, ha="right")
        hline(ax, 10.2, yi-0.1, 9.2,
              color=C["border"], lw=0.5)

    hcd_tag(ax, 0.3, 0.12,
            "[N]#3 User Control — model, dataset, "
            "threshold all user-chosen",
            C["blue"], 9.5)

    plt.tight_layout()
    _save(fig, "wf03_classifier_playground")


# ═══════════════════════════════════════════════════════════════════
#  WF-04  MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
def wf04_model_performance():
    fig, ax = setup_fig(18, 11,
        "WF-04 | Model Performance | "
        "HCD: Visibility of All Metrics | "
        "Nielsen #1 #4 | PSB605IT")

    box(ax, 0.3, 9.8, 17.4, 1.0,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 10.5,
        "[MDL] Model Performance — CARDIO",
        size=11, bold=True, color=C["text"], ha="left")

    # Tabs
    for i, (t, active) in enumerate([
        ("Metrics Heatmap",     True),
        ("ROC Curves",          False),
        ("Confusion Matrices",  False),
        ("Full Metrics Table",  False),
    ]):
        xi = 0.3 + i*4.38
        box(ax, xi, 9.0, 4.22, 0.68,
            fc=C["purple"] if active else C["panel"],
            ec=C["purple"] if active else C["border"],
            lw=1.5 if active else 0.8)
        txt(ax, xi+2.11, 9.34, t,
            size=7.5, bold=active,
            color=C["text"] if active else C["muted"])
    hcd_tag(ax, 0.3, 8.82,
            "[N]#4 Consistency — same tab layout on all pages",
            C["purple"], 7.5)

    # Heatmap
    box(ax, 0.3, 1.5, 11.5, 7.2,
        fc=C["panel"], ec=C["border"])
    txt(ax, 6.05, 8.55,
        "Evaluation Metrics Heatmap — All 3 Models",
        size=8.5, bold=True, color=C["text"])

    metrics = ["AUC","CA","F1","Prec","Recall","MCC","Spec","LogLoss"]
    models  = ["LR","RF","XGB"]
    data    = [
        [0.7489,0.7025,0.7002,0.7052,0.7008,0.4060,0.7707,0.6061],
        [0.7776,0.7000,0.6985,0.7012,0.6987,0.3999,0.7512,0.5700],
        [0.7754,0.7000,0.6996,0.6998,0.6996,0.3994,0.7171,0.5695],
    ]
    cw, ch = 1.28, 0.72
    for ci, m in enumerate(metrics):
        txt(ax, 2.2+ci*cw, 7.9, m,
            size=6.5, bold=True, color=C["muted"])
    for ri, r in enumerate(models):
        txt(ax, 1.0, 7.4-ri*ch, r,
            size=8, bold=True, color=C["purple"])
        for ci, v in enumerate(data[ri]):
            best = (v == max(data[r_][ci]
                             for r_ in range(3)))
            worst= (v == min(data[r_][ci]
                             for r_ in range(3)))
            fc_v = (C["green"]+"44" if best and ci < 7
                    else C["red"]+"44" if worst and ci == 7
                    else C["panel"])
            box(ax, 1.58+ci*cw, 7.08-ri*ch,
                cw-0.07, ch-0.06,
                fc=fc_v, ec=C["border"], lw=0.5)
            txt(ax, 1.58+ci*cw+cw/2-0.04,
                7.42-ri*ch,
                f"{v:.4f}", size=6.5, color=C["text"])

    txt(ax, 0.6, 2.2,
        "Green = best per metric  |  Red = worst",
        size=6.5, color=C["dim"], ha="left")
    hcd_tag(ax, 0.3, 1.95,
            "[N]#1 Visibility — all metrics visible at once",
            C["blue"], 6.5)
    hcd_tag(ax, 0.3, 1.65,
            "[N]#4 Consistency — colour coding same throughout",
            C["purple"], 6.0)

    # ROC preview
    box(ax, 12.1, 5.5, 5.6, 3.2,
        fc=C["panel"], ec=C["border"])
    txt(ax, 14.9, 8.5,
        "ROC Curves Preview",
        size=7.5, bold=True, color=C["text"])
    x_r = np.linspace(0, 1, 50)
    for vals, clr, lbl_t in [
        (x_r**0.4,  C["blue"],  "LR 0.7489"),
        (x_r**0.35, C["green"], "RF 0.7776"),
        (x_r**0.38, C["purple"],"XGB 0.7754"),
    ]:
        ax.plot(x_r*5.2+12.2, vals*2.8+5.6,
                color=clr, lw=1.5, label=lbl_t)
    ax.plot([12.2,17.4],[5.6,8.4],
            color=C["dim"], lw=0.8, ls="--")
    txt(ax, 14.9, 5.75,
        "LR  |  RF  |  XGB  |  Random",
        size=6.5, color=C["muted"])

    # Confusion matrix preview
    box(ax, 12.1, 1.5, 5.6, 3.8,
        fc=C["panel"], ec=C["border"])
    txt(ax, 14.9, 5.1,
        "Confusion Matrix Preview",
        size=7.5, bold=True, color=C["text"])
    cm_txt = [["TN=158","FP=47"],["FN=72","TP=123"]]
    cm_fc  = [[C["green"]+"33",C["red"]+"33"],
               [C["red"]+"33", C["green"]+"33"]]
    for ri in range(2):
        for ci in range(2):
            box(ax, 12.6+ci*2.5, 4.4-ri*1.3,
                2.35, 1.1,
                fc=cm_fc[ri][ci], ec=C["border"], lw=0.8)
            txt(ax, 13.78+ci*2.5, 4.95-ri*1.3,
                cm_txt[ri][ci],
                size=8, bold=True, color=C["text"])

    # Download button
    box(ax, 12.1, 0.3, 5.6, 1.0,
        fc=C["blue"]+"33", ec=C["blue"], lw=1.2)
    txt(ax, 14.9, 0.82,
        "[DL] Download Metrics CSV",
        size=8, bold=True, color=C["blue"])
    hcd_tag(ax, 0.3, 1.32,
            "[N]#3 User Control — export for own analysis",
            C["green"], 5.0)

    plt.tight_layout()
    _save(fig, "wf04_model_performance")


# ═══════════════════════════════════════════════════════════════════
#  WF-05  SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
def wf05_shap():
    fig, ax = setup_fig(18, 11,
        "WF-05 | SHAP Explainability | "
        "HCD: Transparency & Trust | "
        "Lundberg & Lee (2017) | PSB605IT")

    box(ax, 0.3, 9.5, 17.4, 1.3,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 10.48,
        "[SHAP] SHAP Explainability — Champion Model",
        size=11, bold=True, color=C["text"], ha="left")
    txt(ax, 0.7, 9.88,
        "SHapley Additive exPlanations (Lundberg & Lee, 2017) — "
        "How much each feature contributed to the prediction",
        size=7.5, color=C["muted"], ha="left")

    box(ax, 0.3, 7.8, 17.4, 1.5,
        fc=C["purple"]+"11", ec=C["purple"], lw=1.2)
    txt(ax, 9.0, 9.1,
        "Positive SHAP -> pushes toward CVD (1)   |   "
        "Negative SHAP -> pushes toward No CVD (0)",
        size=8, color=C["text"])
    txt(ax, 9.0, 8.62,
        "HCD: Explainability builds user TRUST in AI decisions "
        "(Norman 2013 — Transparency) | ISO 9241-210",
        size=7, color=C["purple"])
    txt(ax, 9.0, 8.2,
        "Clinicians can validate model logic against medical knowledge",
        size=7, color=C["muted"])
    hcd_tag(ax, 0.3, 7.65,
            "[NRM] Transparency principle (Norman 2013)",
            C["purple"], 5.5)

    # SHAP bar chart
    box(ax, 0.3, 1.5, 10.5, 6.1,
        fc=C["panel"], ec=C["border"])
    txt(ax, 5.55, 7.4,
        "SHAP Summary — CARDIO (Logistic Regression Champion)",
        size=8.5, bold=True, color=C["text"])

    feats = [
        ("ap_hi",      0.85, C["red"]),
        ("age",        0.72, C["red"]),
        ("ap_lo",      0.61, C["amber"]),
        ("cholesterol",0.48, C["amber"]),
        ("weight",     0.35, C["blue"]),
        ("gluc",       0.28, C["blue"]),
        ("active",    -0.22, C["green"]),
        ("smoke",      0.18, C["amber"]),
        ("alco",       0.08, C["blue"]),
        ("gender",     0.05, C["muted"]),
        ("height",     0.03, C["muted"]),
    ]
    for i, (feat, val, clr) in enumerate(feats):
        yi   = 6.5 - i*0.44
        bw   = abs(val)*5.5
        bx   = 3.0 if val >= 0 else 3.0-bw
        box(ax, bx, yi-0.14, bw, 0.27,
            fc=clr+"77", ec=clr, lw=0.8, radius=0.04)
        ax.scatter(3.0+val*5.5, yi,
                   c=clr, s=20, alpha=0.85, zorder=5)
        txt(ax, 2.8, yi, feat,
            size=7, bold=True, color=clr, ha="right")
        txt(ax, 3.0+val*5.5 + (0.28 if val>=0 else -0.28),
            yi, f"{val:+.2f}",
            size=6, color=clr,
            ha="left" if val>=0 else "right")

    vline(ax, 3.0, 1.6, 5.6,
          color=C["dim"], lw=1.0, ls="--")
    txt(ax, 3.0, 1.65, "0",
        size=6.5, color=C["dim"])

    # Feature guide
    box(ax, 11.1, 1.5, 6.6, 6.1,
        fc=C["panel"], ec=C["border"])
    txt(ax, 14.4, 7.4,
        "Feature Importance Guide",
        size=8.5, bold=True, color=C["text"])
    guide = [
        ("ap_hi",      "Systolic BP",
         "Strongest predictor", "Very High",C["red"]),
        ("age",        "Patient Age",
         "Non-modifiable risk", "High",     C["amber"]),
        ("ap_lo",      "Diastolic BP",
         "Hypertension risk",   "High",     C["amber"]),
        ("cholesterol","Cholesterol",
         "Atherosclerosis link","Med-High", C["purple"]),
        ("weight",     "Body Weight",
         "Obesity link",        "Medium",   C["blue"]),
        ("gluc",       "Blood Glucose",
         "Diabetic CVD",        "Medium",   C["blue"]),
        ("active",     "Physical Act",
         "PROTECTIVE [OK]",     "Protect",  C["green"]),
        ("smoke",      "Smoking",
         "Modifiable risk [X]", "Risk",     C["red"]),
    ]
    for i, (feat, name, desc, impact, clr) in enumerate(guide):
        yi = 6.9 - i*0.6
        box(ax, 11.2, yi-0.22, 6.3, 0.5,
            fc=clr+"11", ec=clr+"44",
            lw=0.8, radius=0.06)
        txt(ax, 11.38, yi, feat,
            size=7, bold=True, color=clr, ha="left")
        txt(ax, 12.8, yi, name,
            size=6.5, color=C["muted"], ha="left")
        txt(ax, 17.5, yi, impact,
            size=6, color=clr, ha="right")
        txt(ax, 12.8, yi-0.17, desc,
            size=5.5, color=C["dim"], ha="left")

    hcd_tag(ax, 0.3, 1.3,
            "[NRM] Transparency builds trust | "
            "Lundberg & Lee (2017) NeurIPS",
            C["purple"], 8.5)

    plt.tight_layout()
    _save(fig, "wf05_shap")


# ═══════════════════════════════════════════════════════════════════
#  WF-06  HCD DESIGN PROCESS (ISO 9241-210 cycle)
# ═══════════════════════════════════════════════════════════════════
def wf06_hcd_process():
    fig, ax = setup_fig(20, 13,
        "WF-06 | HCD Design Process | ISO 9241-210:2019 | "
        "Norman (2013) | Nielsen (1994) | PSB605IT")

    box(ax, 0.3, 11.5, 19.4, 1.3,
        fc=C["purple"]+"33", ec=C["purple"], lw=2)
    txt(ax, 10, 12.35,
        "Human-Centered Design Process — CVD Risk Prediction",
        size=13, bold=True, color=C["purple"])
    txt(ax, 10, 11.78,
        "ISO 9241-210:2019  |  Norman (2013)  |  "
        "Nielsen 10 Usability Heuristics (1994)  |  "
        "Sudhan Nagarajan (050J0DAD)",
        size=8, color=C["muted"])

    # ISO 9241-210 iterative cycle
    cx, cy, radius = 10.0, 7.5, 3.0
    steps = [
        (90,  "Plan HCD\nProcess",
         "Define scope\nSeed=42\nChoose datasets",    C["blue"]),
        (0,   "Understand\nContext",
         "CVD patients\nClinicians\nML researchers",  C["green"]),
        (270, "Specify User\nRequirements",
         "Risk scores\nClear output\nExplainability", C["amber"]),
        (180, "Produce\nDesign Solutions",
         "Dashboard\nForms Charts\nSHAP Alerts",      C["purple"]),
    ]
    for angle_deg, title, desc, clr in steps:
        ang = np.deg2rad(angle_deg)
        xc  = cx + radius*np.cos(ang)
        yc  = cy + radius*np.sin(ang)
        box(ax, xc-2.1, yc-0.95, 4.2, 1.9,
            fc=clr+"22", ec=clr, lw=2, radius=0.2)
        txt(ax, xc, yc+0.45, title,
            size=9, bold=True, color=clr)
        txt(ax, xc, yc-0.18, desc,
            size=7, color=C["muted"])

    # Cycle arrows
    for a1, a2 in [(90,0),(0,270),(270,180),(180,90)]:
        r1 = np.deg2rad(a1)
        r2 = np.deg2rad(a2)
        x1 = cx + radius*np.cos(r1)
        y1 = cy + radius*np.sin(r1)
        x2 = cx + radius*np.cos(r2)
        y2 = cy + radius*np.sin(r2)
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=C["purple"], lw=1.5,
                connectionstyle="arc3,rad=0.35"))

    # Centre: Evaluate
    box(ax, 8.2, 6.7, 3.6, 1.6,
        fc=C["red"]+"22", ec=C["red"],
        lw=2, radius=0.15)
    txt(ax, 10.0, 7.52,
        "Evaluate Against\nRequirements",
        size=8.5, bold=True, color=C["red"])

    # Nielsen heuristics
    box(ax, 0.3, 0.3, 9.0, 5.8,
        fc=C["panel"], ec=C["border"])
    txt(ax, 4.8, 5.9,
        "Nielsen 10 Usability Heuristics (1994)",
        size=8.5, bold=True, color=C["text"])
    heuristics = [
        ("#1  Visibility of System Status",
         "Model status  |  Spinners  |  Risk banners",
         C["blue"]),
        ("#2  Match System & Real World",
         "Clinical terms  |  BP categories  |  BMI calc",
         C["green"]),
        ("#3  User Control & Freedom",
         "Threshold slider  |  Dataset selector  |  Filters",
         C["amber"]),
        ("#4  Consistency & Standards",
         "Red=danger throughout  |  Same card layout",
         C["purple"]),
        ("#5  Error Prevention",
         "Input limits  |  Disclaimers  |  Ranges shown",
         C["red"]),
        ("#6  Recognition over Recall",
         "Icons + labels  |  Always-visible sidebar",
         C["blue"]),
        ("#7  Flexibility & Efficiency",
         "Defaults for novices  |  Sliders for experts",
         C["green"]),
        ("#8  Aesthetic & Minimalist Design",
         "Dark theme  |  Card layout  |  No clutter",
         C["amber"]),
        ("#9  Help Recover from Errors",
         "Alert boxes with fix guidance  |  Danger alerts",
         C["purple"]),
        ("#10 Help & Documentation",
         "Contextual tooltips  |  About page  |  Ref labels",
         C["muted"]),
    ]
    for i, (h, ex, clr) in enumerate(heuristics):
        yi = 5.38 - i*0.49
        box(ax, 0.45, yi-0.23, 8.7, 0.45,
            fc=clr+"11", ec=clr+"44",
            lw=0.7, radius=0.05)
        txt(ax, 0.65, yi,
            h, size=6.5, bold=True,
            color=clr, ha="left")
        txt(ax, 0.65, yi-0.17,
            ex, size=5.5, color=C["dim"], ha="left")

    # Norman principles
    box(ax, 9.8, 0.3, 9.9, 5.8,
        fc=C["panel"], ec=C["border"])
    txt(ax, 14.75, 5.9,
        "Norman (2013) Design Principles",
        size=8.5, bold=True, color=C["text"])
    norman = [
        ("Visibility",
         "All model status shown in sidebar",    C["blue"]),
        ("Feedback",
         "Immediate risk banner on submit",      C["green"]),
        ("Affordance",
         "Buttons look clickable — gradient",    C["amber"]),
        ("Mapping",
         "Red=danger | Green=safe | real world", C["red"]),
        ("Constraints",
         "Slider min/max prevents invalid input",C["purple"]),
        ("Consistency",
         "Same colour system every page",        C["blue"]),
        ("Conceptual Model",
         "Dashboard mirrors clinical workflow",  C["green"]),
        ("Simplicity",
         "Progressive disclosure — detail later",C["amber"]),
        ("Error Recovery",
         "Alert boxes with actionable solutions",C["purple"]),
        ("Transparency",
         "SHAP explains every AI decision",      C["muted"]),
    ]
    for i, (p, ex, clr) in enumerate(norman):
        yi = 5.38 - i*0.49
        box(ax, 9.95, yi-0.23, 9.5, 0.45,
            fc=clr+"11", ec=clr+"44",
            lw=0.7, radius=0.05)
        txt(ax, 10.15, yi, p,
            size=7, bold=True, color=clr, ha="left")
        txt(ax, 10.15, yi-0.17, ex,
            size=5.5, color=C["dim"], ha="left")

    plt.tight_layout()
    _save(fig, "wf06_hcd_process")


# ═══════════════════════════════════════════════════════════════════
#  WF-07  PREDICTIONS VIEWER
# ═══════════════════════════════════════════════════════════════════
def wf07_predictions_viewer():
    fig, ax = setup_fig(18, 11,
        "WF-07 | Predictions Viewer | "
        "HCD: User Control & Data Exploration | "
        "Nielsen #3 #7 | PSB605IT")

    box(ax, 0.3, 9.5, 17.4, 1.3,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 10.48,
        "[PRED] Predictions Viewer — CARDIO",
        size=11, bold=True, color=C["text"], ha="left")
    txt(ax, 0.7, 9.88,
        "Browse, filter and download all model predictions  |  "
        "HCD: User control (Nielsen #3)",
        size=7.5, color=C["muted"], ha="left")

    # Model selector
    box(ax, 0.3, 8.6, 17.4, 0.72,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 9.05,
        "Select Model:",
        size=7.5, bold=True, color=C["muted"], ha="left")
    for i, (m, active, clr) in enumerate([
        ("LR  Logistic Regression", True,  C["purple"]),
        ("RF  Random Forest",        False, C["muted"]),
        ("XGB XGBoost",              False, C["muted"]),
    ]):
        xi = 2.5 + i*5.2
        box(ax, xi, 8.7, 4.8, 0.52,
            fc=clr+"33" if active else C["panel"],
            ec=clr if active else C["border"],
            lw=1.5 if active else 0.8)
        txt(ax, xi+2.4, 8.96, m,
            size=7.5, bold=active,
            color=clr if active else C["muted"])

    # Stat cards
    for i, (v, l, c) in enumerate([
        ("400",   "Total Instances", C["blue"]),
        ("281",   "[OK] Correct",    C["green"]),
        ("119",   "[X] Misclassified",C["red"]),
        ("70.25%","Accuracy",        C["purple"]),
    ]):
        xi = 0.3 + i*4.38
        box(ax, xi, 7.2, 4.22, 1.2,
            fc=C["panel"], ec=c+"88", lw=1.2)
        txt(ax, xi+2.11, 8.0, v,
            size=14, bold=True, color=c)
        txt(ax, xi+2.11, 7.38, l,
            size=6.5, color=C["muted"])
    hcd_tag(ax, 0.3, 7.05,
            "[N]#1 Visibility — summary statistics shown at top",
            C["blue"], 8.5)

    # Filter controls
    box(ax, 0.3, 6.1, 17.4, 0.88,
        fc=C["panel"], ec=C["border"])
    for i, (lbl_t, val, w) in enumerate([
        ("Result Filter",  "All v",      4.2),
        ("Actual Class",   "All v",      4.2),
        ("Rows to Show",   "---O--- 100",4.2),
        ("Sort By",        "Default v",  4.6),
    ]):
        xi = 0.4 + i*4.4
        txt(ax, xi, 6.76, lbl_t,
            size=6.5, bold=True,
            color=C["muted"], ha="left")
        box(ax, xi, 6.22, w, 0.46,
            fc=C["bg"], ec=C["border"])
        txt(ax, xi+w/2, 6.45, val,
            size=7, color=C["text"])
    hcd_tag(ax, 0.3, 5.95,
            "[N]#3 User Control — filter, sort, rows configurable",
            C["amber"], 8.5)

    # Data table
    box(ax, 0.3, 1.5, 17.4, 4.3,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 5.6,
        "Showing 100 of 400 filtered rows",
        size=7.5, bold=True, color=C["text"], ha="left")
    cols   = ["Inst","Actual","Predicted","P(NoCVD)","P(CVD)","Sum","Result"]
    col_xs = [0.55,1.35,3.1,5.3,7.3,9.3,10.9]
    for ci, (col, cx) in enumerate(zip(cols, col_xs)):
        txt(ax, cx, 5.2, col, size=6.5,
            bold=True, color=C["purple"], ha="left")
    hline(ax, 0.4, 5.0, 17.2, color=C["purple"]+"66")

    rows_d = [
        (0,"CVD (1)","CVD (1)","0.0768","0.9232","1.0","[OK]",True),
        (1,"CVD (1)","No CVD","0.5716","0.4284","1.0","[X]", False),
        (2,"No CVD","No CVD","0.6902","0.3098","1.0","[OK]",True),
        (3,"CVD (1)","No CVD","0.5633","0.4367","1.0","[X]", False),
        (4,"CVD (1)","No CVD","0.5260","0.4740","1.0","[X]", False),
        (5,"No CVD","No CVD","0.5008","0.4992","1.0","[OK]",True),
        (6,"No CVD","CVD (1)","0.2635","0.7365","1.0","[X]", False),
        (7,"No CVD","No CVD","0.5692","0.4308","1.0","[OK]",True),
        (8,"No CVD","No CVD","0.7319","0.2681","1.0","[OK]",True),
    ]
    for ri, (inst,actual,pred,pno,pcvd,sp,res,correct) in \
            enumerate(rows_d):
        yi = 4.7 - ri*0.36
        rfc = C["green"]+"11" if correct else C["red"]+"11"
        box(ax, 0.4, yi-0.17, 17.2, 0.33,
            fc=rfc, ec="none", lw=0)
        vals = [str(inst),actual,pred,pno,pcvd,sp,res]
        vclrs = [C["text"],
                 C["green"] if "NoCVD" in actual or "No CVD" in actual
                 else C["red"],
                 C["green"] if "NoCVD" in pred or "No CVD" in pred
                 else C["red"],
                 C["text"],
                 C["red"] if float(pcvd)>0.5 else C["green"],
                 C["dim"],
                 C["green"] if correct else C["red"]]
        for ci, (v, cx, vc) in enumerate(
                zip(vals, col_xs, vclrs)):
            txt(ax, cx, yi, v, size=6.5,
                color=vc, ha="left",
                bold=(ci==6))

    txt(ax, 9.0, 1.72,
        "... 91 more rows ...",
        size=7, color=C["dim"])

    # Download buttons
    box(ax, 0.3, 0.3, 8.6, 1.0,
        fc=C["blue"]+"33", ec=C["blue"], lw=1.2)
    txt(ax, 4.75, 0.82,
        "[DL] Download All Predictions",
        size=8, bold=True, color=C["blue"])
    box(ax, 9.1, 0.3, 8.6, 1.0,
        fc=C["green"]+"33", ec=C["green"], lw=1.2)
    txt(ax, 13.4, 0.82,
        "[DL] Download Filtered Results",
        size=8, bold=True, color=C["green"])
    hcd_tag(ax, 0.3, 0.12,
            "[N]#7 Flexibility — download full or filtered data",
            C["purple"], 9.0)

    plt.tight_layout()
    _save(fig, "wf07_predictions_viewer")


# ═══════════════════════════════════════════════════════════════════
#  WF-08  MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════
def wf08_model_comparison():
    fig, ax = setup_fig(20, 12,
        "WF-08 | Model Comparison | "
        "HCD: Consistent Visual Comparison | "
        "Nielsen #4 #8 | PSB605IT")

    box(ax, 0.3, 10.5, 19.4, 1.2,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 11.35,
        "[CMP] Model Comparison — Both Datasets",
        size=12, bold=True, color=C["text"], ha="left")

    # Tabs
    for i, (t, active) in enumerate([
        ("Side-by-Side",        True),
        ("Radar Chart",         False),
        ("Overfitting Analysis",False),
    ]):
        xi = 0.3 + i*6.55
        box(ax, xi, 9.7, 6.2, 0.65,
            fc=C["purple"] if active else C["panel"],
            ec=C["purple"] if active else C["border"],
            lw=1.5 if active else 0.8)
        txt(ax, xi+3.1, 10.03, t,
            size=8, bold=active,
            color=C["text"] if active else C["muted"])
    hcd_tag(ax, 0.3, 9.55,
            "[N]#4 Consistency — same layout both datasets",
            C["purple"], 8.0)

    # Comparison tables
    for i, (title, rows, border) in enumerate([
        ("CARDIO — 68,723 patients",
         [("(*) LR",  "0.7489","0.7002","0.4060",True),
          ("   RF",   "0.7776","0.6985","0.3999",False),
          ("   XGB",  "0.7754","0.6996","0.3994",False)],
         C["green"]),
        ("UCI — 297 patients",
         [("(*) LR",  "0.9989","0.9665","0.9330",True),
          ("   RF",   "0.9408","0.8490","0.6984",False),
          ("   XGB",  "0.9609","0.8825","0.7655",False)],
         C["blue"]),
    ]):
        xi = 0.3 + i*10.0
        box(ax, xi, 7.2, 9.5, 2.3,
            fc=C["panel"], ec=border, lw=1.2)
        txt(ax, xi+4.75, 9.28, title,
            size=8, bold=True, color=border)
        for j,(m,auc,f1,mcc,champ) in enumerate(rows):
            yi = 8.88 - j*0.52
            if champ:
                box(ax,xi+0.12,yi-0.21,9.25,0.42,
                    fc=C["amber"]+"22",
                    ec=C["amber"],lw=0.8)
            txt(ax,xi+0.35,yi,m,size=7.5,bold=champ,
                color=C["amber"] if champ else C["muted"],
                ha="left")
            txt(ax,xi+4.0,yi,auc,size=7.5,color=C["text"])
            txt(ax,xi+6.0,yi,f1,size=7.5,color=C["text"])
            txt(ax,xi+8.0,yi,mcc,size=7.5,color=C["text"])

    # Bar chart
    box(ax, 0.3, 0.3, 9.5, 6.7,
        fc=C["panel"], ec=C["border"])
    txt(ax, 5.05, 6.8,
        "CARDIO — Metric Bar Chart",
        size=8, bold=True, color=C["text"])
    metric_names = ["AUC","CA","F1","MCC"]
    lr_v  = [0.7489,0.7025,0.7002,0.4060]
    rf_v  = [0.7776,0.7000,0.6985,0.3999]
    xgb_v = [0.7754,0.7000,0.6996,0.3994]
    bw = 0.55
    for mi,(m,lv,rv,xv) in enumerate(
            zip(metric_names,lr_v,rf_v,xgb_v)):
        xb = 1.5 + mi*2.0
        for j,(v,clr) in enumerate([
            (lv,C["blue"]),
            (rv,C["green"]),
            (xv,C["purple"])]):
            bxj = xb + j*0.65
            h   = v*5.5
            box(ax, bxj, 0.55, bw, h,
                fc=clr+"88", ec=clr, lw=0.8)
            txt(ax, bxj+bw/2, h+0.75,
                f"{v:.3f}",
                size=5.5, color=clr)
        txt(ax, xb+0.95, 0.38, m,
            size=7, bold=True, color=C["muted"])
    txt(ax, 5.0, 6.55,
        "Blue=LR  Green=RF  Purple=XGB",
        size=6.5, color=C["muted"])

    # Radar chart
    box(ax, 10.2, 0.3, 9.5, 6.7,
        fc=C["panel"], ec=C["border"])
    txt(ax, 14.95, 6.8,
        "Radar Chart — CARDIO",
        size=8, bold=True, color=C["text"])
    cx, cy, r = 14.95, 3.8, 2.5
    n = 5
    cats = ["AUC","CA","F1","MCC","Spec"]
    for i in range(n):
        angle = np.pi/2 - 2*np.pi*i/n
        xp = cx + r*np.cos(angle)
        yp = cy + r*np.sin(angle)
        ax.plot([cx,xp],[cy,yp],
                color=C["border"], lw=0.7)
        txt(ax, xp+0.2*np.cos(angle),
            yp+0.15*np.sin(angle),
            cats[i], size=6.5, color=C["muted"])
    for clr, vals in [
        (C["blue"],  [0.749,0.703,0.700,0.406,0.771]),
        (C["green"], [0.778,0.700,0.699,0.400,0.751]),
        (C["purple"],[0.775,0.700,0.700,0.399,0.717]),
    ]:
        pts = []
        for i,v in enumerate(vals):
            angle = np.pi/2 - 2*np.pi*i/n
            pts.append((cx+v*r*np.cos(angle),
                         cy+v*r*np.sin(angle)))
        pts.append(pts[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=clr, lw=1.5)
        ax.fill(xs, ys, color=clr, alpha=0.12)
    txt(ax, 14.95, 0.65,
        "Blue=LR  Green=RF  Purple=XGB",
        size=6.5, color=C["muted"])

    hcd_tag(ax, 0.3, 0.12,
            "[N]#4 Consistency — same colours for same models everywhere",
            C["blue"], 9.5)
    hcd_tag(ax, 10.2, 0.12,
            "[N]#8 Minimalist — only key comparisons shown",
            C["amber"], 9.2)

    plt.tight_layout()
    _save(fig, "wf08_model_comparison")


# ═══════════════════════════════════════════════════════════════════
#  WF-09  USER JOURNEY MAP
# ═══════════════════════════════════════════════════════════════════
def wf09_user_journey():
    fig, ax = setup_fig(22, 14,
        "WF-09 | UCD User Journey Map | "
        "3 Personas | ISO 9241-210 | PSB605IT")

    box(ax, 0.3, 12.5, 21.4, 1.2,
        fc=C["purple"]+"33", ec=C["purple"], lw=2)
    txt(ax, 11, 13.22,
        "User Journey Map — CVD Risk Prediction System",
        size=13, bold=True, color=C["purple"])
    txt(ax, 11, 12.72,
        "User-Centered Design (UCD)  |  ISO 9241-210:2019  |  "
        "3 Personas  |  Task Analysis  |  "
        "Sudhan Nagarajan (050J0DAD)",
        size=8, color=C["muted"])

    # Personas
    personas = [
        ("Dr Sarah Chen",
         "Cardiologist",
         "Assess patient CVD risk\nCompare model predictions\n"
         "Understand feature importance\nExport results for records",
         C["green"],
         "Expert | Medical knowledge\nNeeds: accuracy, trust, SHAP"),
        ("Raj Kumar",
         "ML Researcher",
         "Evaluate 3 models on 2 datasets\nCompare sampling strategies\n"
         "Analyse overfitting metrics\nDownload CSV results",
         C["blue"],
         "Expert | ML knowledge\nNeeds: metrics, charts, data"),
        ("Mr Ahmad",
         "Patient (60 yrs)",
         "Enter personal health data\nUnderstand risk level simply\n"
         "Get actionable health advice\nShare report with doctor",
         C["amber"],
         "Novice | No ML knowledge\nNeeds: simple, clear, actionable"),
    ]
    for i, (name, role, goals, clr, needs) in enumerate(personas):
        xi = 0.4 + i*7.18
        box(ax, xi, 9.8, 6.8, 2.5,
            fc=clr+"22", ec=clr, lw=1.5)
        txt(ax, xi+3.4, 12.08, name,
            size=9.5, bold=True, color=clr)
        txt(ax, xi+3.4, 11.65,
            f"Persona: {role}",
            size=7, color=C["muted"])
        txt(ax, xi+0.25, 11.35, goals,
            size=7, color=C["text"],
            ha="left", va="top")
        box(ax, xi, 9.8, 6.8, 0.72,
            fc=clr+"33", ec=clr, lw=0.8)
        txt(ax, xi+3.4, 10.15, needs,
            size=6, color=clr)

    # Task flows
    txt(ax, 11, 9.55,
        "USER TASK FLOWS",
        size=10, bold=True, color=C["text"])

    flows = [
        ("Dr Sarah (Cardiologist)", C["green"],
         ["1. Open Dashboard",
          "2. Select CARDIO dataset",
          "3. Go to Patient Predictor",
          "4. Enter patient vitals",
          "5. View ensemble risk banner",
          "6. Check SHAP feature importance",
          "7. Compare 3 model probabilities",
          "8. Download patient summary"]),
        ("Raj Kumar (ML Researcher)", C["blue"],
         ["1. Open Dashboard",
          "2. Switch between datasets",
          "3. Check Model Performance",
          "4. View ROC curves and CM",
          "5. Examine sampling comparison",
          "6. Analyse overfitting gaps",
          "7. Open Predictions Viewer",
          "8. Download metrics CSV"]),
        ("Mr Ahmad (Patient)", C["amber"],
         ["1. Open Dashboard -> Home",
          "2. Read navigation guide",
          "3. Select Patient Predictor",
          "4. Fill simple health form",
          "5. Read large risk banner",
          "6. Follow clinical advice",
          "7. Download patient summary",
          "8. Share PDF with doctor"]),
    ]
    for fi, (name, clr, steps) in enumerate(flows):
        xi = 0.4 + fi*7.18
        box(ax, xi, 1.5, 6.8, 7.8,
            fc=C["panel"], ec=clr, lw=1.2)
        txt(ax, xi+3.4, 9.12, name,
            size=8, bold=True, color=clr)
        for si, step in enumerate(steps):
            yi = 8.7 - si*0.87
            sc = (C["green"] if "SHAP" in step or "risk" in step.lower()
                  else C["amber"] if "Download" in step else clr)
            box(ax, xi+0.15, yi-0.38,
                6.45, 0.72,
                fc=sc+"22", ec=sc+"55",
                lw=0.7, radius=0.07)
            txt(ax, xi+0.38, yi, step,
                size=7, color=C["text"], ha="left")
            if si < len(steps)-1:
                arrow(ax, xi+3.4, yi-0.38,
                      xi+3.4, yi-0.41,
                      color=clr, lw=0.8)
        hcd_tag(ax, xi, 1.55,
                "[N]#2 Real World Match",
                clr, 3.5)

    # Footer
    box(ax, 0.3, 0.3, 21.4, 1.0,
        fc=C["panel"], ec=C["border"])
    txt(ax, 11, 1.05,
        "UCD Core Principles (ISO 9241-210): User needs drive design  |  "
        "Iterative evaluation  |  Multi-perspective testing  |  "
        "WCAG 2.1 AA  |  Contextual help at point of need",
        size=7.5, color=C["muted"])
    txt(ax, 11, 0.55,
        "Norman (2013)  |  ISO 9241-210  |  Nielsen (1994)  |  "
        "Hosmer et al. (2013)  |  Breiman (2001)  |  "
        "Chen & Guestrin (2016)  |  Lundberg & Lee (2017)",
        size=6.5, color=C["dim"])

    plt.tight_layout()
    _save(fig, "wf09_user_journey")


# ═══════════════════════════════════════════════════════════════════
#  WF-10  RISK RESULT DETAIL
# ═══════════════════════════════════════════════════════════════════
def wf10_risk_result():
    fig, ax = setup_fig(18, 12,
        "WF-10 | Risk Prediction Result | "
        "HCD: Feedback Colour Coding Clinical Guidance | "
        "Norman (2013) | PSB605IT")

    txt(ax, 9, 11.7,
        "HCD Result Page — Immediate Feedback "
        "(Norman 2013 Feedback Principle)",
        size=9, bold=True, color=C["purple"])

    # HIGH risk banner
    box(ax, 0.3, 9.0, 17.4, 2.5,
        fc=C["red"]+"22", ec=C["red"],
        lw=2.5, radius=0.2)
    txt(ax, 9, 11.07, "[!]",
        size=22, color=C["red"])
    txt(ax, 9, 10.48,
        "HIGH CVD RISK DETECTED",
        size=14, bold=True, color=C["red"])
    txt(ax, 9, 9.98,
        "Ensemble CVD Probability:  73.2%",
        size=10, color=C["text"])
    txt(ax, 9, 9.55,
        "3/3 models predict CVD  |  Threshold: 50%",
        size=8.5, color=C["muted"])
    txt(ax, 9, 9.15,
        "[Vote: 3/3 CVD]   [Threshold: 50%]   [Risk: HIGH]",
        size=7.5, color=C["text"])
    hcd_tag(ax, 0.3, 8.85,
            "[NRM] Red = danger (universal) | "
            "Norman 2013 Mapping principle",
            C["red"], 9.5)

    # Gauge
    box(ax, 0.3, 7.8, 17.4, 0.9,
        fc=C["panel"], ec=C["border"])
    txt(ax, 0.7, 8.52,
        "CVD Probability Gauge",
        size=8, bold=True, color=C["text"], ha="left")
    box(ax, 0.5, 7.85, 5.8, 0.56,
        fc=C["green"]+"44", ec=C["green"],
        lw=0.8, radius=0.05)
    txt(ax, 3.4, 8.13, "LOW RISK (0-35%)",
        size=6.5, color=C["green"])
    box(ax, 6.4, 7.85, 4.1, 0.56,
        fc=C["amber"]+"44", ec=C["amber"],
        lw=0.8, radius=0.05)
    txt(ax, 8.45, 8.13, "MODERATE (35-60%)",
        size=6.5, color=C["amber"])
    box(ax, 10.6, 7.85, 6.6, 0.56,
        fc=C["red"]+"66", ec=C["red"],
        lw=1.5, radius=0.05)
    txt(ax, 13.9, 8.13, "HIGH RISK  73.2% [O]",
        size=7, bold=True, color=C["red"])

    # Model cards
    txt(ax, 9, 7.6,
        "Individual Model Predictions",
        size=8.5, bold=True, color=C["text"])
    for i, (m, p, clr) in enumerate([
        ("Logistic Regression","P(CVD)=0.7451",C["red"]),
        ("Random Forest",      "P(CVD)=0.6924",C["red"]),
        ("XGBoost",            "P(CVD)=0.7613",C["red"]),
    ]):
        xi = 0.4 + i*5.88
        box(ax, xi, 5.62, 5.55, 1.8,
            fc=C["panel"], ec=clr, lw=1.8)
        txt(ax, xi+2.78, 7.19, m,
            size=7, color=C["muted"])
        txt(ax, xi+2.78, 6.75,
            "[!] CVD", size=11,
            bold=True, color=clr)
        txt(ax, xi+2.78, 6.35, p,
            size=8, bold=True, color=C["text"])
        box(ax, xi+0.2, 5.73, 5.15, 0.3,
            fc=clr+"77", ec=clr,
            lw=0.5, radius=0.04)
        txt(ax, xi+2.78, 5.86,
            "XXXXXXXXXX 74%",
            size=7, color="#fff")
    hcd_tag(ax, 0.3, 5.46,
            "[HCD] Ensemble voting reduces single-model error",
            C["amber"], 9.0)

    # Bar chart + Clinical advice
    box(ax, 0.3, 2.5, 10.5, 2.8,
        fc=C["panel"], ec=C["border"])
    txt(ax, 5.55, 5.1,
        "Probability Comparison",
        size=8, bold=True, color=C["text"])
    p_cvd = [0.7451, 0.6924, 0.7613]
    p_no  = [0.2549, 0.3076, 0.2387]
    for mi, (pc, pn, m) in enumerate(
            zip(p_cvd, p_no, ["LR","RF","XGB"])):
        xi = 1.0 + mi*3.1
        box(ax, xi,    2.65, 0.8, pn*5.5,
            fc=C["green"]+"88", ec=C["green"], lw=0.8)
        box(ax, xi+1.0,2.65, 0.8, pc*5.5,
            fc=C["red"]+"88", ec=C["red"], lw=0.8)
        txt(ax, xi+0.9, 2.45, m,
            size=7.5, bold=True, color=C["muted"])
    ax.axhline(y=2.65+0.5*5.5,
               xmin=0.02, xmax=0.62,
               color=C["amber"],lw=1.5,ls="--")
    txt(ax, 9.5, 5.3,
        "Threshold\n50%",
        size=6, color=C["amber"])

    box(ax, 11.1, 2.5, 6.6, 2.8,
        fc=C["red"]+"11", ec=C["red"], lw=1.2)
    txt(ax, 14.4, 5.1,
        "[X] High CVD Risk — Actions",
        size=8, bold=True, color=C["red"])
    for ai, act in enumerate([
        "Urgent cardiology consult (24-48h)",
        "Daily BP monitoring (<140/90 mmHg)",
        "Stop smoking immediately",
        "DASH diet (low sodium)",
        "No strenuous exercise yet",
        "Request ECG + full lipid panel",
    ]):
        txt(ax, 11.3, 4.72-ai*0.37,
            f"  {act}",
            size=6.5, color=C["text"], ha="left")
    hcd_tag(ax, 11.1, 2.35,
            "[NRM] Actionable guidance (Norman Feedback)",
            C["red"], 6.3)

    # Disclaimer
    box(ax, 0.3, 0.3, 17.4, 1.9,
        fc=C["amber"]+"11", ec=C["amber"], lw=1.2)
    txt(ax, 9, 2.02,
        "[!] CLINICAL DISCLAIMER (HCD Error Prevention, Nielsen #5)",
        size=8, bold=True, color=C["amber"])
    txt(ax, 9, 1.55,
        "This AI prediction is for EDUCATIONAL and RESEARCH purposes "
        "only. Does NOT constitute medical advice,",
        size=7.5, color=C["text"])
    txt(ax, 9, 1.12,
        "diagnosis or treatment. Consult a qualified healthcare "
        "professional for clinical decisions.",
        size=7.5, color=C["text"])
    txt(ax, 9, 0.65,
        "Norman (2013)  |  ISO 9241-210  |  WCAG 2.1  |  "
        "PSB605IT  |  Sudhan Nagarajan (050J0DAD)",
        size=6.5, color=C["dim"])

    plt.tight_layout()
    _save(fig, "wf10_risk_result")


# ═══════════════════════════════════════════════════════════════════
#  WF-11  WIREFRAME INDEX  (summary page)
# ═══════════════════════════════════════════════════════════════════
def wf11_index():
    fig, ax = setup_fig(20, 14,
        "WF-INDEX | Wireframe Catalogue | "
        "PSB605IT | Sudhan Nagarajan (050J0DAD)")

    box(ax, 0.3, 12.5, 19.4, 1.3,
        fc=C["purple"]+"33", ec=C["purple"], lw=2)
    txt(ax, 10, 13.28,
        "CVD Risk Prediction — HCD/UCD Wireframe Catalogue",
        size=14, bold=True, color=C["purple"])
    txt(ax, 10, 12.72,
        "PSB605IT | Sudhan Nagarajan (050J0DAD) | "
        "PSB Academy 2025/2026 | "
        "HCD (Norman 2013) | ISO 9241-210 | Nielsen (1994)",
        size=8, color=C["muted"])

    wf_list = [
        ("WF-00","System Architecture",
         "Full 4-layer system: Data->Pipeline->GUI->HCD",
         "All layers","all",               C["purple"]),
        ("WF-01","Home Page",
         "Hero, stat cards, champion results, nav guide",
         "Nielsen #1 #6","[HOME]",         C["blue"]),
        ("WF-02","Patient Risk Predictor",
         "Clinical form, ensemble result, risk banner",
         "Nielsen #2 #5","[PT]",           C["green"]),
        ("WF-03","Classifier Playground",
         "Model select, threshold, pie chart, breakdown",
         "Nielsen #3 #7","[LAB]",          C["amber"]),
        ("WF-04","Model Performance",
         "Metrics heatmap, ROC curves, confusion matrices",
         "Nielsen #1 #4","[MDL]",          C["purple"]),
        ("WF-05","SHAP Explainability",
         "SHAP bar chart, feature guide, trust building",
         "Norman Transparency","[SHAP]",   C["blue"]),
        ("WF-06","HCD Design Process",
         "ISO 9241-210 cycle, Nielsen 10, Norman principles",
         "ISO 9241-210","[HCD]",           C["red"]),
        ("WF-07","Predictions Viewer",
         "Data table, filters, sort, dual download",
         "Nielsen #3 #7","[PRED]",         C["green"]),
        ("WF-08","Model Comparison",
         "Side-by-side tables, bar chart, radar chart",
         "Nielsen #4 #8","[CMP]",          C["amber"]),
        ("WF-09","User Journey Map",
         "3 personas (Dr/Researcher/Patient), task flows",
         "ISO 9241-210 UCD","[UCD]",       C["blue"]),
        ("WF-10","Risk Result Detail",
         "HIGH risk banner, gauge, 3 cards, clinical advice",
         "Norman Feedback","[HIGH]",       C["red"]),
    ]

    for i, (wf_id, title, desc, principle, icon, clr) in \
            enumerate(wf_list):
        xi = 0.4 + (i % 3)*6.55
        yi = 10.8 - (i // 3)*2.75
        box(ax, xi, yi, 6.3, 2.5,
            fc=clr+"15", ec=clr, lw=1.5)

        # WF id badge
        box(ax, xi+0.12, yi+2.05, 1.2, 0.35,
            fc=clr, ec=clr, lw=0, radius=0.06)
        txt(ax, xi+0.72, yi+2.22, wf_id,
            size=7, bold=True, color="#fff")

        # Icon badge
        box(ax, xi+5.0, yi+2.05, 1.1, 0.35,
            fc=clr+"33", ec=clr, lw=0.8, radius=0.06)
        txt(ax, xi+5.55, yi+2.22, icon,
            size=7, bold=True, color=clr)

        txt(ax, xi+0.18, yi+1.72, title,
            size=9, bold=True, color=clr, ha="left")
        txt(ax, xi+0.18, yi+1.32, desc,
            size=6.5, color=C["muted"], ha="left")
        hline(ax, xi+0.18, yi+1.1, 5.9,
              color=clr+"44", lw=0.7)
        txt(ax, xi+0.18, yi+0.85,
            f"HCD: {principle}",
            size=6, color=clr, ha="left", bold=True)

        # Thumbnail placeholder
        box(ax, xi+0.18, yi+0.12, 5.9, 0.62,
            fc=clr+"11", ec=clr+"44",
            lw=0.5, radius=0.06)
        txt(ax, xi+3.13, yi+0.43,
            f"-> outputs/wireframes/{wf_id.lower().replace('-','')}"
            f"_*.png",
            size=6, color=clr+"99")

    # Design principles legend
    box(ax, 0.3, 0.3, 19.4, 1.8,
        fc=C["panel"], ec=C["border"])
    txt(ax, 10, 1.9,
        "HCD/UCD Design Standard — Applied Principles",
        size=8, bold=True, color=C["text"])
    principles = [
        ("[N]#1 Visibility",    C["blue"]),
        ("[N]#2 Real World",    C["green"]),
        ("[N]#3 User Control",  C["amber"]),
        ("[N]#4 Consistency",   C["purple"]),
        ("[N]#5 Error Prev",    C["red"]),
        ("[N]#6 Recognition",   C["blue"]),
        ("[N]#7 Flexibility",   C["green"]),
        ("[N]#8 Minimalist",    C["amber"]),
        ("[N]#9 Recovery",      C["purple"]),
        ("[N]#10 Help",         C["muted"]),
    ]
    for i, (p, c) in enumerate(principles):
        xi = 0.5 + i*1.94
        box(ax, xi, 0.42, 1.82, 0.45,
            fc=c+"22", ec=c, lw=0.8, radius=0.06)
        txt(ax, xi+0.91, 0.64, p,
            size=5.8, bold=True, color=c)

    txt(ax, 10, 0.25,
        "N = Nielsen (1994)  |  NRM = Norman (2013)  |  "
        "ISO = ISO 9241-210:2019  |  WCAG 2.1 AA  |  "
        "PSB605IT | PSB Academy 2025/2026",
        size=6, color=C["dim"])

    plt.tight_layout()
    _save(fig, "wf11_index")


# ═══════════════════════════════════════════════════════════════════
#  SAVE HELPER
# ═══════════════════════════════════════════════════════════════════
def _save(fig, name):
    path = f"outputs/wireframes/{name}.png"
    plt.savefig(path, dpi=150,
                bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    size = os.path.getsize(path) // 1024
    print(f"  [Saved] {path}  ({size} KB)")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("CVD Risk Prediction — HCD/UCD Wireframe Generator v2")
    print("PSB605IT | Sudhan Nagarajan (050J0DAD)")
    print("Norman (2013) | ISO 9241-210 | Nielsen (1994)")
    print("FIX: ASCII icons — no emoji font warnings")
    print("=" * 65)

    wireframes = [
        ("WF-00 System Architecture",    wf00_architecture),
        ("WF-01 Home Page",              wf01_home),
        ("WF-02 Patient Risk Predictor", wf02_patient_predictor),
        ("WF-03 Classifier Playground",  wf03_classifier_playground),
        ("WF-04 Model Performance",      wf04_model_performance),
        ("WF-05 SHAP Explainability",    wf05_shap),
        ("WF-06 HCD Design Process",     wf06_hcd_process),
        ("WF-07 Predictions Viewer",     wf07_predictions_viewer),
        ("WF-08 Model Comparison",       wf08_model_comparison),
        ("WF-09 User Journey Map",       wf09_user_journey),
        ("WF-10 Risk Result Detail",     wf10_risk_result),
        ("WF-11 Wireframe Index",        wf11_index),
    ]

    ok = err = 0
    for name, fn in wireframes:
        print(f"\n[Generating] {name}...")
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            err += 1

    print("\n" + "=" * 65)
    print(f"DONE — {ok} wireframes saved, {err} errors")
    print("Output -> outputs/wireframes/")
    print("=" * 65)

    files = sorted(os.listdir("outputs/wireframes"))
    total = sum(os.path.getsize(
        f"outputs/wireframes/{f}") for f in files) // 1024
    print(f"\n{len(files)} files  |  Total: {total} KB\n")
    for f in files:
        kb = os.path.getsize(f"outputs/wireframes/{f}") // 1024
        print(f"  {f:<50} {kb:>5} KB")

    print(f"\nHCD References:")
    for ref in [
        "Norman (2013) The Design of Everyday Things",
        "ISO 9241-210:2019 Human-Centred Design",
        "Nielsen (1994) 10 Usability Heuristics",
        "WCAG 2.1 Web Content Accessibility Guidelines",
        "Hosmer et al. (2013) Applied Logistic Regression",
        "Breiman (2001) Random Forests",
        "Chen & Guestrin (2016) XGBoost",
        "Lundberg & Lee (2017) SHAP",
        "Pedregosa et al. (2011) scikit-learn",
        "Efron & Tibshirani (1993) Bootstrap",
        "Ulianova (2019) CARDIO Dataset",
        "Detrano et al. (1989) UCI Dataset",
    ]:
        print(f"  {ref}")
    print("\nPSB605IT | PSB Academy | 2025/2026")