# src/app/streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go

# ---------------- Page ----------------
st.set_page_config(page_title="HealthAI Suite", page_icon="🩺", layout="wide")

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------- LIGHT palette (soft baby blue) ----------------
st.markdown(
    """
    <style>
      :root{
        --ink:#0F172A;           /* headings */
        --text:#334155;          /* body text */
        --muted:#64748B;
        --paper:#FFFFFF;         /* cards/inputs */
        --line:#D7E6FF;          /* borders */
        --primary:#4F8EF7;       /* button blue */
        --primary-hover:#3E7BEA; /* button hover */
        --bg:#F6FAFF;            /* page background */
        --health-blue:#2563EB;   /* brand blue for numbers */
        --accent:#1dd0e2;        /* >>> requested tab/page header color <<< */
      }

      .stApp{
        background:
          radial-gradient(900px 480px at 15% -10%, rgba(99,179,237,.18), transparent 55%),
          radial-gradient(1000px 520px at 95% 10%, rgba(34,197,194,.12), transparent 60%),
          var(--bg);
      }
      .block-container{ padding-top:0.8rem; padding-bottom:2rem; }

      /* hide sidebar + hamburger */
      [data-testid="stSidebar"]{display:none!important;}
      button[kind="header"]{display:none!important;}
      [data-testid="stHeader"]{z-index:0;}

      .narrow{max-width:980px;margin:0 auto;}
      .brand{ text-align:center;font-weight:900;margin:22px 0 6px 0;font-size:34px;letter-spacing:.2px;}
      .brand .h1{color:#2563EB;}
      .brand .h2{color:#10B981;}
      .tagline{ color:#6B7280; letter-spacing:.22em; font-size:.78rem; text-align:center; margin-bottom:18px; }
      .section-title{ color:#6B7280; text-align:center; margin:8px 0 14px 0; }

      .card{
        background:var(--paper);
        border:1px solid var(--line);
        border-radius:16px; padding:24px;
        box-shadow:0 8px 24px rgba(15,23,42,.08);
        transition:transform .12s ease, box-shadow .18s ease;
      }
      .card:hover{ transform:translateY(-2px); box-shadow:0 12px 28px rgba(15,23,42,.10); }
      .badge{
        width:46px;height:46px;border-radius:9999px;margin:0 auto 12px auto;
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 8px 18px rgba(79,142,247,.25);
      }
      .b1{background:#4F8EF7;} /* risk + notes */
      .b2{background:#06B6D4;} /* sentiment icon (home) */
      .b3{background:#06B6D4;} /* translator icon (home) */
      .b4{background:#4F8EF7;} /* clinical notes = same as risk */
      .card h3{ text-align:center;margin:0 0 6px 0;color:var(--ink); }
      .muted{ text-align:center;color:var(--muted); }

      /* >>> Page (tab) headers in accent color <<< */
      /* These target st.subheader() H2 elements on the feature pages.
         Home page isn't affected because it uses custom markup, not H2. */
      h2 { color: var(--accent) !important; }

      /* INPUTS — force white on light theme */
      .stNumberInput input, .stTextInput input, .stTextArea textarea{
        background:var(--paper) !important;
        color:var(--ink) !important;
        border:1px solid var(--line) !important;
        border-radius:12px !important;
        box-shadow:none !important;
      }
      /* select */
      div[data-baseweb="select"]>div{
        background:var(--paper) !important;
        border:1px solid var(--line) !important;
        border-radius:12px !important;
      }
      div[data-baseweb="select"] *{ color:var(--ink) !important; }

      /* labels */
      label, .stMarkdown p{ color:var(--text) !important; }

      /* THEMED primary button */
      .stButton>button{
        background:var(--primary);
        color:white; border:none; border-radius:12px;
        padding:.66rem 1rem; font-weight:700;
        box-shadow:0 8px 18px rgba(79,142,247,.25);
        transition:transform .08s ease, background .08s ease, box-shadow .18s ease;
      }
      .stButton>button:hover{
        background:var(--primary-hover);
        transform:translateY(-1px);
        box-shadow:0 10px 22px rgba(62,123,234,.28);
      }
      .stButton>button:focus-visible{ outline:3px solid #93C5FD77; }

      .metric-num{ font-size:2rem; font-weight:800; color:var(--health-blue); }
      .result-pad{ margin-top:.7rem; }
      .risk-title{ color:#29a1d6 !important; font-weight:800; margin:0 0 .5rem 0; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------- tiny SVG icons (white) --------
ICON_RISK = '<svg width="22" height="22" viewBox="0 0 24 24" stroke="white" fill="none" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2z"/></svg>'
ICON_SENT = '<svg width="22" height="22" viewBox="0 0 24 24" stroke="white" fill="none" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>'
ICON_TRANS= '<svg width="22" height="22" viewBox="0 0 24 24" stroke="white" fill="none" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"/></svg>'
ICON_NOTES= '<svg width="22" height="22" viewBox="0 0 24 24" stroke="white" fill="none" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>'

# -------- load artifacts --------
pre_path = Path("artifacts/preprocessor.joblib")
cls_path = Path("models/model_cls.joblib")
reg_lin = list(Path("models").glob("model_reg_*.joblib"))

pre = joblib.load(pre_path) if pre_path.exists() else None
clf = joblib.load(cls_path) if cls_path.exists() else None
reg = joblib.load(reg_lin[0]) if reg_lin else None

feature_cols = None
try:
    feature_cols = (
        pd.read_csv("data/raw/tabular.csv", nrows=1)
        .drop(columns=["diabetes_risk","los_days"])
        .columns.tolist()
    )
except Exception:
    pass

# -------- helpers --------
def nav_to(page:str):
    st.session_state.page = page
    if hasattr(st,"rerun"): st.rerun()
    elif hasattr(st,"experimental_rerun"): st.experimental_rerun()

def risk_gauge(prob: float, title="Risk"):
    v = float(np.clip(prob, 0, 1)) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix":"%","font":{"size":30,"color":"#2563EB"}},
            title={"text":title,"font":{"size":18,"color":"#1E3A8A"}},
            gauge={
                "axis":{
                    "range":[0,100],
                    "tickwidth":1,
                    "tickcolor":"#1F2937",
                    "tickfont":{"color":"#1F2937"}
                },
                "bar":{"thickness":0.22,"color":"#7ec6e6"},
                "steps":[
                    {"range":[0,33],"color":"#279acd"},
                    {"range":[33,66],"color":"#1c6e92"},
                    {"range":[66,100],"color":"#0f3c50"},
                ],
                "threshold":{"line":{"color":"#7ec6e6","width":4},"thickness":0.75,"value":v},
            },
        )
    )
    fig.update_layout(
        height=260, margin=dict(l=20,r=20,t=80,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def backbar():
    cols = st.columns([0.16,0.84])
    with cols[0]:
        if st.button("← Home", use_container_width=True):
            nav_to("home")
    with cols[1]: st.write("")

if "page" not in st.session_state:
    st.session_state.page = "home"

# -------- pages --------
def render_home():
    # (unchanged)
    st.markdown('<div class="narrow">', unsafe_allow_html=True)
    st.markdown('<div class="brand"><span class="h1">Health</span><span class="h2">AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">INTELLIGENT PATIENT CARE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Core Capabilities</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f'<div class="card"><div class="badge b1">{ICON_RISK}</div><h3>Risk Prediction</h3><div class="muted">Analyze patient vitals</div></div>', unsafe_allow_html=True)
        if st.button("Open Risk", use_container_width=True, key="risk"): nav_to("risk")
    with c2:
        st.markdown(f'<div class="card"><div class="badge b2">{ICON_SENT}</div><h3>Sentiment Analysis</h3><div class="muted">Understand patient mood</div></div>', unsafe_allow_html=True)
        if st.button("Open Sentiment", use_container_width=True, key="sent"): nav_to("sentiment")

    c3,c4 = st.columns(2, gap="large")
    with c3:
        st.markdown(f'<div class="card"><div class="badge b3">{ICON_TRANS}</div><h3>Translation</h3><div class="muted">Multi-language support</div></div>', unsafe_allow_html=True)
        if st.button("Open Translator", use_container_width=True, key="trans"): nav_to("translator")
    with c4:
        st.markdown(f'<div class="card"><div class="badge b4">{ICON_NOTES}</div><h3>Clinical Notes</h3><div class="muted">Analyze documentation</div></div>', unsafe_allow_html=True)
        if st.button("Open Notes", use_container_width=True, key="notes"): nav_to("notes")

    st.markdown('</div>', unsafe_allow_html=True)

def render_risk():
    backbar()
    # Heading in #29a1d6
    st.markdown('<h2 class="risk-title">Patient Risk</h2>', unsafe_allow_html=True)
    # --- brief explanation (added) ---
    st.caption("Enter patient vitals to estimate a probability of risk (0–100%). Models are for demo/education only.")

    if pre is None or clf is None:
        st.warning("Models not found. Train the models before using this page.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 95, 55)
        gender = st.selectbox("Gender", ["M", "F"])
    with c2:
        bmi = st.number_input("BMI", 10.0, 60.0, 27.0)
        glucose = st.number_input("Glucose (mg/dL)", 60, 300, 115)
    with c3:
        sbp = st.number_input("Systolic BP (mmHg)", 80, 220, 130)
        dbp = st.number_input("Diastolic BP (mmHg)", 40, 140, 82)

    row = {
        "age": age, "gender": gender, "bmi": bmi,
        "sbp": sbp, "dbp": dbp, "glucose": glucose,
        "hypertension": int(sbp > 140),
    }

    if st.button("Predict Risk", use_container_width=True):
        try:
            row_df = pd.DataFrame([row])
            if feature_cols:
                for col in feature_cols:
                    if col not in row_df:
                        row_df[col] = np.nan
                row_df = row_df[feature_cols]

            x = pre.transform(row_df)
            prob = float(clf.predict_proba(x)[:, 1][0])

            st.markdown("<div class='result-pad'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(risk_gauge(prob, title="Risk"), use_container_width=True)
                st.markdown(f'<div class="metric-num">{prob*100:0.2f}%</div>', unsafe_allow_html=True)
                st.caption("Probability estimate (0–100).")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def render_sentiment():
    backbar()
    st.markdown('<h2 class="risk-title">Sentiment Analysis</h2>', unsafe_allow_html=True)
    # --- brief explanation (added) ---
    st.caption("Classify a sentence as Positive/Negative using DistilBERT (SST-2 fine-tuned).")

    with st.container(border=True):
        txt = st.text_area("Input text", "Patient is feeling much better today.")
        if st.button("Analyze", use_container_width=True):
            from transformers import pipeline
            nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            out = nlp(txt)[0]
            st.json({"label": out["label"], "score": float(out["score"])})

def render_notes():
    backbar()
    st.markdown('<h2 class="risk-title">Clinical Notes → Risk (TF-IDF + Logistic Regression)</h2>', unsafe_allow_html=True)
    # --- brief explanation (added) ---
    st.caption("Type a short clinical note; a TF-IDF + Logistic Regression model estimates risk from the text.")

    note = st.text_area("Clinical note", "High glucose and obesity; recommend screening.")
    if st.button("Predict from Note", use_container_width=True):
        try:
            vec = joblib.load("models/notes_tfidf.joblib")
            logreg = joblib.load("models/notes_logreg.joblib")
            p = float(logreg.predict_proba(vec.transform([note]))[:,1][0])

            st.markdown("<div class='result-pad'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(risk_gauge(p, title="Risk (from note)"), use_container_width=True)
                st.markdown(f'<div class="metric-num">{p*100:0.2f}%</div>', unsafe_allow_html=True)
                st.caption("Probability derived from note content.")
        except Exception as e:
            st.error(f"Notes model not available: {e}")

def render_translator():
    backbar()
    st.markdown('<h2 class="risk-title">Translator (English → Tamil/Hindi)</h2>', unsafe_allow_html=True)
    # --- brief explanation (added) ---
    st.caption("Simple demo translator for patient-facing phrases: English ➜ Tamil or Hindi.")

    with st.container(border=True):
        txt = st.text_input("English text", "patient has high blood glucose")
        tgt = st.selectbox("Target language", ["ta (Tamil)","hi (Hindi)"])
        if st.button("Translate", use_container_width=True):
            from nlp.translator_stub import translate_en
            code = "ta" if tgt.startswith("ta") else "hi"
            st.write(translate_en(txt, target=code))

# -------- Router --------
page = st.session_state.page
if page=="home": render_home()
elif page=="risk": render_risk()
elif page=="sentiment": render_sentiment()
elif page=="notes": render_notes()
elif page=="translator": render_translator()
