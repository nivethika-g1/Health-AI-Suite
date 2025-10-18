# HealthAI Suite 🩺

A clean, sky-blue **Streamlit** app for quick clinical insights.

**Tabs included**
- **Patient Risk** — Tabular inputs → probability estimate (classic ML).
- **Sentiment** — Classify text with DistilBERT (SST-2).
- **Clinical Notes → Risk** — TF-IDF + Logistic Regression on short notes.
- **Translator** — English → Tamil/Hindi (simple demo).

> ⚠️ Educational demo only. Not medical advice.

---

## ✨ Highlights
- Minimal, responsive UI with custom CSS (light theme).
- Plotly **gauge** visual for probabilities.
- Reusable preprocessing (joblib) + modular code layout.
- Works locally with a few commands.

---

## ⚙️ How it Works 

***Risk (Tabular)*** -  Preprocessor (artifacts/preprocessor.joblib) transforms inputs → classifier (models/model_cls.joblib) outputs probability → Plotly gauge renders it.

***Sentiment*** - Hugging Face transformers pipeline with distilbert-base-uncased-finetuned-sst-2-english.

***Clinical Notes → Risk***  - notes_tfidf.joblib vectorizer + notes_logreg.joblib logistic regression.

***Translator*** - Simple rule/lexicon demo (src/nlp/translator_stub.py) for EN→TA/HI phrases.

---

## 📦 Project Structure
 ```plaintext     
HealthAI
      ├─ FINAL PROJECT (DS-C-WD-E-B68)HealthAI.docx.pdf     # Project problem statement
      ├─ src/                                               # Source code
      │  └─ app/
      │     └─ streamlit_app.py                             # Streamlit app (main UI)
      ├─ models/                                            # Saved ML artifacts
      │  ├─ model_cls.joblib                                # Tabular risk classifier
      │  ├─ notes_tfidf.joblib                              # TF-IDF vectorizer for notes
      │  └─ notes_logreg.joblib                             # Logistic regression (notes→risk)
      ├─ reports/                                           # Docs & figures
      │  ├─ REPORT.md
      │  └─ figures/
      ├─ assets/                                            # Images/icons/etc.
      ├─ .streamlit/                                        # Streamlit config
      │  └─ config.toml
      └─ requirements.txt                                   # Python dependencies


