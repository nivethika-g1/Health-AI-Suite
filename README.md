\# HealthAI Suite

End-to-end mini suite for healthcare ML/NLP.



\## Features

\- Tabular risk prediction (LogReg baseline, MLP DL)

\- Regression (LOS) baseline

\- Clustering (KMeans) + Association Rules (Apriori)

\- NLP: Sentiment (DistilBERT), Clinical Notes TF-IDF baseline, Translator stub

\- Streamlit UI + FastAPI API



\## Quickstart

```bash

python -m venv .venv

.\\.venv\\Scripts\\activate

pip install -r requirements.txt

python src\\models\\train\_classification.py

python -m streamlit run src\\app\\streamlit\_app.py





data/            # raw, images, etc.

src/

&nbsp; app/           # streamlit

&nbsp; api/           # fastapi

&nbsp; models/        # training scripts (ML \& DL)

&nbsp; nlp/           # sentiment, notes, translator

artifacts/       # preprocessor

models/          # saved models

reports/         # figs, rules csv



