\# HealthAI – Final Report (Brief)



\## 1. Data \& Prep

\- Synthetic tabular (n=1000): age, gender, BMI, SBP, DBP, glucose, hypertension (derived), targets: diabetes\_risk, los\_days.

\- Preprocessing: OHE + StandardScaler via ColumnTransformer; 70/15/15 split.



\## 2. Models \& Metrics

\*\*Classification (diabetes\_risk)\*\*

\- Logistic Regression (class\_weight="balanced"): report val/test F1, ROC-AUC.

\- MLP (PyTorch): 15 epochs; report val/test AUC \& F1.

\- Confusion matrix figure: `reports/figures/confusion\_val\_cls.png`.



\*\*Regression (LOS)\*\*

\- Linear vs RandomForest; choose best by val MAE; report test MAE/RMSE/R².



\*\*Unsupervised\*\*

\- KMeans (k=3): elbow \& silhouette; PCA 2D plot (`reports/figures/kmeans\_pca\_val.png`).

\- Association rules (Apriori ≥10% support): top rules in `reports/association\_rules.csv`.



\*\*NLP\*\*

\- Sentiment: DistilBERT SST-2 (pipeline).

\- Clinical notes: TF-IDF + Logistic (synthetic notes baseline).

\- Translator stub: EN→TA/HI dictionary demo.



\## 3. App

\- Streamlit tabs: Risk, Sentiment, Notes, Translator.

\- FastAPI endpoints: `/predict\_risk`, `/sentiment`, `/translate/{ta|hi}`.



\## 4. Results (fill from your console)

\- LogReg: val AUC=…, F1=…; test AUC=…, F1=…

\- MLP: val AUC=…, F1=…; test AUC=…, F1=…

\- Regression best: MAE=…, RMSE=…, R²=…

\- KMeans silhouette ≈ …

\- Example association rule with lift > 1.2.



\## 5. How to Run

```bash

.\\.venv\\Scripts\\activate

python -m streamlit run src\\app\\streamlit\_app.py

uvicorn src.api.main:app --reload



