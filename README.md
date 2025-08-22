
# CredTech - Real‑Time Explainable Credit Intelligence

Produces dynamic **issuer‑level creditworthiness scores**, and explains every score with **feature‑level attributions** (SHAP) and **news‑driven context**. Ships with a **Streamlit dashboard**, robust pipeline, and Docker support.

## ✨ Highlights
- **Multi‑source ingestion** (equities, sector ETF, macro index, commodity; plus **news headlines**).
- **Adaptive scoring engine** (LightGBM) retrained on every refresh.
- **Explainability layer** with **SHAP** (waterfall + beeswarm) and **plain‑language** drivers.
- **Unstructured data integration** via VADER, BERT and **FinBERT**.
- **Analyst dashboard**: trend lines, alerts for sudden changes, news feed, and agency‑rating overlay.
- **Error‑tolerant** pipeline with graceful fallbacks and clear logging.

---

## 🧱 Project Structure
```
credtech_hackathon/
├─ data/
│  ├─ raw/                # downloaded CSVs (prices, news)
│  └─ processed/          # feature matrices, news with sentiment
├─ models/                # trained models + SHAP bundles
├─ data_ingestion/
│  ├─ yfinance_ingestor.py
│  └─ news_ingestor.py
├─ feature_engineering/
│  ├─ structured_features.py
│  └─ unstructured_features.py
├─ modeling/
│  ├─ train.py
│  └─ explain.py
├─ utils/
│  ├─ logging_utils.py
│  └─ mock_data_generator.py
├─ config.py
├─ main.py                # orchestrates the whole pipeline
├─ dashboard.py           # Streamlit app
└─ requirements.txt
```

---

## 🚀 Quickstart (Local)

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Provide News API Key
If you have a [newsapi.org](https://newsapi.org) key, export it so news ingestion uses it:
```bash
export NEWS_API_KEY="6689139f1e9f4f1e81206d1c36e3dc18"   # Windows PowerShell: $env:NEWS_API_KEY="YOUR_KEY_HERE"
```

> If No key given, The pipeline will fall back to `yfinance`'s built‑in news, and if that is unavailable, a bundled `sample_news.csv` is used so the app always runs.

### 4) Run the end‑to‑end pipeline
```bash
python main.py
```
This will:
- download market data (equities, sector ETF, macro index, crude futures),
- ingest news,
- compute sentiment and features,
- train a LightGBM model per issuer,
- pre‑compute SHAP values, and
- generate a mocked agency‑rating series for overlay.

### 5) Launch the dashboard
```bash
streamlit run dashboard.py
```
Open the URL shown (usually http://localhost:8501).

---

## 🧠 How the Score Works (Demo Mode)
We compute a **synthetic, but realistic** credit score combining: lower 30D volatility (good), positive price momentum (good), supportive sector & macro trends (good), and rising crude (generally bad for costs). **News sentiment** provides an incremental adjustment. We **train LightGBM** to fit this target so **SHAP** explanations are meaningful and fast.

> In a production system, you can replace the synthetic target with **historical credit events/ratings** or proprietary labels.

---

## 🧩 Dashboard Features
- **Overview** tab:
  - Dynamic score trend with **agency rating overlay** (step line).
  - KPI cards: latest score, 30D volatility, recent news sentiment.
  - **News feed** per issuer.
- **Score Deconstruction** tab:
  - **SHAP waterfall** for a selected date.
  - **Plain‑language** bullet points extracted from the top SHAP contributors.
  - **SHAP beeswarm** for global importance.
- **Data Explorer** tab: visualize any input feature.
- **Alerts** (sidebar): flags issuers with **sudden score changes** (configurable threshold).

---

## ⚙️ Configuration
Edit `config.py`:
- Universe of tickers (`ISSUERS`, `SECTOR_ETFS`, `MACRO_TICKERS`, `COMMODITY_TICKERS`).
- Dates, model parameters, and NLP mode (`USE_LIGHT_NLP`).

---

## 🧪 Troubleshooting
- **No charts / artifacts missing** → run `python main.py` first.
- **SHAP plots not showing** → ensure **shap** and **streamlit‑shap** installed, then refresh.
- **FinBERT too slow / heavy** → leave `USE_LIGHT_NLP = True` (default VADER).

---

## 🐳 Docker (Optional)
```dockerfile
# Dockerfile (example)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && python - <<'PY'\nimport nltk;nltk.download('vader_lexicon')\nPY
COPY . .
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
CMD ["bash","-lc","python main.py && streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0"]
```
Build & run:
```bash
docker build -t credtech .
docker run -p 8501:8501 -e NEWS_API_KEY=$NEWS_API_KEY credtech
```

---

## 🧯 Design Notes & Trade‑offs
- **Speed vs Accuracy**: LightGBM chosen for fast retrains with strong tabular performance.
- **Data sources**: to keep the project **key‑free** by default, market & commodity prices use `yfinance` tickers; news uses a **graceful      fallback** chain.
- **Explainability**: **model‑intrinsic** SHAP on tree ensembles, not LLM‑based summaries.
- **Storage**: file‑based CSVs for simplicity; swap for DB in production.
- **Scheduling**: run `main.py` via cron/GitHub Actions for daily refresh; adopt Airflow/Prefect later.

---

## 📜 License
IITK
