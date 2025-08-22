# config.py
# Central configuration for the CredTech Hackathon project

from pathlib import Path
import os
from datetime import datetime

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Universe ---
# Use Yahoo Finance tickers for India (NSE) with ".NS" suffix
ISSUERS = {
    "INFY.NS": "Infosys Ltd",
    "TCS.NS": "Tata Consultancy Services",
    "RELIANCE.NS": "Reliance Industries"
}

# Sector ETF (IT sector proxy)
SECTOR_ETFS = {
    "ITBEES.NS": "Nifty IT ETF"
}

# Macro indicators (broad market proxy)
MACRO_TICKERS = {
    "^NSEI": "NIFTY 50"
}

# Commodity (use yfinance symbols to avoid paid APIs by default)
COMMODITY_TICKERS = {
    "CL=F": "WTI Crude Oil Futures"
}

# --- Extra Structured Source (Economic Indicators from World Bank/FRED/MCA etc.) ---
# Each will be stored as CSV in data/raw with Date, Value columns
ECONOMIC_INDICATORS = {
    "IND_GDP": "India GDP Growth (%)",
    "IND_CPI": "India CPI Inflation (%)",
    "US_RATE": "US Federal Funds Rate (%)"
}
# --- Dates ---
DATA_START_DATE = "2022-01-01"
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

# --- NLP ---
USE_LIGHT_NLP = False                     # False => use transformers
FINBERT_MODEL_NAME = "ProsusAI/finbert"   # finance-tuned
BERT_MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"  # general sentiment
ENABLE_NLP_ENSEMBLE = True               # use FinBERT + general BERT (+ VADER as tie-breaker)
NLP_WEIGHTS = {"finbert": 0.6, "bert": 0.3, "vader": 0.1}

# --- Targets/Model ---
TARGET_VARIABLE = "credit_score"
TEST_SIZE = 0.2
RANDOM_STATE = 42

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "boosting_type": "gbdt",
    "max_depth": 7,
    "num_leaves": 31,
    "random_state": RANDOM_STATE
}

# --- Mock Agency Ratings ---
MOCK_AGENCY_RATINGS_PATH = DATA_DIR / "mock_agency_ratings.csv"

# --- Optional API Keys (only used if present) ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
