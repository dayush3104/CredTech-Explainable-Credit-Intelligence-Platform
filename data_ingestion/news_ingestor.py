
# data_ingestion/news_ingestor.py
"""
News ingestion with graceful fallbacks:
1) If NEWS_API_KEY is set, uses NewsAPI.org.
2) Else, tries yfinance's .news for each ticker.
3) Else, loads sample news from data/raw/sample_news.csv (bundled).
Produces data/raw/news.csv with columns: ticker, date, title, source.
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests

from config import RAW_DATA_DIR, ISSUERS, NEWS_API_KEY
from utils.logging_utils import setup_logger

logger = setup_logger("news_ingestor")

NEWS_OUT = RAW_DATA_DIR / "news.csv"

def _from_newsapi() -> pd.DataFrame:
    headers = {"X-Api-Key": NEWS_API_KEY}
    since = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    rows = []
    for t, name in ISSUERS.items():
        params = {"q": f"\"{name}\"", "from": since, "language": "en", "pageSize": 50, "sortBy": "publishedAt"}
        try:
            resp = requests.get("https://newsapi.org/v2/everything", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            for a in articles:
                rows.append({
                    "ticker": t,
                    "date": a.get("publishedAt", "")[:10],
                    "title": a.get("title", ""),
                    "source": (a.get("source") or {}).get("name", ""),
                })
            logger.info(f"NewsAPI: {t} -> {len(articles)} articles")
        except Exception as e:
            logger.warning(f"NewsAPI failed for {t}: {e}")
    return pd.DataFrame(rows)

def _from_yfinance_news() -> pd.DataFrame:
    rows = []
    for t in ISSUERS.keys():
        try:
            tk = yf.Ticker(t)
            items = tk.news or []
            for a in items:
                # yfinance news schema varies; be defensive
                title = a.get("title") or ""
                provider = a.get("provider") or ""
                pub = a.get("providerPublishTime") or 0
                date = datetime.utcfromtimestamp(pub).strftime("%Y-%m-%d") if pub else ""
                if title:
                    rows.append({"ticker": t, "date": date, "title": title, "source": provider})
            logger.info(f"yfinance.news: {t} -> {len(items)} items")
        except Exception as e:
            logger.warning(f"yfinance.news failed for {t}: {e}")
    return pd.DataFrame(rows)

def _from_sample() -> pd.DataFrame:
    sample = RAW_DATA_DIR / "sample_news.csv"
    if sample.exists():
        logger.info("Loading bundled sample_news.csv")
        return pd.read_csv(sample)
    logger.warning("No sample_news.csv found; creating a tiny placeholder dataset.")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    rows = []
    for t in ISSUERS.keys():
        rows.append({"ticker": t, "date": today, "title": f"{t} announces quarterly results; outlook stable", "source": "Sample"})
    return pd.DataFrame(rows)

def fetch_news():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame()
    if NEWS_API_KEY:
        df = _from_newsapi()
    if df.empty:
        df = _from_yfinance_news()
    if df.empty:
        df = _from_sample()

    # Cleanup
    df["title"] = df["title"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "title"]).drop_duplicates(subset=["ticker", "date", "title"])
    df.to_csv(NEWS_OUT, index=False, encoding="utf-8")
    logger.info(f"Saved news -> {NEWS_OUT} ({len(df)} rows)")
    return NEWS_OUT

if __name__ == "__main__":
    fetch_news()
