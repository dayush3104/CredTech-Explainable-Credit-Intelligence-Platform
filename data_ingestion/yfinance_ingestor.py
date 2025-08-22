
# data_ingestion/yfinance_ingestor.py
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import RAW_DATA_DIR, DATA_START_DATE, ISSUERS, SECTOR_ETFS, MACRO_TICKERS, COMMODITY_TICKERS
from utils.logging_utils import setup_logger

logger = setup_logger("yfinance_ingestor")

def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for ticker {ticker}")
    df = df.reset_index()
    return df

def fetch_yfinance_data():
    all_tickers = list(ISSUERS.keys()) + list(SECTOR_ETFS.keys()) + list(MACRO_TICKERS.keys()) + list(COMMODITY_TICKERS.keys())
    end = datetime.utcnow().strftime("%Y-%m-%d")
    logger.info(f"Fetching {len(all_tickers)} tickers from {DATA_START_DATE} to {end}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for t in all_tickers:
        safe = t.replace("^", "").replace("=", "_").replace(".", "_")
        out = RAW_DATA_DIR / f"{safe}.csv"
        try:
            df = _download_one(t, DATA_START_DATE, end)
            df.to_csv(out, index=False, encoding="utf-8")
            logger.info(f"Saved {t} -> {out.name} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Failed to fetch {t}: {e}")

if __name__ == "__main__":
    fetch_yfinance_data()
