import pandas as pd
import numpy as np
from pathlib import Path
from utils.logging_utils import setup_logger
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    ISSUERS, SECTOR_ETFS, MACRO_TICKERS, COMMODITY_TICKERS, ECONOMIC_INDICATORS
)

logger = setup_logger("structured_features")

def _load_csv(ticker: str) -> pd.DataFrame:
    safe = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    path = RAW_DATA_DIR / f"{safe}.csv"
    if not path.exists():
        logger.warning(f"Missing {path.name}")
        return pd.DataFrame()
    
    df = pd.read_csv(path, parse_dates=["Date"])
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Date")
    df = df.rename(columns={c: f"{safe}_{c.replace(' ', '')}" for c in df.columns})
    return df


def _tech_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df[price_col].pct_change()
    df["volatility_30d"] = df["returns"].rolling(30).std() * np.sqrt(252)
    df["momentum_5d"] = df[price_col].pct_change(5)
    df["momentum_20d"] = df[price_col].pct_change(20)
    df["sma_50"] = df[price_col].rolling(50).mean()
    df["sma_200"] = df[price_col].rolling(200).mean()
    return df


def process_structured_and_build_features():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load bases (finance, sector, macro, commodity)
    bases = {t: _load_csv(t) for t in (
        list(ISSUERS.keys())
        + list(SECTOR_ETFS.keys())
        + list(MACRO_TICKERS.keys())
        + list(COMMODITY_TICKERS.keys())
    )}

    # Load extra economic indicators
    econ_bases = {}
    for econ in ECONOMIC_INDICATORS.keys():
        path = RAW_DATA_DIR / f"{econ}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            df = df.rename(columns={"Value": f"econ_{econ}"})
            econ_bases[econ] = df
        else:
            logger.warning(f"Missing economic file: {econ}.csv")

    # Build features per issuer
    for t in ISSUERS.keys():
        safe = t.replace("^", "").replace("=", "_").replace(".", "_")
        base = bases.get(t, pd.DataFrame()).copy()
        if base.empty:
            logger.warning(f"No base price data for {t}; skipping feature build.")
            continue

        price_col = f"{safe}_AdjClose" if f"{safe}_AdjClose" in base.columns else f"{safe}_Close"
        base = _tech_indicators(base, price_col)

        feat = base.copy()

        # Join sector ETF
        for sec_t in SECTOR_ETFS.keys():
            df = bases.get(sec_t, pd.DataFrame())
            if not df.empty:
                feat = feat.join(df[[c for c in df.columns if c.endswith("AdjClose") or c.endswith("Close")]].add_prefix("sector_"), how="left")

        # Join macro
        for m in MACRO_TICKERS.keys():
            df = bases.get(m, pd.DataFrame())
            if not df.empty:
                feat = feat.join(df[[c for c in df.columns if c.endswith("AdjClose") or c.endswith("Close")]].add_prefix("macro_"), how="left")

        # Join commodities
        for cm in COMMODITY_TICKERS.keys():
            df = bases.get(cm, pd.DataFrame())
            if not df.empty:
                feat = feat.join(df[[c for c in df.columns if c.endswith("AdjClose") or c.endswith("Close")]].add_prefix("comm_"), how="left")

        # Join economic indicators
        for econ, df in econ_bases.items():
            if not df.empty:
                feat = feat.join(df, how="left")

        # Drop duplicate columns if any
        feat = feat.loc[:, ~feat.columns.duplicated()]

        # --- Fix for annual economic indicators ---
        feat = feat.sort_index()

        # Forward fill econ cols across daily index
        econ_cols = [c for c in feat.columns if c.startswith("econ_")]
        if econ_cols:
            feat[econ_cols] = feat[econ_cols].ffill()

        # Forward/backward fill everything else
        feat = feat.ffill().bfill()

        # Drop rows only if key features are NaN
        feat = feat.dropna(subset=["volatility_30d", "momentum_20d", "momentum_5d"])

        # --- Synthetic Credit Score ---
        vol = feat["volatility_30d"].clip(lower=0, upper=0.8).fillna(0)
        mom20 = feat["momentum_20d"].clip(-0.5, 0.5).fillna(0)
        mom5 = feat["momentum_5d"].clip(-0.5, 0.5).fillna(0)

        sec_cols = [c for c in feat.columns if c.startswith("sector_") and (c.endswith("AdjClose") or c.endswith("Close"))]
        mac_cols = [c for c in feat.columns if c.startswith("macro_") and (c.endswith("AdjClose") or c.endswith("Close"))]
        com_cols = [c for c in feat.columns if c.startswith("comm_") and (c.endswith("AdjClose") or c.endswith("Close"))]
        econ_cols = [c for c in feat.columns if c.startswith("econ_")]

        def ret(s): 
            return s.pct_change().clip(-0.3, 0.3).fillna(0)

        sec_ret = feat[sec_cols].mean(axis=1).pipe(ret) if sec_cols else pd.Series(0.0, index=feat.index)
        mac_ret = feat[mac_cols].mean(axis=1).pipe(ret) if mac_cols else pd.Series(0.0, index=feat.index)
        com_ret = feat[com_cols].mean(axis=1).pipe(ret) if com_cols else pd.Series(0.0, index=feat.index)
        econ_signal = feat[econ_cols].mean(axis=1).pct_change().clip(-0.2, 0.2).fillna(0) if econ_cols else pd.Series(0.0, index=feat.index)

        score = (
            65
            - 60 * vol
            + 25 * mom20
            + 10 * mom5
            + 10 * sec_ret
            + 5 * mac_ret
            - 8 * com_ret
            + 5 * econ_signal
        )
        score = (score.clip(0, 100)).rename("credit_score")
        feat = feat.join(score)

        out = PROCESSED_DATA_DIR / f"features_{safe}.csv"
        feat.to_csv(out, encoding="utf-8")
        logger.info(f"Saved base features for {t} -> {out.name} ({len(feat)} rows)")

    # --- Join sentiment later ---
    sent_path = PROCESSED_DATA_DIR / "daily_sentiment.csv"
    sent = pd.read_csv(sent_path) if sent_path.exists() else pd.DataFrame(columns=["ticker", "date", "avg_sentiment_score"])
    if not sent.empty:
        sent["date"] = pd.to_datetime(sent["date"])
    for t in ISSUERS.keys():
        safe = t.replace("^", "").replace("=", "_").replace(".", "_")
        fpath = PROCESSED_DATA_DIR / f"features_{safe}.csv"
        if not fpath.exists():
            continue
        feat = pd.read_csv(fpath, parse_dates=["Date"]).set_index("Date")
        s = sent[sent["ticker"] == t].copy()
        if not s.empty:
            s = s.set_index("date")[["avg_sentiment_score"]]
            feat = feat.join(s, how="left")
            feat["avg_sentiment_score"] = feat["avg_sentiment_score"].fillna(0.0)
            feat["credit_score"] = (feat["credit_score"] + 5 * feat["avg_sentiment_score"]).clip(0, 100)
        else:
            feat["avg_sentiment_score"] = feat["avg_sentiment_score"].fillna(0.0)
        feat.to_csv(fpath, encoding="utf-8")
        logger.info(f"Updated features with sentiment for {t} -> {fpath.name}")


if __name__ == "__main__":
    process_structured_and_build_features()
