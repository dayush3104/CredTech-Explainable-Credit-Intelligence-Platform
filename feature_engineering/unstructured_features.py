# feature_engineering/unstructured_features.py
import os
import pandas as pd
from pathlib import Path
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, USE_LIGHT_NLP, FINBERT_MODEL_NAME, \
                   BERT_MODEL_NAME, ENABLE_NLP_ENSEMBLE, NLP_WEIGHTS
from utils.logging_utils import setup_logger


logger = setup_logger("unstructured_features")


# Optional imports (lazily initialised)
_finbert = None
_bert = None
_vader = None

def _lazy_vader():
    global _vader
    if _vader is None:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            _vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"VADER unavailable: {e}")
            _vader = False
    return _vader

def _lazy_hf(model_name):
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)
    except Exception as e:
        logger.warning(f"HF model {model_name} unavailable: {e}")
        return False

def _lazy_finbert():
    global _finbert
    if _finbert is None:
        _finbert = _lazy_hf(FINBERT_MODEL_NAME)
    return _finbert

def _lazy_bert():
    global _bert
    if _bert is None:
        _bert = _lazy_hf(BERT_MODEL_NAME)
    return _bert

def _score_finbert(text: str) -> float | None:
    clf = _lazy_finbert()
    if not clf: return None
    try:
        res = clf(text)[0]  # list of dicts
        d = {r["label"].lower(): r["score"] for r in res}
        # Map to [-1,1] with neutral ~0
        return (d.get("positive", 0) - d.get("negative", 0))
    except Exception:
        return None

def _score_bert(text: str) -> float | None:
    clf = _lazy_bert()
    if not clf: return None
    try:
        res = clf(text)[0]
        # models often output POSITIVE/NEGATIVE
        d = {r["label"].lower(): r["score"] for r in res}
        if "positive" in d or "negative" in d:
            return d.get("positive", 0) - d.get("negative", 0)
        # if it’s 5-star style, approximate:
        labels = [r["label"] for r in res]
        if any(l.startswith(str(i)) for i in range(1,6) for l in labels):
            # weighted stars → [-1,1]
            score = sum((int(r["label"][0]) - 3) * r["score"] for r in res) / 2.0
            return float(score)
    except Exception:
        return None

def _score_vader(text: str) -> float | None:
    vader = _lazy_vader()
    if not vader: return None
    try:
        return float(vader.polarity_scores(text)["compound"])
    except Exception:
        return None

def _score_text(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0

    if USE_LIGHT_NLP:
        s = _score_vader(text)
        return 0.0 if s is None else s

    if ENABLE_NLP_ENSEMBLE:
        parts, w = [], NLP_WEIGHTS
        for name, fn in [("finbert", _score_finbert), ("bert", _score_bert), ("vader", _score_vader)]:
            s = fn(text)
            if s is not None and w.get(name, 0) > 0:
                parts.append((s, w[name]))
        if not parts:  # fallback
            s = _score_vader(text)
            return 0.0 if s is None else s
        # weighted average clipped to [-1,1]
        num = sum(s*wt for s, wt in parts)
        den = sum(wt for _, wt in parts)
        return float(max(-1.0, min(1.0, num/den)))
    else:
        s = _score_finbert(text)
        if s is None:
            s = _score_vader(text)
        return 0.0 if s is None else s

def analyze_sentiment():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / "news.csv"
    if not raw_path.exists():
        logger.warning("news.csv not found; skipping NLP.")
        return

    df = pd.read_csv(raw_path, encoding="utf-8")
    # expected columns: ticker, date, title, source
    keep = ["ticker", "date", "title", "source"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sentiment_score"] = df["title"].astype(str).apply(_score_text)

    # Save article-level and daily aggregates
    (PROCESSED_DATA_DIR / "news_with_sentiment.csv").write_text(
        df.to_csv(index=False), encoding="utf-8"
    )

    daily = df.groupby(["date", "ticker"], as_index=False)["sentiment_score"].mean()
    daily.rename(columns={"sentiment_score": "avg_sentiment_score"}, inplace=True)
    out = PROCESSED_DATA_DIR / "daily_sentiment.csv"
    daily.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Saved {out} ({len(daily)} rows)")
