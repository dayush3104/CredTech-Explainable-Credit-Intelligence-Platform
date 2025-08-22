import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import shap
from streamlit_shap import st_shap

from config import MODELS_DIR, PROCESSED_DATA_DIR, ISSUERS, MOCK_AGENCY_RATINGS_PATH

st.set_page_config(layout="wide", page_title="CredTech — Explainable Credit Intelligence")

@st.cache_data
def load_features(ticker: str):
    safe = ticker.replace(".", "_").replace("^","")
    path = PROCESSED_DATA_DIR / f"features_{safe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df

@st.cache_resource
def load_model_and_shap(ticker: str):
    safe = ticker.replace(".", "_").replace("^","")
    model_path = MODELS_DIR / f"model_{safe}.joblib"
    shap_path = MODELS_DIR / f"shap_{safe}.joblib"
    if not (model_path.exists() and shap_path.exists()):
        return None, None
    model = joblib.load(model_path)
    shap_bundle = joblib.load(shap_path)
    return model, shap_bundle

@st.cache_data
def load_news():
    p = PROCESSED_DATA_DIR / "news_with_sentiment.csv"
    if p.exists():
        df = pd.read_csv(p, encoding="utf-8")

        # Normalize column names (lowercase, strip)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})

        # Ensure consistent datetime column
        if "date" in df.columns:
            df["publishedAt"] = pd.to_datetime(df["date"], errors="coerce")
        elif "publishedat" in df.columns:
            df["publishedAt"] = pd.to_datetime(df["publishedat"], errors="coerce")
        else:
            df["publishedAt"] = pd.NaT

        # Ensure required cols
        for col in ["ticker", "title", "source", "sentiment_score"]:
            if col not in df.columns:
                df[col] = None

        return df[["ticker","publishedAt","title","source","sentiment_score"]]

    # fallback
    return pd.DataFrame(columns=["ticker","publishedAt","title","source","sentiment_score"])

@st.cache_data
def load_agency_ratings():
    if MOCK_AGENCY_RATINGS_PATH.exists():
        return pd.read_csv(MOCK_AGENCY_RATINGS_PATH, parse_dates=["Date"]).set_index("Date")
    return pd.DataFrame()

def plain_language_from_shap(shap_row: shap._explanation.Explanation, feature_names: list, k: int = 5):
    vals = np.abs(shap_row.values)
    order = np.argsort(vals)[::-1][:k]
    bullets = []
    for idx in order:
        fname = feature_names[idx]
        contrib = shap_row.values[idx]
        direction = "increased" if contrib > 0 else "decreased"
        reason = fname
        if "volatility_30d" in fname: reason = "higher 30-day volatility"
        elif "momentum_20d" in fname: reason = "20-day momentum"
        elif "momentum_5d" in fname: reason = "5-day momentum"
        elif "sector_" in fname: reason = "sector trend"
        elif "macro_" in fname: reason = "macro market trend"
        elif "comm_" in fname: reason = "commodity trend (e.g., crude)"
        elif "avg_sentiment_score" in fname: reason = "recent news sentiment"
        bullets.append(f"• {reason} {direction} the score by ~{abs(contrib):.2f} points")
    return bullets

# Sidebar
st.sidebar.header("Analyst Controls")
ticker = st.sidebar.selectbox("Select Issuer", options=list(ISSUERS.keys()), format_func=lambda x: f"{x} — {ISSUERS[x]}")

df = load_features(ticker)
model, shap_bundle = load_model_and_shap(ticker)
news_df = load_news()
agency = load_agency_ratings()

st.title("CredTech — Real-Time, Explainable Credit Intelligence")
st.caption("Dynamic scores, feature-level explanations, and event-aware insights.")

if df is None or model is None or shap_bundle is None:
    st.error("Artifacts missing. Please run the pipeline first: python main.py then reload this app.")
    st.stop()

# Alerts (bonus)
st.sidebar.subheader("⚠ Sudden Score Change Alerts")
alerts = []
for t in ISSUERS.keys():
    d = load_features(t)
    if d is None or len(d) < 2: 
        continue
    change = d["credit_score"].iloc[-1] - d["credit_score"].iloc[-2]
    if abs(change) >= 3.0:
        alerts.append((t, change))
if alerts:
    for (t, ch) in sorted(alerts, key=lambda x: -abs(x[1])):
        st.sidebar.write(f"{t}: {ch:+.2f} pts")
else:
    st.sidebar.write("No large changes in last update.")

# Tabs
# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Score Deconstruction", "Issuer Comparison", "Data Explorer"])

with tab1:
    st.subheader(f"Overview — {ISSUERS[ticker]} ({ticker})")

    latest = float(df["credit_score"].iloc[-1])
    delta = float(df["credit_score"].iloc[-1] - df["credit_score"].iloc[-2]) if len(df) >= 2 else 0.0

    c1, c2, c3 = st.columns(3)
    vol_col = [c for c in df.columns if c.endswith("volatility_30d")]
    vol_val = float(df[vol_col[0]].iloc[-1]) if vol_col else 0.0
    sent_val = float(df["avg_sentiment_score"].iloc[-1]) if "avg_sentiment_score" in df.columns else 0.0

    c1.metric("Latest Credit Score", f"{latest:.2f}", f"{delta:+.2f}")
    c2.metric("30D Volatility", f"{vol_val:.2%}")
    c3.metric("Recent News Sentiment", f"{sent_val:+.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["credit_score"], mode="lines", name="Dynamic Credit Score"))
    if not agency.empty and ticker in agency.columns:
        fig.add_trace(go.Scatter(
            x=agency.index, y=agency[ticker], mode="lines", name="Agency Rating (Mock)",
            line=dict(dash="dash"), connectgaps=True))
    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Score (0-100)", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recent News & Events")
    if not news_df.empty:
        nd = news_df[news_df["ticker"] == ticker].copy()
        nd = nd.sort_values("publishedAt", ascending=False)
        st.dataframe(nd[["publishedAt","title","source","sentiment_score"]].head(12),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No recent news found.")

with tab2:
    st.subheader("Score Deconstruction")
    st.caption("Understand how each feature contributed to the selected day's score.")

    dates = df.index.strftime("%Y-%m-%d").tolist()
    sel = st.selectbox("Choose date", options=dates, index=len(dates)-1)
    idx = dates.index(sel)

    shap_values = shap_bundle["values"]
    feature_names = shap_bundle["columns"]

    st.markdown("*Waterfall plot (local explanation)*")
    st_shap(shap.plots.waterfall(shap_values[idx]), height=370)

    st.markdown("*Top drivers (plain language)*")
    bullets = plain_language_from_shap(shap_values[idx], feature_names, k=5)
    st.write("\n".join(bullets))

    st.markdown("*Global importance (beeswarm)*")
    st_shap(shap.plots.beeswarm(shap_values), height=380)

with tab3:  # Issuer Comparison
    st.subheader("Issuer Comparison")
    picks = st.multiselect("Select issuers", options=list(ISSUERS.keys()),
                           default=list(ISSUERS.keys()))

    # Load features for each selected issuer
    series = {}
    for t in picks:
        d = load_features(t)
        if d is not None and "credit_score" in d.columns:
            series[t] = d["credit_score"]

    if not series:
        st.info("Run the pipeline first to compare issuers.")
    else:
        # Overlay chart
        fig = go.Figure()
        for t, s in series.items():
            fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=f"{t}"))
        fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Score", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        # Correlation of daily changes
        df_cmp = pd.DataFrame(series)
        dchg = df_cmp.diff().dropna()
        if not dchg.empty:
            import plotly.express as px
            corr = dchg.corr()
            heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation of daily score changes")
            st.plotly_chart(heat, use_container_width=True)

        # Latest snapshot
        latest = {t: float(s.dropna().iloc[-1]) for t, s in series.items() if len(s.dropna())}
        st.write(pd.DataFrame.from_dict(latest, orient="index", columns=["Latest score"]).sort_values("Latest score", ascending=False))

with tab4:
    st.subheader("Data Explorer")
    cols = [c for c in df.columns if c != "credit_score"]
    pick = st.multiselect("Select features to plot", options=cols, default=[cols[0]] if cols else [])
    if pick:
        st.line_chart(df[pick])
    else:
        st.info("Select one or more features to visualize.")
