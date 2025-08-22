import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.logging_utils import setup_logger
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, ISSUERS,
    TARGET_VARIABLE, TEST_SIZE, RANDOM_STATE, LGBM_PARAMS
)

logger = setup_logger("train")

def _load_features(ticker: str) -> pd.DataFrame:
    """Load processed feature file for a given ticker."""
    safe = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    f = PROCESSED_DATA_DIR / f"features_{safe}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing features for {ticker}: {f}")
    return pd.read_csv(f, parse_dates=["Date"]).set_index("Date")

def train_models():
    """Train one model per issuer and save it."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for t in ISSUERS.keys():
        try:
            df = _load_features(t)
        except Exception as e:
            logger.warning(f"Skipping {t}: {e}")
            continue

        if TARGET_VARIABLE not in df.columns:
            logger.warning(f"Target {TARGET_VARIABLE} missing for {t}, skipping.")
            continue

        y = df[TARGET_VARIABLE].astype(float)
        X = df.drop(columns=[TARGET_VARIABLE])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # --- Safety checks ---
        if X.empty or y.empty:
            logger.warning(f"No data available for {t}, skipping training.")
            continue
        if len(X) < 50:  # too few samples
            logger.warning(f"Not enough rows ({len(X)}) for {t}, skipping training.")
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
            )

            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"Train/test split failed for {t}, skipping.")
                continue

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )

            pred = model.predict(X_test)
            from sklearn.metrics import root_mean_squared_error
            rmse = root_mean_squared_error(y_test, pred)

            logger.info(f"{t}: RMSE={rmse:.4f}, train_rows={len(X_train)}, test_rows={len(X_test)}")

            out = MODELS_DIR / f"model_{t.replace('.', '_').replace('^','')}.joblib"
            joblib.dump(model, out)
            logger.info(f"Saved model -> {out.name}")

        except Exception as e:
            logger.error(f"Training failed for {t}: {e}")

if __name__ == "__main__":
    train_models()
