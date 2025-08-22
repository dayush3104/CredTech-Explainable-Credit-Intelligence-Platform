
# modeling/explain.py
import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
from utils.logging_utils import setup_logger
from config import PROCESSED_DATA_DIR, MODELS_DIR, ISSUERS, TARGET_VARIABLE

logger = setup_logger("explain")

def generate_shap_values():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for t in ISSUERS.keys():
        safe_model = t.replace(".", "_").replace("^","")
        model_path = MODELS_DIR / f"model_{safe_model}.joblib"
        feat_path = PROCESSED_DATA_DIR / f"features_{t.replace('.', '_').replace('^','')}.csv"
        if not (model_path.exists() and feat_path.exists()):
            logger.warning(f"Missing artifacts for {t}; skipping SHAP.")
            continue

        model = joblib.load(model_path)
        df = pd.read_csv(feat_path, parse_dates=["Date"]).set_index("Date")
        if TARGET_VARIABLE not in df.columns:
            logger.warning(f"No target in features for {t}; skipping.")
            continue
        X = df.drop(columns=[TARGET_VARIABLE])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        # Save explanations (use joblib to keep objects)
        joblib.dump({"columns": X.columns.tolist(), "index": X.index, "values": shap_values}, MODELS_DIR / f"shap_{safe_model}.joblib")
        logger.info(f"Saved SHAP for {t}")

if __name__ == "__main__":
    generate_shap_values()
