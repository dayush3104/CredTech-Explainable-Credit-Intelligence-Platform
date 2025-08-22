# data_ingestion/worldbank_ingestor.py
import pandas as pd
from pathlib import Path
from utils.logging_utils import setup_logger
from config import RAW_DATA_DIR, ECONOMIC_INDICATORS
import wbdata

logger = setup_logger("worldbank_ingestor")

def fetch_worldbank_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Map config indicators to World Bank codes
    indicator_map = {
        "IND_GDP": "NY.GDP.MKTP.KD.ZG",   # GDP growth (annual %)
        "IND_CPI": "FP.CPI.TOTL.ZG",      # Inflation, CPI (%)
        "US_RATE": "FR.INR.LEND"          # Lending interest rate
    }

    for econ, wb_code in indicator_map.items():
        try:
            logger.info(f"Fetching {econ} ({wb_code}) from World Bank API...")
            
            # Fetch data (date will be index, indicator is column)
            data = wbdata.get_dataframe(
                {wb_code: "Value"},
                country=("IND" if "IND" in econ else "USA")
            )

            # Reset index (date is index by default, often year as string)
            data = data.reset_index().rename(columns={"date": "Date"})

            # Convert Date safely
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data = data.sort_values("Date").dropna(subset=["Date"])

            # Save to raw folder
            out_path = RAW_DATA_DIR / f"{econ}.csv"
            data.to_csv(out_path, index=False)
            logger.info(f"Saved {econ} data -> {out_path.name} ({len(data)} rows)")
        
        except Exception as e:
            logger.error(f"Failed to fetch {econ}: {e}")

if __name__ == "__main__":
    fetch_worldbank_data()
