
# utils/mock_data_generator.py
from datetime import datetime
import pandas as pd
from pathlib import Path
from config import MOCK_AGENCY_RATINGS_PATH, ISSUERS

# Simple mocked quarterly ratings (step series) on 0-100 scale for overlay
# AA = 85, A = 75, BBB = 65, BB = 55
RATING_MAP = {"AA": 85, "A": 75, "BBB": 65, "BB": 55}

def create_mock_agency_ratings():
    dates = pd.date_range("2024-01-01", periods=8, freq="Q")
    data = {t: [] for t in ISSUERS.keys()}
    for d in dates:
        # Rotate between AA and A to simulate slowly changing ratings
        for t in ISSUERS.keys():
            if d.quarter % 2 == 0:
                data[t].append(RATING_MAP["AA"])
            else:
                data[t].append(RATING_MAP["A"])
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    MOCK_AGENCY_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MOCK_AGENCY_RATINGS_PATH, encoding="utf-8")
    return MOCK_AGENCY_RATINGS_PATH
