
# main.py
from data_ingestion.yfinance_ingestor import fetch_yfinance_data
from data_ingestion.news_ingestor import fetch_news
from feature_engineering.unstructured_features import analyze_sentiment
from data_ingestion.worldbank_ingestor import fetch_worldbank_data
from feature_engineering.structured_features import process_structured_and_build_features
from modeling.train import train_models
from modeling.explain import generate_shap_values
from utils.mock_data_generator import create_mock_agency_ratings
from utils.logging_utils import setup_logger

logger = setup_logger("main")

def run_pipeline():
    logger.info("=== Ingesting market data ===")
    fetch_yfinance_data()

    fetch_worldbank_data()


    logger.info("=== Ingesting news ===")
    fetch_news()

    logger.info("=== NLP sentiment ===")
    analyze_sentiment()

    logger.info("=== Feature engineering ===")
    process_structured_and_build_features()

    logger.info("=== Model training ===")
    train_models()

    logger.info("=== SHAP explanations ===")
    generate_shap_values()

    logger.info("=== Mock ratings ===")
    create_mock_agency_ratings()

    logger.info("=== Pipeline complete ===")

if __name__ == "__main__":
    run_pipeline()
