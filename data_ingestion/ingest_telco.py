"""
data_ingestion/ingest_telco.py
Downloads IBM Telco Customer Churn dataset and saves to local Bronze layer.
"""

import io
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

RAW_DATA_DIR = Path("data/raw")
BRONZE_DATA_DIR = Path("data/bronze")


def download_telco_data() -> pd.DataFrame:
    """Download Telco dataset from GitHub."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DATA_DIR / "telco_churn.csv"

    if cache_path.exists():
        logger.info(f"Using cached file: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info("Downloading IBM Telco Customer Churn dataset...")
    response = requests.get(TELCO_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df.to_csv(cache_path, index=False)
    logger.info(f"Downloaded {len(df):,} rows → saved to {cache_path}")
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize before saving to Bronze."""
    df = df.copy()

    # Standardize column names to snake_case
    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Fix TotalCharges — blank for brand new customers
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(0.0)

    # Add ingestion metadata
    df["_ingested_at"] = datetime.utcnow().isoformat()
    df["_source"] = "ibm_telco"
    df["_batch_id"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Cleaned: {len(df):,} rows | {len(df.columns)} columns")
    return df


def run():
    logger.info("=" * 55)
    logger.info("  Telco Churn Ingestion Pipeline")
    logger.info("=" * 55)

    df_raw = download_telco_data()
    df_clean = clean_telco_data(df_raw)

    BRONZE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_DATA_DIR / "bronze_telco_customers.parquet"
    df_clean.to_parquet(out_path, index=False)

    logger.info(f"Saved {len(df_clean):,} rows → {out_path}")
    logger.info("Ingestion complete!")

    # Quick summary
    churn_rate = (df_clean["churn"].str.upper() == "YES").mean()
    logger.info(f"Churn rate in dataset: {churn_rate:.1%}")
    logger.info(f"Columns: {list(df_clean.columns)}")


if __name__ == "__main__":
    run()
