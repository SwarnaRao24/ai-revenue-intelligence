"""
data_ingestion/ingest_ecommerce.py
Generates synthetic e-commerce + support ticket data → Bronze layer.
(Mirrors the Olist schema without needing Kaggle credentials)
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

BRONZE_DATA_DIR = Path("data/bronze")
RANDOM_SEED = 42
N_CUSTOMERS = 5000


def generate_customers(rng, fake, n: int) -> pd.DataFrame:
    logger.info(f"Generating {n:,} customers...")
    return pd.DataFrame({
        "customer_id":    [f"CUST_{i:06d}" for i in range(n)],
        "customer_name":  [fake.name() for _ in range(n)],
        "customer_email": [fake.email() for _ in range(n)],
        "city":           [fake.city() for _ in range(n)],
        "state":          [fake.state_abbr() for _ in range(n)],
        "signup_date":    pd.date_range("2020-01-01", periods=n, freq="1h")[:n],
        "_ingested_at":   datetime.utcnow().isoformat(),
        "_source":        "synthetic_ecommerce",
        "_batch_id":      datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    })


def generate_orders(rng, customers: pd.DataFrame) -> pd.DataFrame:
    n_orders = len(customers) * 3
    logger.info(f"Generating {n_orders:,} orders...")

    customer_ids = rng.choice(customers["customer_id"].values, size=n_orders)
    order_dates = pd.to_datetime(
        rng.choice(
            pd.date_range("2020-01-01", "2024-06-01", freq="D"),
            size=n_orders
        )
    )

    return pd.DataFrame({
        "order_id":       [f"ORD_{i:08d}" for i in range(n_orders)],
        "customer_id":    customer_ids,
        "order_date":     order_dates,
        "order_status":   rng.choice(
            ["delivered", "shipped", "cancelled", "processing"],
            size=n_orders,
            p=[0.75, 0.12, 0.08, 0.05]
        ),
        "payment_value":  rng.lognormal(mean=4.0, sigma=0.8, size=n_orders).round(2),
        "freight_value":  rng.exponential(scale=15, size=n_orders).round(2),
        "_ingested_at":   datetime.utcnow().isoformat(),
        "_source":        "synthetic_ecommerce",
    })


def generate_support_tickets(rng, customers: pd.DataFrame) -> pd.DataFrame:
    n_tickets = len(customers) // 2
    logger.info(f"Generating {n_tickets:,} support tickets...")

    return pd.DataFrame({
        "ticket_id":   [f"TKT_{i:06d}" for i in range(n_tickets)],
        "customer_id": rng.choice(customers["customer_id"].values, size=n_tickets),
        "created_at":  pd.to_datetime(
            rng.choice(
                pd.date_range("2020-01-01", "2024-06-01", freq="D"),
                size=n_tickets
            )
        ),
        "category":    rng.choice(
            ["billing", "technical", "shipping", "product", "account"],
            size=n_tickets
        ),
        "sentiment":   rng.choice(
            ["positive", "neutral", "negative"],
            size=n_tickets,
            p=[0.20, 0.50, 0.30]
        ),
        "resolved":    rng.choice([True, False], size=n_tickets, p=[0.80, 0.20]),
        "_ingested_at": datetime.utcnow().isoformat(),
        "_source":      "synthetic_ecommerce",
    })


def run():
    logger.info("=" * 55)
    logger.info("  E-Commerce Ingestion Pipeline")
    logger.info("=" * 55)

    BRONZE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    fake = Faker()
    Faker.seed(RANDOM_SEED)

    customers = generate_customers(rng, fake, N_CUSTOMERS)
    orders = generate_orders(rng, customers)
    tickets = generate_support_tickets(rng, customers)

    datasets = {
        "bronze_ecommerce_customers": customers,
        "bronze_ecommerce_orders": orders,
        "bronze_support_tickets": tickets,
    }

    for name, df in datasets.items():
        out = BRONZE_DATA_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Saved {len(df):,} rows → {out}")

    logger.info("E-Commerce ingestion complete!")


if __name__ == "__main__":
    run()