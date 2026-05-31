"""
ml_training/utils/data_loader.py
Loads feature data from local Bronze parquet files.
Falls back to synthetic data if files are missing.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

BRONZE_DIR = Path("data/bronze")


def load_churn_features() -> pd.DataFrame:
    """
    Build the full churn feature set from Bronze parquet files.
    Falls back to synthetic data if Bronze files are not found.
    """
    telco_path = BRONZE_DIR / "bronze_telco_customers.parquet"

    if not telco_path.exists():
        logger.warning("Bronze files not found — using synthetic data")
        return _generate_synthetic_churn_features()

    logger.info("Loading from Bronze layer...")
    df = pd.read_parquet(telco_path)
    df.columns = [c.lower() for c in df.columns]
    df = _engineer_churn_features(df)
    logger.info(f"Loaded {len(df):,} rows from Bronze")
    return df


def _engineer_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Telco columns into ML-ready features.
    Maps every column to a clean numeric representation.
    """
    feat = pd.DataFrame()

    feat["customer_id"] = df["customerid"]

    # ── Target ───────────────────────────────────────────
    feat["label"] = (df["churn"].str.upper() == "YES").astype(int)

    # ── Demographics ─────────────────────────────────────
    feat["is_male"]          = (df["gender"].str.lower() == "male").astype(int)
    feat["is_senior_citizen"] = df["seniorcitizen"].astype(int)
    feat["has_partner"]      = (df["partner"].str.upper() == "YES").astype(int)
    feat["has_dependents"]   = (df["dependents"].str.upper() == "YES").astype(int)

    # ── Account ───────────────────────────────────────────
    feat["tenure_months"]    = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    feat["monthly_charges"]  = pd.to_numeric(df["monthlycharges"], errors="coerce").fillna(0)
    feat["total_charges"]    = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)
    feat["implied_tenure"]   = (
        feat["total_charges"] / feat["monthly_charges"].replace(0, np.nan)
    ).fillna(0)

    # ── Contract ──────────────────────────────────────────
    feat["is_monthly_contract"]  = (df["contract"] == "Month-to-month").astype(int)
    feat["is_annual_contract"]   = (df["contract"] == "One year").astype(int)
    feat["is_biannual_contract"] = (df["contract"] == "Two year").astype(int)
    feat["uses_electronic_check"] = (
        df["paymentmethod"] == "Electronic check"
    ).astype(int)
    feat["has_paperless_billing"] = (
        df["paperlessbilling"].str.upper() == "YES"
    ).astype(int)

    # ── Services ──────────────────────────────────────────
    feat["has_fiber_optic"]    = (df["internetservice"] == "Fiber optic").astype(int)
    feat["has_dsl"]            = (df["internetservice"] == "DSL").astype(int)
    feat["no_internet"]        = (df["internetservice"] == "No").astype(int)
    feat["has_online_security"] = (df["onlinesecurity"] == "Yes").astype(int)
    feat["has_tech_support"]   = (df["techsupport"] == "Yes").astype(int)
    feat["has_phone"]          = (df["phoneservice"] == "Yes").astype(int)
    feat["has_multiple_lines"] = (df["multiplelines"] == "Yes").astype(int)
    feat["has_online_backup"]  = (df["onlinebackup"] == "Yes").astype(int)
    feat["has_device_protect"] = (df["deviceprotection"] == "Yes").astype(int)
    feat["has_streaming_tv"]   = (df["streamingtv"] == "Yes").astype(int)
    feat["has_streaming_movies"] = (df["streamingmovies"] == "Yes").astype(int)

    # ── Engineered Risk Signals ───────────────────────────
    feat["is_new_customer"]    = (feat["tenure_months"] < 12).astype(int)
    feat["is_long_tenure"]     = (feat["tenure_months"] > 48).astype(int)
    feat["charge_gap"]         = (
        feat["monthly_charges"] * feat["tenure_months"] - feat["total_charges"]
    )
    feat["services_count"] = (
        feat["has_online_security"] + feat["has_tech_support"] +
        feat["has_online_backup"] + feat["has_device_protect"] +
        feat["has_streaming_tv"] + feat["has_streaming_movies"]
    )
    feat["is_high_value"] = (feat["monthly_charges"] > 70).astype(int)
    feat["is_at_risk"] = (
        (feat["is_monthly_contract"] == 1) &
        (feat["tenure_months"] < 12)
    ).astype(int)

    # ── Synthetic engagement (since Telco has no transactions) ──
    rng = np.random.default_rng(42)
    n = len(feat)
    feat["total_support_tickets"]     = rng.integers(0, 8, size=n)
    feat["negative_support_tickets"]  = rng.integers(0, 4, size=n)
    feat["tickets_last_6mo"]          = rng.integers(0, 5, size=n)
    feat["total_orders_6mo"]          = rng.integers(0, 20, size=n)
    feat["total_revenue_6mo"]         = rng.uniform(0, 800, size=n).round(2)
    feat["avg_monthly_revenue_6mo"]   = rng.uniform(0, 150, size=n).round(2)
    feat["revenue_volatility_6mo"]    = rng.uniform(0, 40, size=n).round(2)
    feat["total_cancellations_6mo"]   = rng.integers(0, 4, size=n)
    feat["active_months_6mo"]         = rng.integers(1, 7, size=n)
    feat["lifetime_value"]            = (
        feat["monthly_charges"] * feat["tenure_months"] +
        feat["total_revenue_6mo"]
    ).round(2)
    feat["avg_order_value"]           = rng.uniform(20, 180, size=n).round(2)
    feat["total_orders"]              = rng.integers(1, 40, size=n)
    feat["negative_ticket_ratio"]     = (
        feat["negative_support_tickets"] /
        feat["total_support_tickets"].replace(0, 1)
    ).round(3)
    feat["charge_to_ltv_ratio"] = (
        feat["monthly_charges"] /
        feat["lifetime_value"].replace(0, np.nan)
    ).fillna(0).round(4)

    return feat.fillna(0)


def _generate_synthetic_churn_features(n: int = 7043) -> pd.DataFrame:
    """Pure synthetic fallback — no files needed."""
    rng = np.random.default_rng(42)
    n_churn = int(n * 0.265)
    label = np.zeros(n, dtype=int)
    label[:n_churn] = 1
    rng.shuffle(label)

    tenure = rng.integers(1, 72, size=n)
    monthly = rng.uniform(18, 118, size=n).round(2)

    df = pd.DataFrame({
        "customer_id":            [f"CUST_{i:06d}" for i in range(n)],
        "label":                  label,
        "is_male":                rng.integers(0, 2, size=n),
        "is_senior_citizen":      rng.choice([0, 1], size=n, p=[0.84, 0.16]),
        "has_partner":            rng.integers(0, 2, size=n),
        "has_dependents":         rng.choice([0, 1], size=n, p=[0.70, 0.30]),
        "tenure_months":          tenure,
        "monthly_charges":        monthly,
        "total_charges":          (tenure * monthly).round(2),
        "implied_tenure":         tenure + rng.integers(-2, 3, size=n),
        "is_monthly_contract":    rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        "is_annual_contract":     rng.choice([0, 1], size=n, p=[0.79, 0.21]),
        "is_biannual_contract":   rng.choice([0, 1], size=n, p=[0.83, 0.17]),
        "uses_electronic_check":  rng.choice([0, 1], size=n, p=[0.67, 0.33]),
        "has_paperless_billing":  rng.integers(0, 2, size=n),
        "has_fiber_optic":        rng.choice([0, 1], size=n, p=[0.56, 0.44]),
        "has_dsl":                rng.choice([0, 1], size=n, p=[0.79, 0.21]),
        "no_internet":            rng.choice([0, 1], size=n, p=[0.79, 0.21]),
        "has_online_security":    rng.integers(0, 2, size=n),
        "has_tech_support":       rng.integers(0, 2, size=n),
        "has_phone":              rng.choice([0, 1], size=n, p=[0.10, 0.90]),
        "has_multiple_lines":     rng.integers(0, 2, size=n),
        "has_online_backup":      rng.integers(0, 2, size=n),
        "has_device_protect":     rng.integers(0, 2, size=n),
        "has_streaming_tv":       rng.integers(0, 2, size=n),
        "has_streaming_movies":   rng.integers(0, 2, size=n),
        "is_new_customer":        (tenure < 12).astype(int),
        "is_long_tenure":         (tenure > 48).astype(int),
        "charge_gap":             rng.uniform(-100, 100, size=n).round(2),
        "services_count":         rng.integers(0, 6, size=n),
        "is_high_value":          (monthly > 70).astype(int),
        "is_at_risk":             rng.choice([0, 1], size=n, p=[0.70, 0.30]),
        "total_support_tickets":  rng.integers(0, 8, size=n),
        "negative_support_tickets": rng.integers(0, 4, size=n),
        "tickets_last_6mo":       rng.integers(0, 5, size=n),
        "total_orders_6mo":       rng.integers(0, 20, size=n),
        "total_revenue_6mo":      rng.uniform(0, 800, size=n).round(2),
        "avg_monthly_revenue_6mo": rng.uniform(0, 150, size=n).round(2),
        "revenue_volatility_6mo": rng.uniform(0, 40, size=n).round(2),
        "total_cancellations_6mo": rng.integers(0, 4, size=n),
        "active_months_6mo":      rng.integers(1, 7, size=n),
        "lifetime_value":         rng.lognormal(6.5, 1.0, size=n).round(2),
        "avg_order_value":        rng.uniform(20, 180, size=n).round(2),
        "total_orders":           rng.integers(1, 40, size=n),
        "negative_ticket_ratio":  rng.uniform(0, 1, size=n).round(3),
        "charge_to_ltv_ratio":    rng.uniform(0, 0.5, size=n).round(4),
    })

    # inject realistic churn signal
    churn = df["label"] == 1
    df.loc[churn, "is_monthly_contract"] = 1
    df.loc[churn, "tenure_months"]       = rng.integers(1, 24, size=churn.sum())
    df.loc[churn, "negative_support_tickets"] = rng.integers(2, 6, size=churn.sum())
    df.loc[churn, "has_tech_support"]    = 0

    return df