"""
ml_training/anomaly/train_anomaly.py
Isolation Forest for detecting anomalous customer behaviour.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent.parent))
from ml_training.utils.data_loader import load_churn_features
from ml_training.utils.mlflow_utils import (
    log_model_metrics,
    log_model_params,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "anomaly_detection"

ANOMALY_FEATURES = [
    "monthly_charges",
    "total_charges",
    "total_support_tickets",
    "negative_support_tickets",
    "total_cancellations_6mo",
    "revenue_volatility_6mo",
    "charge_gap",
    "tickets_last_6mo",
    "negative_ticket_ratio",
]


def train():
    logger.info("=" * 55)
    logger.info("  Anomaly Detection — Isolation Forest")
    logger.info("=" * 55)

    # 1. Load
    df = load_churn_features()
    features = [f for f in ANOMALY_FEATURES if f in df.columns]
    X_raw = df[features].fillna(0)

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. MLflow
    setup_mlflow(EXPERIMENT_NAME)
    run_name = f"isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        params = {
            "n_estimators": 100,
            "contamination": 0.05,
            "random_state": 42,
            "n_jobs": -1,
        }
        log_model_params(params)

        model = IsolationForest(**params)
        model.fit(X_scaled)

        scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
        n_anomalies = int((predictions == -1).sum())
        anomaly_rate = n_anomalies / len(predictions)

        log_model_metrics(
            {
                "n_anomalies": n_anomalies,
                "anomaly_rate": anomaly_rate,
                "n_samples": len(df),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
            }
        )

        # Show top anomalies
        df["anomaly_score"] = scores
        df["is_anomaly"] = (predictions == -1).astype(int)
        top_anomalies = df[df["is_anomaly"] == 1].nsmallest(10, "anomaly_score")[
            [
                "customer_id",
                "anomaly_score",
                "monthly_charges",
                "total_support_tickets",
                "negative_support_tickets",
            ]
        ]
        logger.info(
            f"\nTop 10 Anomalous Customers:\n{top_anomalies.to_string(index=False)}"
        )

        # Save
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/anomaly_model.pkl")
        joblib.dump(scaler, "models/anomaly_scaler.pkl")
        joblib.dump(features, "models/anomaly_feature_cols.pkl")
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"\nAnomalies detected : {n_anomalies:,} ({anomaly_rate:.1%})")
        logger.info("Anomaly model saved to models/anomaly_model.pkl")

    return model


if __name__ == "__main__":
    train()
