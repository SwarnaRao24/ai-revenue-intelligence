"""
ml_training/segmentation/train_segmentation.py
K-Means customer segmentation on RFM + behavioural features.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent.parent))
from ml_training.utils.data_loader import load_churn_features
from ml_training.utils.mlflow_utils import (
    log_model_metrics,
    log_model_params,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "customer_segmentation"

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At Risk",
    3: "Hibernating",
    4: "Lost",
}

RFM_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "lifetime_value", "avg_order_value", "total_orders",
    "active_months_6mo", "total_revenue_6mo",
    "total_support_tickets", "services_count",
]


def train():
    logger.info("=" * 55)
    logger.info("  Customer Segmentation — K-Means Clustering")
    logger.info("=" * 55)

    # 1. Load
    df = load_churn_features()
    features = [f for f in RFM_FEATURES if f in df.columns]
    X_raw = df[features].fillna(0)

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Find best k via silhouette
    setup_mlflow(EXPERIMENT_NAME)
    run_name = f"kmeans_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):

        logger.info("Testing k=3 to k=7...")
        best_k, best_score = 5, -1
        scores = {}

        for k in range(3, 8):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_tmp = km_tmp.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels_tmp)
            scores[k] = round(score, 4)
            logger.info(f"  k={k} → silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k     = k

        logger.info(f"Best k={best_k} (silhouette={best_score:.4f})")

        # 4. Train final model
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        model.fit(X_scaled)
        labels = model.labels_
        final_score = silhouette_score(X_scaled, labels)

        log_model_params({
            "n_clusters": best_k,
            "features":   str(features),
            "n_samples":  len(df),
        })
        log_model_metrics({
            "silhouette_score": final_score,
            "n_clusters":       best_k,
        })

        # 5. Segment profile
        df["segment_id"]   = labels
        df["segment_name"] = df["segment_id"].map(
            {i: SEGMENT_LABELS.get(i, f"Segment_{i}") for i in range(best_k)}
        )
        profile = (
            df.groupby("segment_name")[features]
            .mean()
            .round(1)
        )
        logger.info(f"\nSegment Profiles:\n{profile.to_string()}")

        counts = df["segment_name"].value_counts()
        logger.info(f"\nSegment Counts:\n{counts.to_string()}")

        # 6. Save
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model,    "models/segmentation_model.pkl")
        joblib.dump(scaler,   "models/segmentation_scaler.pkl")
        joblib.dump(features, "models/segmentation_feature_cols.pkl")
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"\nSilhouette Score : {final_score:.4f}")
        logger.info("Segmentation model saved to models/segmentation_model.pkl")

    return model


if __name__ == "__main__":
    train()