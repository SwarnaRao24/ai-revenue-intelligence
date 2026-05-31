"""
ml_training/churn/train_churn.py
XGBoost + LightGBM ensemble for customer churn prediction.
Tracks all experiments in MLflow.
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
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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

# ── Config ────────────────────────────────────────────────────
EXPERIMENT_NAME = "churn_prediction"
MODEL_NAME      = "customer_churn_classifier"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

FEATURE_COLS = [
    "is_male", "is_senior_citizen", "has_partner", "has_dependents",
    "tenure_months", "monthly_charges", "total_charges", "implied_tenure",
    "is_monthly_contract", "is_annual_contract", "is_biannual_contract",
    "uses_electronic_check", "has_paperless_billing",
    "has_fiber_optic", "has_dsl", "no_internet",
    "has_online_security", "has_tech_support", "has_phone",
    "has_multiple_lines", "has_online_backup", "has_device_protect",
    "has_streaming_tv", "has_streaming_movies",
    "is_new_customer", "is_long_tenure", "charge_gap",
    "services_count", "is_high_value", "is_at_risk",
    "total_support_tickets", "negative_support_tickets", "tickets_last_6mo",
    "total_orders_6mo", "total_revenue_6mo", "avg_monthly_revenue_6mo",
    "revenue_volatility_6mo", "total_cancellations_6mo", "active_months_6mo",
    "lifetime_value", "avg_order_value", "total_orders",
    "negative_ticket_ratio", "charge_to_ltv_ratio",
]

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbosity":        0,
}

LGB_PARAMS = {
    "n_estimators":   300,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "class_weight":   "balanced",
    "random_state":   RANDOM_STATE,
    "n_jobs":         -1,
    "verbose":        -1,
}


def prepare_features(df: pd.DataFrame):
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].fillna(0).astype(float)
    y = df["label"].astype(int)
    return X, y


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc":       roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1":            f1_score(y_true, y_pred),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
    }


def train():
    logger.info("=" * 55)
    logger.info("  Churn Model Training — XGBoost + LightGBM")
    logger.info("=" * 55)

    # 1. Load features
    df = load_churn_features()
    X, y = prepare_features(df)
    logger.info(f"Dataset: {len(X):,} rows | {len(FEATURE_COLS)} features")
    logger.info(f"Churn rate: {y.mean():.1%}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. MLflow run
    setup_mlflow(EXPERIMENT_NAME)
    run_name = f"churn_ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        # Log params
        log_model_params({
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "churn_rate": round(float(y.mean()), 4),
            "n_features": len(FEATURE_COLS),
            "xgb_n_estimators": XGB_PARAMS["n_estimators"],
            "xgb_max_depth":    XGB_PARAMS["max_depth"],
            "xgb_lr":           XGB_PARAMS["learning_rate"],
            "lgb_n_estimators": LGB_PARAMS["n_estimators"],
        })

        # 4. Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_prob)
        log_model_metrics({f"xgb_{k}": v for k, v in xgb_metrics.items()})

        # 5. Train LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(X_train, y_train)
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
        lgb_pred = (lgb_prob >= 0.5).astype(int)
        lgb_metrics = compute_metrics(y_test, lgb_pred, lgb_prob)
        log_model_metrics({f"lgb_{k}": v for k, v in lgb_metrics.items()})

        # 6. Ensemble
        logger.info("Building ensemble (XGB 60% + LGB 40%)...")
        ensemble = VotingClassifier(
            estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
            voting="soft",
            weights=[0.6, 0.4],
        )
        ensemble.fit(X_train, y_train)
        ens_prob = ensemble.predict_proba(X_test)[:, 1]
        ens_pred = (ens_prob >= 0.5).astype(int)
        ens_metrics = compute_metrics(y_test, ens_pred, ens_prob)
        log_model_metrics(ens_metrics)

        # 7. Report
        logger.info("\n" + classification_report(
            y_test, ens_pred,
            target_names=["Stays", "Churns"]
        ))

        # 8. Feature importance
        fi = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": xgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info(f"\nTop 10 Features:\n{fi.head(10).to_string(index=False)}")

        # 9. Save
        Path("models").mkdir(exist_ok=True)
        joblib.dump(ensemble,     "models/churn_model.pkl")
        joblib.dump(FEATURE_COLS, "models/churn_feature_cols.pkl")
        mlflow.sklearn.log_model(ensemble, "model")

        logger.info(f"\nRun ID : {run_id}")
        logger.info(f"AUC-ROC: {ens_metrics['roc_auc']:.4f}")
        logger.info(f"F1     : {ens_metrics['f1']:.4f}")
        logger.info(f"Recall : {ens_metrics['recall']:.4f}")
        logger.info("Churn model saved to models/churn_model.pkl")

    return ensemble, ens_metrics


if __name__ == "__main__":
    train()