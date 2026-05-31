"""
ml_training/revenue/train_revenue.py
XGBoost regressor for monthly revenue forecasting.
Uses lag + rolling features as time series inputs.
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
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent))
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

EXPERIMENT_NAME = "revenue_forecasting"


def generate_revenue_data(n_months: int = 54) -> pd.DataFrame:
    """Synthetic monthly revenue with trend + seasonality."""
    rng = np.random.default_rng(42)
    dates   = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    trend   = np.linspace(50_000, 130_000, n_months)
    season  = 9_000 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    noise   = rng.normal(0, 3_000, n_months)
    revenue = (trend + season + noise).clip(min=0).round(2)
    return pd.DataFrame({"ds": dates, "y": revenue})


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features."""
    df = df.copy().sort_values("ds").reset_index(drop=True)

    df["month"]   = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["year"]    = df["ds"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in [3, 6]:
        df[f"rolling_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"]  = df["y"].shift(1).rolling(window).std()
        df[f"rolling_max_{window}"]  = df["y"].shift(1).rolling(window).max()

    return df.dropna().reset_index(drop=True)


def train():
    logger.info("=" * 55)
    logger.info("  Revenue Forecasting Model — XGBoost Regressor")
    logger.info("=" * 55)

    # 1. Data
    df_raw  = generate_revenue_data(54)
    df_feat = build_time_features(df_raw)
    feature_cols = [c for c in df_feat.columns if c not in ["ds", "y"]]

    X = df_feat[feature_cols]
    y = df_feat["y"]

    # Time-based split (no shuffle — respects time order)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info(f"Train months: {split} | Test months: {len(X_test)}")

    # 2. MLflow
    setup_mlflow(EXPERIMENT_NAME)
    run_name = f"revenue_xgb_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        params = {
            "n_estimators":  200,
            "max_depth":     4,
            "learning_rate": 0.05,
            "subsample":     0.8,
            "random_state":  42,
            "n_jobs":        -1,
            "verbosity":     0,
        }
        log_model_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        mape   = mean_absolute_percentage_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        mae    = np.mean(np.abs(y_test - y_pred))

        log_model_metrics({"mape": mape, "rmse": rmse, "mae": mae})

        # Show predictions vs actuals
        comparison = pd.DataFrame({
            "actual":    y_test.values.round(0),
            "predicted": y_pred.round(0),
            "error_pct": ((y_pred - y_test.values) / y_test.values * 100).round(1),
        })
        logger.info(f"\nPredictions vs Actuals:\n{comparison.to_string(index=False)}")

        # Save
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model,        "models/revenue_model.pkl")
        joblib.dump(feature_cols, "models/revenue_feature_cols.pkl")
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"\nMAPE : {mape:.2%}")
        logger.info(f"RMSE : ${rmse:,.0f}")
        logger.info(f"MAE  : ${mae:,.0f}")
        logger.info("Revenue model saved to models/revenue_model.pkl")

    return model


if __name__ == "__main__":
    train()