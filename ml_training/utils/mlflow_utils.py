"""
ml_training/utils/mlflow_utils.py
MLflow helpers — works on Windows using SQLite backend.
"""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str) -> str:
    """
    Configure MLflow tracking URI and set/create experiment.
    Uses SQLite locally on Windows (avoids file:// URI issues).
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Try remote server first
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow | experiment='{experiment_name}' "
            f"id={experiment.experiment_id} | {tracking_uri}"
        )
        return experiment.experiment_id

    except Exception as e:
        logger.warning(f"MLflow server unreachable — using local SQLite store")

    # Fall back to SQLite (works perfectly on Windows)
    db_path = Path("mlflow.db").absolute()
    sqlite_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(sqlite_uri)

    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLflow | experiment='{experiment_name}' "
        f"id={experiment.experiment_id} | {sqlite_uri}"
    )
    return experiment.experiment_id


def log_model_metrics(metrics: dict[str, float]) -> None:
    """Log all metrics to the active MLflow run."""
    for key, value in metrics.items():
        try:
            mlflow.log_metric(key, float(value))
        except Exception:
            pass
    logger.info("Metrics: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def log_model_params(params: dict[str, Any]) -> None:
    """Log all params to the active MLflow run."""
    for key, value in params.items():
        try:
            mlflow.log_param(key, str(value))
        except Exception:
            pass
