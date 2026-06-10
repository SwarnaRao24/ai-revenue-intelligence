"""
airflow_dags/training_dag.py
Weekly ML retraining DAG.
Runs every Sunday at 2am — retrains all 4 models.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner":            "ml_platform",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}

dag = DAG(
    dag_id="ml_retraining_pipeline",
    default_args=default_args,
    description="Weekly ML model retraining — all 4 models",
    schedule_interval="0 2 * * 0",   # 2am every Sunday
    catchup=False,
    tags=["training", "ml", "weekly"],
)


def _import_and_train(module_path: str, fn_name: str = "train"):
    """Dynamically import and run a training function."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path("/opt/airflow")))
    import importlib
    mod = importlib.import_module(module_path)
    getattr(mod, fn_name)()


def train_churn():
    _import_and_train("ml_training.churn.train_churn")


def train_revenue():
    _import_and_train("ml_training.revenue.train_revenue")


def train_segmentation():
    _import_and_train("ml_training.segmentation.train_segmentation")


def train_anomaly():
    _import_and_train("ml_training.anomaly.train_anomaly")


def validate_models():
    """Confirm all model files were saved after retraining."""
    from pathlib import Path
    import joblib

    required = [
        "models/churn_model.pkl",
        "models/revenue_model.pkl",
        "models/segmentation_model.pkl",
        "models/anomaly_model.pkl",
    ]
    for path in required:
        p = Path(f"/opt/airflow/{path}")
        if not p.exists():
            raise FileNotFoundError(f"Model missing after training: {path}")
        obj = joblib.load(p)
        print(f"  Validated: {path} ({type(obj).__name__})")
    print("All models validated successfully!")


def notify_complete(**context):
    """Log training summary."""
    run_date = context["ds"]
    print(f"ML retraining pipeline completed successfully for {run_date}")
    print("Models retrained: churn, revenue, segmentation, anomaly")


# ── Tasks ─────────────────────────────────────────────────────
t_start = BashOperator(
    task_id="retraining_start",
    bash_command='echo "Starting retraining pipeline $(date)"',
    dag=dag,
)

t_churn = PythonOperator(
    task_id="train_churn_model",
    python_callable=train_churn,
    dag=dag,
    execution_timeout=timedelta(minutes=30),
)

t_revenue = PythonOperator(
    task_id="train_revenue_model",
    python_callable=train_revenue,
    dag=dag,
    execution_timeout=timedelta(minutes=20),
)

t_segment = PythonOperator(
    task_id="train_segmentation_model",
    python_callable=train_segmentation,
    dag=dag,
    execution_timeout=timedelta(minutes=15),
)

t_anomaly = PythonOperator(
    task_id="train_anomaly_model",
    python_callable=train_anomaly,
    dag=dag,
    execution_timeout=timedelta(minutes=15),
)

t_validate = PythonOperator(
    task_id="validate_models",
    python_callable=validate_models,
    dag=dag,
)

t_notify = PythonOperator(
    task_id="notify_complete",
    python_callable=notify_complete,
    provide_context=True,
    dag=dag,
)

# ── Dependencies ──────────────────────────────────────────────
t_start >> [t_churn, t_revenue, t_segment, t_anomaly] >> t_validate >> t_notify