"""
airflow_dags/ingestion_dag.py
Daily data ingestion DAG — Bronze layer pipeline.
Runs at midnight, ingests Telco + E-Commerce data.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml_platform",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="data_ingestion_pipeline",
    default_args=default_args,
    description="Daily Bronze layer ingestion — Telco + E-Commerce",
    schedule_interval="0 0 * * *",  # midnight daily
    catchup=False,
    tags=["ingestion", "bronze", "daily"],
)


def ingest_telco():
    """Download and save IBM Telco dataset to Bronze layer."""
    import sys
    from pathlib import Path

    sys.path.append(str(Path("/opt/airflow")))
    from data_ingestion.ingest_telco import run

    run()


def ingest_ecommerce():
    """Generate and save synthetic e-commerce data to Bronze layer."""
    import sys
    from pathlib import Path

    sys.path.append(str(Path("/opt/airflow")))
    from data_ingestion.ingest_ecommerce import run

    run()


def validate_bronze():
    """Check Bronze layer files exist and have expected row counts."""
    from pathlib import Path

    import pandas as pd

    bronze_dir = Path("/opt/airflow/data/bronze")
    required = {
        "bronze_telco_customers.parquet": 7000,
        "bronze_ecommerce_orders.parquet": 10000,
        "bronze_support_tickets.parquet": 2000,
    }
    for filename, min_rows in required.items():
        path = bronze_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Bronze file: {filename}")
        df = pd.read_parquet(path)
        if len(df) < min_rows:
            raise ValueError(
                f"{filename} has {len(df)} rows — expected at least {min_rows}"
            )
    print("Bronze layer validation passed!")


# ── Tasks ─────────────────────────────────────────────────────
t_start = BashOperator(
    task_id="pipeline_start",
    bash_command='echo "Starting ingestion pipeline $(date)"',
    dag=dag,
)

t_telco = PythonOperator(
    task_id="ingest_telco",
    python_callable=ingest_telco,
    dag=dag,
)

t_ecommerce = PythonOperator(
    task_id="ingest_ecommerce",
    python_callable=ingest_ecommerce,
    dag=dag,
)

t_validate = PythonOperator(
    task_id="validate_bronze",
    python_callable=validate_bronze,
    dag=dag,
)

t_end = BashOperator(
    task_id="pipeline_complete",
    bash_command='echo "Ingestion complete $(date)"',
    dag=dag,
)

# ── Dependencies ──────────────────────────────────────────────
t_start >> [t_telco, t_ecommerce] >> t_validate >> t_end
