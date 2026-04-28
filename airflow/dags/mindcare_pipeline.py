"""
Airflow DAG: MindCare Oracle Data + Training Pipeline
Orchestrates:
  1. preprocess_data  — runs src/data_preprocessing.py
  2. train_model      — runs src/train.py (logs to MLflow)
Trigger manually via the Airflow UI (http://localhost:8080) or on a schedule.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mindcare",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="mindcare_pipeline",
    description="End-to-end MindCare data preprocessing and model training pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,          # trigger manually or change to "@daily"
    catchup=False,
    tags=["mindcare", "mlops"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=(
            "cd /opt/airflow && "
            "python src/data_preprocessing.py"
        ),
        doc_md="""
## Preprocess Data
Reads `data/survey.csv`, cleans/normalises fields, writes
`data/processed/survey_cleaned.csv`. Implements data validation
and feature engineering as per MLOps guidelines.
        """,
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=(
            "cd /opt/airflow && "
            "python src/train.py "
            "--data data/processed/survey_cleaned.csv "
            "--n_estimators 100 "
            "--max_depth 2"
        ),
        doc_md="""
## Train Model
Trains an AdaBoost classifier on the processed survey data.
Logs parameters, metrics (accuracy, ROC-AUC, F1), and the
serialised sklearn Pipeline to MLflow experiment
`mindcare-oracle-treatment`. Also saves baseline_stats.json
for downstream drift detection.
        """,
    )

    # Pipeline dependency: preprocess must complete before training
    preprocess >> train
