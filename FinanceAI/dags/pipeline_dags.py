"""
Apache Airflow DAGs:
  1. daily_market_data_ingestion  — fetch OHLCV + news + macros
  2. daily_prediction_pipeline    — full-universe predictions, then WhatsApp notify
  3. weekly_model_retraining      — retrain the shared cross-sectional model

Every task here is a thin subprocess wrapper — see scripts/airflow_tasks.py
for what each one actually does. They can't just `import src.pipelines...`
directly into Airflow's own interpreter: Airflow 2.10's bundled environment
pins sqlalchemy==1.4.x (this project's src/database/models.py uses
SQLAlchemy 2.0's async ORM API, which doesn't exist in 1.4) and is missing
xgboost/tensorflow/yfinance/etc. entirely. docker/Dockerfile.airflow builds
a separate venv at /opt/finai-venv with this project's actual
requirements.txt installed, and every task below runs against THAT
interpreter as a subprocess instead.
"""
import subprocess
from datetime import timedelta

import pendulum

try:  # Airflow 3.x
    from airflow.sdk import DAG
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow import DAG
    from airflow.operators.python import PythonOperator

# Static start_date (Airflow best practice; replaces the removed
# airflow.utils.dates.days_ago helper).
START_DATE = pendulum.datetime(2024, 1, 1, tz="UTC")

DEFAULT_ARGS = {
    "owner":            "financeai",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    # Task-level `email` / `email_on_failure` are deprecated (removed in
    # Airflow 4). For failure alerts, attach an on_failure_callback, e.g.
    # airflow.providers.smtp.notifications.smtp.SmtpNotifier.
}

FINAI_VENV_PYTHON = "/opt/finai-venv/bin/python"
AIRFLOW_TASKS_SCRIPT = "/app/scripts/airflow_tasks.py"


def _run_finai_task(task_name: str) -> None:
    """
    Runs one scripts/airflow_tasks.py task against /opt/finai-venv, streaming
    its stdout/stderr straight into this Airflow task's own log and raising
    (failing the task, triggering DEFAULT_ARGS' retry) on a non-zero exit.
    """
    subprocess.run(
        [FINAI_VENV_PYTHON, AIRFLOW_TASKS_SCRIPT, task_name],
        cwd="/app",
        check=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# DAG 1: Daily Market Data Ingestion
# ════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id      = "daily_market_data_ingestion",
    description = "Fetch OHLCV, news, and macro data",
    schedule    = "30 9 * * 1-5",   # 9:30 AM IST weekdays (4:00 UTC)
    start_date  = START_DATE,
    default_args= DEFAULT_ARGS,
    catchup     = False,
    tags        = ["ingestion", "market-data"],
) as dag1:

    t_ohlcv = PythonOperator(task_id="ingest_ohlcv", python_callable=lambda **ctx: _run_finai_task("ingest_ohlcv"))
    t_news  = PythonOperator(task_id="ingest_news",  python_callable=lambda **ctx: _run_finai_task("ingest_news"))
    t_macro = PythonOperator(task_id="ingest_macro", python_callable=lambda **ctx: _run_finai_task("ingest_macro"))

    [t_ohlcv, t_news, t_macro]  # Parallel


# ════════════════════════════════════════════════════════════════════════════
# DAG 2: Daily Prediction Pipeline
# ════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id       = "daily_prediction_pipeline",
    description  = "Run ML ensemble predictions for the full NSE universe (equities + F&O + indices), then notify via WhatsApp",
    schedule     = "0 10 * * 1-5",   # 10 AM IST weekdays
    start_date   = START_DATE,
    default_args = DEFAULT_ARGS,
    catchup      = False,
    tags         = ["predictions", "ml"],
) as dag2:

    t_predict = PythonOperator(task_id="run_predictions", python_callable=lambda **ctx: _run_finai_task("run_predictions"))
    t_notify  = PythonOperator(task_id="notify_whatsapp", python_callable=lambda **ctx: _run_finai_task("notify_whatsapp"))
    t_predict >> t_notify


# ════════════════════════════════════════════════════════════════════════════
# DAG 3: Weekly Model Retraining
# ════════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id       = "weekly_model_retraining",
    description  = "Retrain the shared cross-sectional XGBoost + LightGBM model on fresh data",
    schedule     = "0 2 * * 0",   # Sunday 2 AM UTC
    start_date   = START_DATE,
    default_args = DEFAULT_ARGS,
    catchup      = False,
    tags         = ["training", "ml"],
) as dag3:

    PythonOperator(task_id="retrain_models", python_callable=lambda **ctx: _run_finai_task("retrain_models"))
