"""
Apache Airflow DAGs:
  1. daily_market_data_ingestion  — fetch OHLCV + news + macros
  2. model_retraining_pipeline    — retrain all models weekly
  3. daily_prediction_pipeline    — run predictions for all watchlist symbols
  4. mf_data_refresh              — refresh MF NAVs daily
"""
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


# ════════════════════════════════════════════════════════════════════════════
# DAG 1: Daily Market Data Ingestion
# ════════════════════════════════════════════════════════════════════════════
def ingest_ohlcv(**ctx):
    from src.ingestion.data_fetcher import YFinanceFetcher
    from config.settings import settings
    import pandas as pd
    from pathlib import Path

    fetcher = YFinanceFetcher()
    data    = fetcher.fetch_multiple(settings.watchlist_nse, period="5d", interval="1d")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    for sym, df in data.items():
        df.to_parquet(f"data/raw/{sym.replace('.', '_')}_ohlcv.parquet")
    print(f"✅ Ingested OHLCV for {len(data)} symbols")


def ingest_news(**ctx):
    import asyncio
    from src.ingestion.data_fetcher import NewsFetcher
    from src.sentiment.analyzer import NewsAggregator
    from config.settings import settings
    import json, time
    from pathlib import Path

    async def _run():
        fetcher = NewsFetcher()
        articles = await fetcher.fetch_all(settings.watchlist_nse)
        agg  = NewsAggregator()
        enriched = agg.process_articles(articles)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        fname = f"data/raw/news_{int(time.time())}.json"
        with open(fname, "w") as f:
            json.dump(enriched, f, default=str)
        return len(enriched)

    count = asyncio.run(_run())
    print(f"✅ Ingested {count} news articles")


def ingest_macro(**ctx):
    from src.ingestion.data_fetcher import MacroFetcher
    macro = MacroFetcher()
    data  = macro.fetch_all(period="5d")
    print(f"✅ Ingested macro data for {len(data)} indicators")


with DAG(
    dag_id      = "daily_market_data_ingestion",
    description = "Fetch OHLCV, news, and macro data",
    schedule    = "30 9 * * 1-5",   # 9:30 AM IST weekdays (4:00 UTC)
    start_date  = START_DATE,
    default_args= DEFAULT_ARGS,
    catchup     = False,
    tags        = ["ingestion", "market-data"],
) as dag1:

    t_ohlcv = PythonOperator(task_id="ingest_ohlcv", python_callable=ingest_ohlcv)
    t_news  = PythonOperator(task_id="ingest_news",  python_callable=ingest_news)
    t_macro = PythonOperator(task_id="ingest_macro", python_callable=ingest_macro)

    [t_ohlcv, t_news, t_macro]  # Parallel


# ════════════════════════════════════════════════════════════════════════════
# DAG 2: Daily Prediction Pipeline
# ════════════════════════════════════════════════════════════════════════════
def run_predictions(**ctx):
    """
    Run ensemble predictions for the FULL NSE universe (~2,300 equities +
    F&O + index F&O) — not just settings.watchlist_nse (20 hardcoded
    large-caps). That 20-symbol list is what this task used to scan, which
    is why the vast majority of NSE-listed stocks (small/mid-caps included —
    e.g. Optiemus Infracom) never showed up anywhere in the app, and why
    predictions looked repetitive day to day: the same tiny, static set was
    the only thing ever refreshed on a schedule.

    src.pipelines.universe_scan.run_universe_scan() is the real, full-
    coverage pipeline (also used by POST /api/v1/predictions/run-universe
    and scripts/run_full_universe_pipeline.py) — it fetches the live NSE
    equity + F&O + index lists itself, so this task doesn't need (and
    shouldn't hardcode) a symbol list at all.
    """
    import asyncio
    from src.pipelines.universe_scan import run_universe_scan

    def _log_progress(event: dict):
        print(f"  {event}")

    summary = asyncio.run(run_universe_scan(on_progress=_log_progress))
    print(f"\n✅ Universe scan complete: {summary}")


with DAG(
    dag_id       = "daily_prediction_pipeline",
    description  = "Run ML ensemble predictions for the full NSE universe (equities + F&O + indices)",
    schedule     = "0 10 * * 1-5",   # 10 AM IST weekdays
    start_date   = START_DATE,
    default_args = DEFAULT_ARGS,
    catchup      = False,
    tags         = ["predictions", "ml"],
) as dag2:

    PythonOperator(task_id="run_predictions", python_callable=run_predictions)


# ════════════════════════════════════════════════════════════════════════════
# DAG 3: Weekly Model Retraining
# ════════════════════════════════════════════════════════════════════════════
def retrain_models(**ctx):
    """
    Retrain the shared cross-sectional XGBoost + LightGBM model (see
    src/pipelines/train_global_model.py) on a fresh, broad sample of NSE
    stocks — this is the model every symbol's prediction actually uses
    (ModelOrchestrator falls back to it whenever no per-symbol model file
    exists, which today is every symbol; see src/models/ensemble/predictor.py).

    This DAG used to loop settings.watchlist_nse (20 hardcoded large-caps)
    training one LSTM+XGBoost+LightGBM per symbol — impractical to extend to
    the full ~2,300-symbol universe, and in practice models_store/ never
    ended up with anything in it. A single cross-sectional model that pools
    data across many symbols is the standard approach for broad-coverage
    equity models and is what actually gets used at inference time now.
    """
    import asyncio
    from src.pipelines.train_global_model import train

    summary = asyncio.run(train(sample_size=500, workers=8))
    print(f"✅ Global model retrained: {summary}")


with DAG(
    dag_id       = "weekly_model_retraining",
    description  = "Retrain all ML models on fresh data",
    schedule     = "0 2 * * 0",   # Sunday 2 AM UTC
    start_date   = START_DATE,
    default_args = DEFAULT_ARGS,
    catchup      = False,
    tags         = ["training", "ml"],
) as dag3:

    PythonOperator(task_id="retrain_models", python_callable=retrain_models)
