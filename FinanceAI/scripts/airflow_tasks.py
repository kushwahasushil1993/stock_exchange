"""
The actual work behind every Airflow DAG task in dags/pipeline_dags.py.

Lives here, not inline in the DAG file, because it has to run under
/opt/finai-venv (docker/Dockerfile.airflow) — a separate virtualenv from
Airflow's own interpreter, built with this project's requirements.txt.
Airflow 2.10's bundled environment pins sqlalchemy==1.4.x and doesn't have
xgboost/tensorflow/yfinance/etc. at all, so none of this can be `import`ed
directly into Airflow's own process — dags/pipeline_dags.py instead runs
`python scripts/airflow_tasks.py <task_name>` as a subprocess against that
venv's python, and this file is that subprocess's entire job.

Usage: python scripts/airflow_tasks.py <ingest_ohlcv|ingest_news|ingest_macro|run_predictions|notify_whatsapp|retrain_models>
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ingest_ohlcv():
    from src.ingestion.data_fetcher import YFinanceFetcher
    from config.settings import settings

    fetcher = YFinanceFetcher()
    data = fetcher.fetch_multiple(settings.watchlist_nse, period="5d", interval="1d")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    for sym, df in data.items():
        df.to_parquet(f"data/raw/{sym.replace('.', '_')}_ohlcv.parquet")
    print(f"✅ Ingested OHLCV for {len(data)} symbols")


def ingest_news():
    from src.ingestion.data_fetcher import NewsFetcher
    from src.sentiment.analyzer import NewsAggregator
    from config.settings import settings

    async def _run():
        fetcher = NewsFetcher()
        articles = await fetcher.fetch_all(settings.watchlist_nse)
        agg = NewsAggregator()
        enriched = agg.process_articles(articles)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        fname = f"data/raw/news_{int(time.time())}.json"
        with open(fname, "w") as f:
            json.dump(enriched, f, default=str)
        return len(enriched)

    count = asyncio.run(_run())
    print(f"✅ Ingested {count} news articles")


def ingest_macro():
    from src.ingestion.data_fetcher import MacroFetcher

    macro = MacroFetcher()
    data = macro.fetch_all(period="5d")
    print(f"✅ Ingested macro data for {len(data)} indicators")


def track_outcomes():
    """
    Evaluates every prediction whose target trading session has already
    happened against what actually occurred (src/pipelines/outcome_tracker.py)
    — run BEFORE run_predictions in the daily DAG, since today's just-
    refreshed price_history is exactly what's needed to score yesterday's
    predictions (whose target_trading_date is today).
    """
    from src.pipelines.outcome_tracker import track_outcomes as _track

    summary = asyncio.run(_track())
    print(f"✅ Outcome tracking: {summary}")


def run_predictions():
    """
    Full NSE universe (~2,300 equities + F&O + index F&O) — not just
    settings.watchlist_nse (20 hardcoded large-caps). That 20-symbol list is
    what this task used to scan, which is why the vast majority of
    NSE-listed stocks (small/mid-caps included — e.g. Optiemus Infracom)
    never showed up anywhere in the app, and why predictions looked
    repetitive day to day: the same tiny, static set was the only thing
    ever refreshed on a schedule.
    """
    from src.pipelines.universe_scan import run_universe_scan

    def _log_progress(event: dict):
        print(f"  {event}")

    summary = asyncio.run(run_universe_scan(on_progress=_log_progress))
    print(f"\n✅ Universe scan complete: {summary}")


def notify_whatsapp():
    """
    Sends today's top BUY/SELL picks over WhatsApp (src/notifications/
    whatsapp.py) once run_predictions has fresh predictions in the DB.
    Skips (exit 0) rather than failing if WhatsApp isn't configured — an
    unconfigured integration shouldn't fail the DAG run.
    """
    from config.settings import settings
    from src.notifications.whatsapp import send_daily_predictions

    missing = [
        name for name, val in (
            ("WHATSAPP_ACCESS_TOKEN", settings.whatsapp_access_token),
            ("WHATSAPP_PHONE_NUMBER_ID", settings.whatsapp_phone_number_id),
            ("WHATSAPP_RECIPIENT_NUMBERS", settings.whatsapp_recipient_numbers),
        ) if not val
    ]
    if missing:
        print(f"⏭️  WhatsApp not fully configured — missing {', '.join(missing)} — skipping notification.")
        return

    try:
        result = asyncio.run(send_daily_predictions())
        print(f"📲 WhatsApp daily predictions: {result}")
    except Exception as exc:
        # Logged, not raised: a notification failure shouldn't be treated as
        # the whole daily pipeline failing — today's predictions are already
        # persisted by run_predictions regardless of whether the message
        # went out.
        print(f"⚠️  WhatsApp notification failed: {exc}")


def retrain_models():
    """
    Retrains the shared cross-sectional XGBoost + LightGBM model (see
    src/pipelines/train_global_model.py) on a fresh, broad sample of NSE
    stocks — this is the model every symbol's prediction actually uses
    (ModelOrchestrator falls back to it whenever no per-symbol model file
    exists, which today is every symbol; see src/models/ensemble/predictor.py).
    """
    from src.pipelines.train_global_model import train

    summary = asyncio.run(train(sample_size=500, workers=8))
    print(f"✅ Global model retrained: {summary}")


def retrain_three_models():
    """
    Retrains the walk-forward-validated three-model split (Direction +
    Expected Return; Risk/Liquidity is rule-based, not trained) — see
    src/pipelines/train_three_models.py. Writes a new (inactive)
    ModelRegistry row; promotion to production is a separate, explicit step
    (champion/challenger, §70), never automatic on a training score alone.
    """
    from src.pipelines.train_three_models import train

    summary = asyncio.run(train(sample_size=500, workers=8))
    print(f"✅ Three-model retrain complete: {summary}")


TASKS = {
    "ingest_ohlcv": ingest_ohlcv,
    "ingest_news": ingest_news,
    "ingest_macro": ingest_macro,
    "track_outcomes": track_outcomes,
    "run_predictions": run_predictions,
    "notify_whatsapp": notify_whatsapp,
    "retrain_models": retrain_models,
    "retrain_three_models": retrain_three_models,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(TASKS)}>")
        sys.exit(2)
    TASKS[sys.argv[1]]()
