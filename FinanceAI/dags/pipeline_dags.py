"""
Apache Airflow DAGs:
  1. daily_market_data_ingestion  — fetch OHLCV + news + macros
  2. model_retraining_pipeline    — retrain all models weekly
  3. daily_prediction_pipeline    — run predictions for all watchlist symbols
  4. mf_data_refresh              — refresh MF NAVs daily
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner":            "financeai",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
    "email":            ["alerts@financeai.io"],
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
    start_date  = days_ago(1),
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
    """Run ensemble predictions for all watchlist symbols."""
    import asyncio, json
    from config.settings import settings
    from src.ingestion.data_fetcher import YFinanceFetcher, NewsFetcher
    from src.features.technical import TechnicalFeatures
    from src.models.ensemble.predictor import ModelOrchestrator
    from src.sentiment.analyzer import NewsAggregator
    from src.recommendation.engine import StockScreener

    async def _fetch_news():
        nf = NewsFetcher()
        return await nf.fetch_all(settings.watchlist_nse)

    articles  = asyncio.run(_fetch_news())
    agg       = NewsAggregator()
    enriched  = agg.process_articles(articles)
    sent_map  = agg.aggregate_by_symbol(enriched, settings.watchlist_nse)

    yf         = YFinanceFetcher()
    orch       = ModelOrchestrator()
    tf         = TechnicalFeatures()
    all_preds  = []

    for sym in settings.watchlist_nse:
        try:
            ohlcv = yf.fetch_ohlcv(sym, period="2y")
            if ohlcv.empty:
                continue
            fund  = yf.fetch_fundamentals(sym)
            score = sent_map.get(sym.replace(".NS", ""), 0.0)
            pred  = orch.predict(sym, ohlcv, fund, score)
            all_preds.append(pred)
            print(f"  {sym}: {pred['signal']} ({pred['confidence']:.2%})")
        except Exception as exc:
            print(f"  ⚠️  {sym} failed: {exc}")

    # Rank & screen
    screener  = StockScreener()
    top_buys  = screener.screen(all_preds, top_n=10, signal_filter="BUY")
    top_sells = screener.screen(all_preds, top_n=5,  signal_filter="SELL")

    print(f"\n🟢 Top BUY picks:")
    for r in top_buys:
        print(f"  #{r.rank} {r.symbol} | Score: {r.composite_score:.2%} | Target: ₹{r.target_price}")

    print(f"\n🔴 Top SELL picks:")
    for r in top_sells:
        print(f"  #{r.rank} {r.symbol} | Score: {r.composite_score:.2%}")

    # Persist predictions to DB
    import asyncio as _asyncio
    _asyncio.run(_persist_predictions(all_preds))


async def _persist_predictions(predictions):
    from src.database.models import AsyncSessionFactory, Stock, Prediction
    from sqlalchemy import select
    async with AsyncSessionFactory() as db:
        for p in predictions:
            stock_res = await db.execute(select(Stock).where(Stock.symbol == p["symbol"]))
            stock = stock_res.scalar_one_or_none()
            if not stock:
                stock = Stock(symbol=p["symbol"], company_name=p["symbol"])
                db.add(stock)
                await db.flush()
            pred = Prediction(
                stock_id          = stock.id,
                model_version     = "1.0.0",
                signal            = p["signal"],
                confidence        = p["confidence"],
                composite_score   = p["composite_score"],
                technical_score   = p["technical_score"],
                fundamental_score = p["fundamental_score"],
                sentiment_score   = p["sentiment_score"],
                target_price      = p.get("target_price"),
                stop_loss         = p.get("stop_loss"),
                risk_reward_ratio = p.get("risk_reward_ratio"),
                predicted_return_1d = p.get("predicted_return_5d"),
                feature_importance  = p.get("feature_importance"),
            )
            db.add(pred)
        await db.commit()


with DAG(
    dag_id       = "daily_prediction_pipeline",
    description  = "Run ML ensemble predictions for all watchlist stocks",
    schedule     = "0 10 * * 1-5",   # 10 AM IST weekdays
    start_date   = days_ago(1),
    default_args = DEFAULT_ARGS,
    catchup      = False,
    tags         = ["predictions", "ml"],
) as dag2:

    PythonOperator(task_id="run_predictions", python_callable=run_predictions)


# ════════════════════════════════════════════════════════════════════════════
# DAG 3: Weekly Model Retraining
# ════════════════════════════════════════════════════════════════════════════
def retrain_models(**ctx):
    """Retrain LSTM + XGBoost + LightGBM for each symbol."""
    from config.settings import settings
    from src.ingestion.data_fetcher import YFinanceFetcher
    from src.features.technical import TechnicalFeatures, FeaturePipeline
    from src.models.lstm.model import LSTMTrainer
    from src.models.xgboost_model.model import XGBoostModel, LightGBMModel
    from sklearn.model_selection import train_test_split

    yf  = YFinanceFetcher()
    tf  = TechnicalFeatures()
    fp  = FeaturePipeline()

    for sym in settings.watchlist_nse:
        sym_clean = sym.replace(".NS", "")
        print(f"🔄 Retraining models for {sym}…")
        try:
            ohlcv    = yf.fetch_ohlcv(sym, period="5y")
            if len(ohlcv) < settings.min_training_samples:
                print(f"  ⚠️  Not enough data ({len(ohlcv)} rows), skipping")
                continue

            enriched = tf.compute(ohlcv)
            fund     = yf.fetch_fundamentals(sym)

            # LSTM
            X_seq, y_seq, scaler, feats = fp.prepare_lstm_input(enriched, lookback=settings.lstm_lookback)
            split = int(0.8 * len(X_seq))
            lstm_trainer = LSTMTrainer()
            result = lstm_trainer.train(X_seq[:split], y_seq[:split], X_seq[split:], y_seq[split:], sym_clean)
            print(f"  LSTM: {result.get('metrics', {})}")

            # XGBoost
            X_tab, y_tab, _, _ = fp.prepare_xgboost_input(enriched, fund)
            xgb = XGBoostModel()
            result_xgb = xgb.train_with_hpo(X_tab, y_tab, sym_clean, n_trials=30)
            print(f"  XGB:  {result_xgb.get('metrics', {})}")

            # LightGBM
            lgbm = LightGBMModel()
            result_lgbm = lgbm.train(X_tab, y_tab, sym_clean)
            print(f"  LGBM: {result_lgbm.get('metrics', {})}")

        except Exception as exc:
            print(f"  ❌ Retraining failed for {sym}: {exc}")


with DAG(
    dag_id       = "weekly_model_retraining",
    description  = "Retrain all ML models on fresh data",
    schedule     = "0 2 * * 0",   # Sunday 2 AM UTC
    start_date   = days_ago(7),
    default_args = DEFAULT_ARGS,
    catchup      = False,
    tags         = ["training", "ml"],
) as dag3:

    PythonOperator(task_id="retrain_models", python_callable=retrain_models)
