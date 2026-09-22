"""
Train a single, cross-sectional (i.e. not per-symbol) XGBoost + LightGBM
model pair on pooled technical/fundamental features from a broad sample of
NSE stocks.

Why this exists: ModelOrchestrator.predict() (src/models/ensemble/predictor.py)
was written expecting a per-symbol trained model file at
models_store/{xgboost,lgbm,lstm}/{SYMBOL}_*.{pkl,keras} — but nothing in this
codebase ever actually trained and saved one for any symbol. models_store/ was
empty. Every prediction ever produced by this app therefore fell through to
the hardcoded neutral placeholders (lstm_prob = xgb_prob = lgbm_prob = 0.5),
meaning the "AI ensemble" contributed a constant, uninformative 0.5 to every
single stock's score — signals were driven entirely by the rule-based
technical/fundamental scoring on top, not by anything learned from data. This
is the root cause behind predictions looking arbitrary/repetitive.

Training a genuinely separate deep-learning model per NSE symbol (2,300+ of
them) is impractical here (no GPU, would take hours-to-days, and most small/
mid-caps don't have enough history to train an LSTM in isolation anyway).
The standard, proven alternative — and what quant desks actually do for
broad-coverage equity models — is a single *cross-sectional* model: pool
(technical + fundamental features, forward-return label) rows from many
stocks together and train one model that generalizes across symbols. That's
what this script does, for XGBoost and LightGBM (LSTM is left on its neutral
0.5 fallback for now — sequence models don't pool across symbols as cleanly,
and XGBoost+LightGBM already give a real, trained signal instead of none).

ModelOrchestrator.predict() falls back to these "_GLOBAL_*" artifacts for any
symbol that doesn't have its own dedicated per-symbol model file — which, in
practice, is every symbol today. That fallback preserves the option to later
fine-tune a dedicated model for a specific high-volume symbol without any
further code changes.

Usage:
    python -m src.pipelines.train_global_model [--sample-size 500] [--workers 8]
"""
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("train_global_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MIN_OHLCV_ROWS = 60


def _fetch_symbol_features(symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Runs in a worker thread. Returns (X, y) for one symbol, or None on failure."""
    from src.ingestion.data_fetcher import YFinanceFetcher
    from src.features.technical import TechnicalFeatures, FeaturePipeline

    try:
        yf = YFinanceFetcher()
        ohlcv = yf.fetch_ohlcv(symbol, period="5y")
        if ohlcv.empty or len(ohlcv) < MIN_OHLCV_ROWS:
            return None
        fund = yf.fetch_fundamentals(symbol)

        enriched = TechnicalFeatures().compute(ohlcv)
        X, y, _scaler, _features = FeaturePipeline().prepare_xgboost_input(
            enriched, fundamental=fund, sentiment=0.0, macro=None, horizon=5,
        )
        if X.empty:
            return None
        return X, y
    except Exception as exc:
        logger.debug("Skipping %s: %s", symbol, exc)
        return None


async def _build_training_universe(sample_size: int) -> List[str]:
    """
    A stride-sampled slice of the full NSE equity list — spread across the
    whole (roughly alphabetical) universe rather than just the large-caps
    already sitting in the DB, so the model sees small/mid-cap behavior too
    (that's exactly the segment missing from predictions today).
    """
    from src.ingestion.data_fetcher import NSEUniverseFetcher

    equities = await NSEUniverseFetcher().fetch_equity_symbols()
    symbols = [e["yf_symbol"] for e in equities]
    if len(symbols) <= sample_size:
        return symbols
    stride = max(1, len(symbols) // sample_size)
    return symbols[::stride][:sample_size]


async def train(sample_size: int = 500, workers: int = 8) -> dict:
    import asyncio
    from src.models.xgboost_model.model import XGBoostModel, LightGBMModel

    symbols = await _build_training_universe(sample_size)
    logger.info("Training universe: %d symbols (stride-sampled for cap-size diversity)", len(symbols))

    start = time.time()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=workers)
    tasks = [loop.run_in_executor(executor, _fetch_symbol_features, sym) for sym in symbols]

    X_parts, y_parts = [], []
    ok = fail = 0
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        if result is None:
            fail += 1
        else:
            X, y = result
            X_parts.append(X)
            y_parts.append(y)
            ok += 1
        if i % 50 == 0 or i == len(symbols):
            logger.info("Fetched %d/%d (ok=%d, fail=%d, elapsed=%ds)",
                         i, len(symbols), ok, fail, round(time.time() - start))
    executor.shutdown(wait=True)

    if not X_parts:
        raise RuntimeError("No symbols produced usable training data — aborting.")

    X_pool = pd.concat(X_parts, ignore_index=True)
    y_pool = pd.concat(y_parts, ignore_index=True)
    logger.info(
        "Pooled training set: %d rows from %d symbols | positive class: %.1f%%",
        len(X_pool), ok, 100 * y_pool.mean(),
    )

    xgb_result = XGBoostModel().train_fast(X_pool, y_pool, symbol="_GLOBAL")
    logger.info("XGBoost (_GLOBAL): %s", xgb_result.get("metrics") or xgb_result.get("error"))

    lgbm_result = LightGBMModel().train(X_pool, y_pool, symbol="_GLOBAL")
    logger.info("LightGBM (_GLOBAL): %s", lgbm_result.get("metrics") or lgbm_result.get("error"))

    return {
        "symbols_attempted": len(symbols), "symbols_ok": ok, "symbols_failed": fail,
        "training_rows": len(X_pool), "positive_class_pct": round(100 * y_pool.mean(), 2),
        "xgboost": xgb_result, "lightgbm": lgbm_result,
        "elapsed_s": round(time.time() - start),
    }


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    summary = asyncio.run(train(sample_size=args.sample_size, workers=args.workers))
    print(summary)
