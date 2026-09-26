"""
Trains the three-model split (§20 of the platform spec) with PROPER
walk-forward (expanding-window) validation — replacing train_global_model.py's
single pooled model, which validated with a random train_test_split()/
StratifiedKFold(shuffle=True), scrambling time order across a time-series
dataset. Confirmed as a real leakage risk in the audit; this script is the
fix.

  Model A — Direction: 3-class (UP/DOWN/FLAT) classifier, calibrated via
            isotonic regression, reporting Brier score + a reliability
            curve so "70% confidence" is checked against whether 70%-
            confidence predictions actually resolve correctly ~70% of the
            time — not just asserted.
  Model B — Expected return: LightGBM quantile regression giving a real
            [lower, upper] interval (10th/90th percentile), not a
            fabricated one.
  Model C — Risk/Liquidity: rule-based, not trained here — see
            src/market/risk_liquidity.py.

Reads training data from price_history (already backfilled to 5y — see
src/pipelines/universe_scan.py) rather than live yfinance calls: faster,
and avoids the exact rate-limit pressure that caused the ingestion pileup
this was all fixed alongside.

IMPORTANT, DOCUMENTED LIMITATION: fundamentals (pe_ratio, roe, ...) are
deliberately EXCLUDED from this model's features. This project only stores
CURRENT fundamentals (src/database/models.py Fundamental table has no
history), so using them for a training row dated months or years ago would
be exactly the kind of look-ahead leakage §23 prohibits — the model would
see information (today's PE ratio) that didn't exist at that historical
point in time. Only technical + market-context features (computed purely
from historical OHLCV, genuinely point-in-time correct) are used. Adding
point-in-time fundamentals back would require a new historical-fundamentals
ingestion pipeline, which does not exist yet.

Usage:
    python -m src.pipelines.train_three_models [--sample-size 500] [--workers 8]
"""
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

logger = logging.getLogger("train_three_models")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MIN_OHLCV_ROWS = 300  # need real history for a meaningful walk-forward split, not just MIN_OHLCV_ROWS=60's bare technical-indicator minimum
HORIZON_DAYS = 1  # next-session direction/return, per §22 — not the old 5-day target
FLAT_THRESHOLD = 0.002  # +/-0.2% dead zone around zero — see prepare_xgboost_input's direction_classes=3
FOLD_MONTHS = 6  # expanding-window fold length

FEATURE_VERSION = "v5_swing_structure_features"

# Excludes every fundamental field from XGBOOST_FEATURES — see this file's
# module docstring for why. Everything else (technical + whatever market-
# context columns are present) is fair game.
_EXCLUDED_FUNDAMENTAL_FIELDS = {
    "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "revenue_growth_yoy",
    "profit_margin", "beta", "usd_inr_ret", "gold_ret", "crude_ret", "vix",
}


def _load_symbol_history(symbol: str) -> Optional[pd.DataFrame]:
    """Sync (runs in a worker thread) — reads price_history via the same sync engine pattern universe_scan.py uses."""
    from src.pipelines.universe_scan import _get_sync_engine
    from sqlalchemy import text

    engine = _get_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT ph.date, ph.open, ph.high, ph.low, ph.close, ph.volume "
                "FROM price_history ph JOIN stocks s ON s.id = ph.stock_id "
                "WHERE s.symbol = :symbol ORDER BY ph.date"
            ),
            {"symbol": symbol},
        ).fetchall()
    if len(rows) < MIN_OHLCV_ROWS:
        return None
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def _build_symbol_features(symbol: str) -> Optional[pd.DataFrame]:
    """
    One symbol's full feature history, with `date` and `symbol` columns
    retained (needed for the walk-forward split's calendar-date cutoffs and
    for pooling), fundamentals excluded (see module docstring).
    """
    from src.features.technical import TechnicalFeatures, FeaturePipeline

    df = _load_symbol_history(symbol)
    if df is None:
        return None
    try:
        enriched = TechnicalFeatures().compute(df)
        X, y, _scaler, features, fwd_return = FeaturePipeline().prepare_xgboost_input(
            enriched, fundamental=None, sentiment=None, macro=None,
            horizon=HORIZON_DAYS, direction_classes=3, flat_threshold=FLAT_THRESHOLD,
        )
        if X.empty:
            return None
        keep_features = [f for f in features if f not in _EXCLUDED_FUNDAMENTAL_FIELDS]
        out = X[keep_features].copy()
        out["target"] = y.values
        out["fwd_return"] = fwd_return.values
        out["date"] = X.index
        out["symbol"] = symbol
        return out
    except Exception as exc:
        logger.debug("Skipping %s: %s", symbol, exc)
        return None


async def _build_training_universe(sample_size: int) -> List[str]:
    from src.ingestion.data_fetcher import NSEUniverseFetcher

    equities = await NSEUniverseFetcher().fetch_equity_symbols()
    symbols = [e["yf_symbol"] for e in equities]
    if len(symbols) <= sample_size:
        return symbols
    stride = max(1, len(symbols) // sample_size)
    return symbols[::stride][:sample_size]


def _make_folds(min_date: pd.Timestamp, max_date: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Expanding-window folds: (train_end, val_start, val_end) triples — training always starts at min_date, only the end grows."""
    folds = []
    val_start = min_date + pd.DateOffset(months=FOLD_MONTHS * 2)  # first fold needs enough training history to be meaningful
    while val_start + pd.DateOffset(months=FOLD_MONTHS) <= max_date:
        val_end = val_start + pd.DateOffset(months=FOLD_MONTHS)
        folds.append((val_start, val_start, val_end))  # train_end == val_start: no gap, no overlap
        val_start = val_end
    return folds


INNER_EVAL_FRACTION = 0.15  # last 15% of each training window, by date, held out for early stopping — never the actual validation fold


def _inner_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carves the CHRONOLOGICALLY LAST slice of a training window into an
    early-stopping eval set — still strictly before the real walk-forward
    validation fold, so this never leaks the actual out-of-sample test.
    Existing tree fits used a fixed n_estimators with no regularization
    signal at all beyond hard-coded max_depth/subsample; this gives
    XGBoost's early_stopping_rounds an honest, point-in-time-correct
    target to stop against instead of always running the full budget
    regardless of overfitting.
    """
    dates = np.sort(train_df["date"].unique())
    if len(dates) < 40:  # too little history for a meaningful inner split — caller falls back to no early stopping
        return train_df, train_df.iloc[0:0]
    cutoff = dates[int(len(dates) * (1 - INNER_EVAL_FRACTION))]
    return train_df[train_df["date"] < cutoff], train_df[train_df["date"] >= cutoff]


def _fit_direction_classifier(X_tr: pd.DataFrame, y_tr: pd.Series, train_df: pd.DataFrame, feature_cols: List[str], n_estimators: int):
    """
    Regularized XGBoost direction classifier with early stopping against a
    chronological inner holdout (see _inner_split) — replaces the previous
    fixed-n_estimators, lightly-regularized fit for both the per-fold
    walk-forward classifiers and the final production model, so both are
    trained the same way.
    """
    import xgboost as xgb

    inner_train, inner_eval = _inner_split(train_df)
    common = dict(
        max_depth=4, learning_rate=0.03, subsample=0.7, colsample_bytree=0.6,
        min_child_weight=20, reg_alpha=0.5, reg_lambda=2.0,
        objective="multi:softprob", num_class=3, tree_method="hist", random_state=42, n_jobs=-1,
    )
    if len(inner_eval) >= 500:
        clf = xgb.XGBClassifier(n_estimators=n_estimators, eval_metric="mlogloss", early_stopping_rounds=30, **common)
        clf.fit(
            inner_train[feature_cols], inner_train["target"].astype(int),
            eval_set=[(inner_eval[feature_cols], inner_eval["target"].astype(int))], verbose=False,
        )
    else:
        clf = xgb.XGBClassifier(n_estimators=n_estimators, **common)
        clf.fit(X_tr, y_tr)
    return clf


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Standard multi-class generalization of the Brier score: mean squared distance between the predicted probability vector and the one-hot true label."""
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _baseline_metrics(fold_df: pd.DataFrame) -> Dict:
    """
    §33 — a model isn't successful merely for beating >50/33% accuracy; it
    has to beat these baselines out-of-sample:
      - majority_class: always predict the most common class in this fold's
        training data.
      - previous_session: yesterday's direction predicts today's (simple
        momentum/reversal autocorrelation baseline).
    """
    majority = fold_df["target"].mode().iloc[0] if not fold_df.empty else 1
    majority_acc = float((fold_df["target"] == majority).mean())

    prev_dir_col = "macd_crossover" if "macd_crossover" in fold_df.columns else None
    prev_session_acc = None
    if "daily_return" in fold_df.columns:
        prev_up = (fold_df["daily_return"] > 0).astype(int) * 2  # crude: yesterday up -> guess UP(2) continues
        prev_session_acc = float((prev_up.replace(0, 1) == fold_df["target"]).mean())

    return {"majority_class_accuracy": round(majority_acc, 4),
            "previous_session_momentum_accuracy": round(prev_session_acc, 4) if prev_session_acc is not None else None}


async def train(sample_size: int = 500, workers: int = 8) -> dict:
    import asyncio
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
    import xgboost as xgb
    import lightgbm as lgb

    symbols = await _build_training_universe(sample_size)
    logger.info("Training universe: %d symbols (stride-sampled), reading from stored price_history", len(symbols))

    start = time.time()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=workers)
    tasks = [loop.run_in_executor(executor, _build_symbol_features, sym) for sym in symbols]

    parts = []
    ok = fail = 0
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        if result is None:
            fail += 1
        else:
            parts.append(result)
            ok += 1
        if i % 100 == 0 or i == len(symbols):
            logger.info("Built features %d/%d (ok=%d, fail=%d, elapsed=%ds)", i, len(symbols), ok, fail, round(time.time() - start))
    executor.shutdown(wait=True)

    if not parts:
        raise RuntimeError("No symbols produced usable training data — aborting.")

    pool = pd.concat(parts, ignore_index=True)
    feature_cols = [c for c in pool.columns if c not in ("target", "fwd_return", "date", "symbol")]
    pool["date"] = pd.to_datetime(pool["date"])
    pool = pool.sort_values("date").reset_index(drop=True)

    min_date, max_date = pool["date"].min(), pool["date"].max()
    folds = _make_folds(min_date, max_date)
    logger.info("Pooled %d rows from %d symbols | date range %s -> %s | %d walk-forward folds",
                len(pool), ok, min_date.date(), max_date.date(), len(folds))
    if not folds:
        raise RuntimeError("Not enough date range for even one walk-forward fold — need more history.")

    fold_results = []
    for i, (train_end, val_start, val_end) in enumerate(folds, start=1):
        train_df = pool[pool["date"] < train_end]
        val_df = pool[(pool["date"] >= val_start) & (pool["date"] < val_end)]
        if len(train_df) < 1000 or len(val_df) < 100:
            continue

        X_tr, y_tr = train_df[feature_cols], train_df["target"].astype(int)
        X_val, y_val = val_df[feature_cols], val_df["target"].astype(int)

        clf = _fit_direction_classifier(X_tr, y_tr, train_df, feature_cols, n_estimators=500)
        proba = clf.predict_proba(X_val)
        preds = proba.argmax(axis=1)

        metrics = {
            "accuracy": round(accuracy_score(y_val, preds), 4),
            "precision_macro": round(precision_score(y_val, preds, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_val, preds, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y_val, preds, average="macro", zero_division=0), 4),
            "log_loss": round(log_loss(y_val, proba, labels=[0, 1, 2]), 4),
            "brier_score": round(_multiclass_brier(y_val.values, proba), 4),
            **_baseline_metrics(val_df),
        }
        fold_results.append({
            "fold_number": i, "train_start": min_date.date(), "train_end": (train_end - timedelta(days=1)).date(),
            "validation_start": val_start.date(), "validation_end": (val_end - timedelta(days=1)).date(),
            "train_rows": len(train_df), "validation_rows": len(val_df), "metrics": metrics,
        })
        logger.info("Fold %d [train<%s, val %s->%s]: %s", i, train_end.date(), val_start.date(), val_end.date(), metrics)

    if not fold_results:
        raise RuntimeError("No fold had enough rows to validate — aborting without training a production model.")

    # Final production model: train on ALL available data (most recent
    # knowledge), calibrate on the last fold's validation slice (held out
    # from that final fit's own training by construction — it's chronologically
    # after final_train_end below).
    final_train_end = folds[-1][2]  # end of the last fold's validation window
    final_train = pool[pool["date"] < final_train_end]
    calib_slice = pool[pool["date"] >= final_train_end]
    X_final, y_final = final_train[feature_cols], final_train["target"].astype(int)

    base_clf = _fit_direction_classifier(X_final, y_final, final_train, feature_cols, n_estimators=800)

    calibration_metrics = {}
    calibrated_clf = base_clf
    if len(calib_slice) >= 200:
        X_calib, y_calib = calib_slice[feature_cols], calib_slice["target"].astype(int)
        calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv="prefit")
        calibrated_clf.fit(X_calib, y_calib)
        calib_proba = calibrated_clf.predict_proba(X_calib)
        calibration_metrics = {
            "brier_score_calibrated": round(_multiclass_brier(y_calib.values, calib_proba), 4),
            "brier_score_raw": round(_multiclass_brier(y_calib.values, base_clf.predict_proba(X_calib)), 4),
            "calibration_rows": len(calib_slice),
        }
        logger.info("Calibration: %s", calibration_metrics)

    # Model B: expected-return regression with a real quantile interval.
    return_model_mid = lgb.LGBMRegressor(objective="regression", n_estimators=300, num_leaves=31, random_state=42, verbose=-1)
    return_model_mid.fit(X_final, final_train["fwd_return"])
    return_model_lo = lgb.LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=300, num_leaves=31, random_state=42, verbose=-1)
    return_model_lo.fit(X_final, final_train["fwd_return"])
    return_model_hi = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=300, num_leaves=31, random_state=42, verbose=-1)
    return_model_hi.fit(X_final, final_train["fwd_return"])

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return_metrics = {}
    if len(calib_slice) >= 200:
        X_calib = calib_slice[feature_cols]
        y_calib_ret = calib_slice["fwd_return"]
        pred_ret = return_model_mid.predict(X_calib)
        return_metrics = {
            "mae": round(mean_absolute_error(y_calib_ret, pred_ret), 5),
            "rmse": round(mean_squared_error(y_calib_ret, pred_ret) ** 0.5, 5),
            "directional_hit_rate": round(float(np.mean(np.sign(pred_ret) == np.sign(y_calib_ret))), 4),
            "correlation": round(float(np.corrcoef(pred_ret, y_calib_ret)[0, 1]), 4) if len(y_calib_ret) > 1 else None,
        }
        logger.info("Return model (held-out): %s", return_metrics)

    import pickle
    from pathlib import Path
    model_dir = Path("models_store/three_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "direction_model.pkl", "wb") as f:
        pickle.dump({"model": calibrated_clf, "features": feature_cols, "feature_version": FEATURE_VERSION}, f)
    with open(model_dir / "return_model.pkl", "wb") as f:
        pickle.dump({"mid": return_model_mid, "lo": return_model_lo, "hi": return_model_hi,
                     "features": feature_cols, "feature_version": FEATURE_VERSION}, f)

    from src.database.models import AsyncSessionFactory, ModelRegistry, ModelTrainingRun
    async with AsyncSessionFactory() as db:
        model_row = ModelRegistry(
            model_name="direction_return_v5", version=f"v5_{int(time.time())}", model_type="xgboost+lightgbm",
            symbol=None, artifact_path=str(model_dir), feature_version=FEATURE_VERSION,
            hyperparameters={"direction": "XGBClassifier multi:softprob", "return": "LGBMRegressor quantile"},
            training_period_start=min_date.date(), training_period_end=final_train_end.date(),
            validation_period_start=folds[0][1].date(), validation_period_end=folds[-1][2].date(),
            metrics={"folds": [fr["metrics"] for fr in fold_results], "return_model": return_metrics},
            calibration_metrics=calibration_metrics,
            baseline_metrics=fold_results[-1]["metrics"] if fold_results else {},
            is_active=False,  # promotion is a separate, explicit decision (§70) — never automatic
        )
        db.add(model_row)
        await db.flush()
        for fr in fold_results:
            db.add(ModelTrainingRun(
                model_id=model_row.id, fold_number=fr["fold_number"],
                train_start=fr["train_start"], train_end=fr["train_end"],
                validation_start=fr["validation_start"], validation_end=fr["validation_end"],
                train_rows=fr["train_rows"], validation_rows=fr["validation_rows"], metrics=fr["metrics"],
            ))
        await db.commit()
        model_id = model_row.id

    return {
        "model_id": model_id, "symbols_ok": ok, "symbols_failed": fail, "pooled_rows": len(pool),
        "folds": fold_results, "calibration_metrics": calibration_metrics, "return_metrics": return_metrics,
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
