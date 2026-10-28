"""
Portfolio-level backtest — the direction+return models (walk-forward
validated per-row in train_three_models.py) evaluated as an actual TRADING
STRATEGY instead of just a classifier: real position sizing, a documented
transaction-cost assumption, and a real portfolio equity curve.

WHY THIS EXISTS: per-row classifier accuracy and portfolio profitability
are different questions. A model whose single-row direction accuracy sits
right at a naive baseline can still, in principle, produce a profitable
diversified basket if the errors across many simultaneous positions are
uncorrelated enough — or it can still lose money even with slightly-above-
baseline accuracy, once real transaction costs are applied. Only an actual
strategy backtest answers "would this have made money," and until this
script, only single-row accuracy had been tested.

Also runs the cross-sectional decile-spread test: even a model whose
absolute BUY/SELL call is unreliable might still RANK stocks better than
random — if it does, decile 10 (highest predicted expected_return each
day) should realize a higher mean forward return than decile 1. If it
doesn't, the model has no usable ranking signal either.

METHODOLOGY, stated plainly:
  - Same walk-forward folds as train_three_models.py (build_pooled_dataset)
    — never a random train_test_split on time-series data.
  - Each fold's direction classifier + return model are fit FRESH on that
    fold's training window only (same _fit_direction_classifier
    train_three_models.py uses) — these are NOT the champion model in
    models_store/, they're the same walk-forward methodology applied to a
    different question.
  - Every candidate row is run through the SAME assess_risk_liquidity() +
    decide() the live system uses — not a backtest-only approximation —
    reconstructed from the pool's own columns (Close, Volume, atr_pct,
    vol_ratio, pct_from_52w_high are already in the feature matrix).
  - Long-only: takes every POSITIVE_DIRECTIONAL_SETUP that day, skips
    NEGATIVE_SETUP (shorting isn't modeled) and NO_CLEAR_SIGNAL (most
    rows, by design).
  - Held for exactly HORIZON_DAYS (1 session, matching the platform's own
    target) — realized return is that row's already-computed,
    point-in-time-correct fwd_return; no re-simulation of intraday price
    action.
  - ROUND_TRIP_COST_PCT is a documented ASSUMPTION (typical NSE delivery
    brokerage + STT + exchange charges + stamp duty + slippage for a
    retail account), not something measured from a real broker on real
    orders — stated once, applied consistently, easy to override.

DOCUMENTED SIMPLIFICATIONS (so this isn't mistaken for a full live-trading
simulation):
  - market_regime is passed as None to decide() (no per-row historical
    regime is reconstructed here) — decide() only hard-blocks on
    MarketRegime.UNDEFINED, so None just skips the regime-alignment
    factor, it never incorrectly disqualifies a trade.
  - No intraday fills, no partial fills, no borrow cost for shorts (none
    are taken), and no capital-gains tax.

Usage:
    python -m src.pipelines.portfolio_backtest [--sample-size 500] [--workers 8] [--capital 1000000]
"""
import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_backtest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Conservative, documented assumption — see module docstring. 0.1% one-way
# (0.2% round trip) is a commonly used retail-delivery-trade estimate for
# NSE brokerage+STT+exchange charges+stamp duty+slippage combined; not
# measured from a real broker on real orders.
ROUND_TRIP_COST_PCT = 0.002
# No more than this fraction of total capital deployed across ALL
# simultaneous open positions on a given day — a portfolio-level risk cap
# independent of size_position()'s per-trade cap.
MAX_DAILY_DEPLOYMENT_FRACTION = 0.80
STARTING_CAPITAL = 1_000_000.0  # ₹10 lakh — an arbitrary but round starting figure; only relative/percentage results matter


def _row_risk_liquidity(row: pd.Series):
    """
    Reconstructs the tiny slice of an "enriched OHLCV" row assess_risk_liquidity()
    actually reads (Close, Volume, atr_pct, vol_ratio, pct_from_52w_high) from
    the pool's own columns — vol_sma_20 isn't itself a pool column, but
    Volume/vol_ratio reconstructs it exactly (vol_ratio is defined as
    Volume/vol_sma_20 in TechnicalFeatures), so this calls the EXACT SAME
    function the live system uses, not a re-implementation of its logic.
    """
    from src.market.risk_liquidity import assess_risk_liquidity

    vol_ratio = row.get("vol_ratio")
    volume = row.get("Volume")
    vol_sma_20 = (volume / vol_ratio) if pd.notna(vol_ratio) and vol_ratio not in (0, None) and pd.notna(volume) else np.nan

    fake_enriched = pd.DataFrame([{
        "Close": row.get("Close"), "Volume": volume, "atr_pct": row.get("atr_pct"),
        "vol_ratio": vol_ratio, "vol_sma_20": vol_sma_20, "pct_from_52w_high": row.get("pct_from_52w_high"),
    }])
    return assess_risk_liquidity(fake_enriched)


def _decide_row(row: pd.Series, probability_up: float, probability_down: float, probability_flat: float, expected_return: float):
    from src.market.decision_engine import decide

    risk_result = _row_risk_liquidity(row)
    return decide(
        probability_up=probability_up, probability_down=probability_down, probability_flat=probability_flat,
        expected_return=expected_return, risk_level=risk_result.risk_level,
        liquidity_level=risk_result.liquidity_level, market_regime=None,
    ), risk_result


def _decile_spread_test(records: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, ranks that day's candidate rows into up to 10 deciles by
    predicted expected_return, then averages each decile's REALIZED
    fwd_return across dates. Ranking is done WITHIN each date (never across
    dates) — cross-sectional signal is about relative ranking on a given
    day, not comparing absolute predicted-return levels from different
    market regimes.
    """
    def _decile(group: pd.DataFrame) -> pd.Series:
        n_bins = min(10, group["expected_return"].nunique())
        if n_bins < 2:
            return pd.Series(np.nan, index=group.index)
        return pd.qcut(group["expected_return"], n_bins, labels=False, duplicates="drop")

    records = records.copy()
    records["decile"] = records.groupby("date", group_keys=False).apply(_decile, include_groups=False)
    per_date = records.dropna(subset=["decile"]).groupby(["date", "decile"])["fwd_return"].mean().reset_index()
    summary = per_date.groupby("decile")["fwd_return"].agg(["mean", "count"]).reset_index()
    summary.columns = ["decile", "mean_realized_fwd_return", "date_decile_observations"]
    return summary.sort_values("decile")


async def run_backtest(sample_size: int = 500, workers: int = 8, capital: float = STARTING_CAPITAL) -> dict:
    from src.pipelines.train_three_models import build_pooled_dataset, _fit_direction_classifier
    from src.database.models import DecisionType
    import lightgbm as lgb

    pool, feature_cols, folds, ok, fail = await build_pooled_dataset(sample_size, workers)

    all_trades: List[dict] = []
    decile_records: List[pd.DataFrame] = []

    for fold_num, (train_end, val_start, val_end) in enumerate(folds, start=1):
        train_df = pool[pool["date"] < train_end]
        val_df = pool[(pool["date"] >= val_start) & (pool["date"] < val_end)]
        if len(train_df) < 1000 or len(val_df) < 100:
            continue

        X_tr, y_tr = train_df[feature_cols], train_df["target"].astype(int)
        clf = _fit_direction_classifier(X_tr, y_tr, train_df, feature_cols, n_estimators=500)

        return_model = lgb.LGBMRegressor(objective="regression", n_estimators=300, num_leaves=31, random_state=42, verbose=-1)
        return_model.fit(X_tr, train_df["fwd_return"])

        X_val = val_df[feature_cols]
        proba = clf.predict_proba(X_val)  # [DOWN, FLAT, UP] — class order 0,1,2, matches direction_classes=3 encoding
        expected_returns = return_model.predict(X_val)

        val_records = val_df[["date", "symbol", "Close", "fwd_return"]].copy()
        val_records["expected_return"] = expected_returns
        decile_records.append(val_records)

        for (idx, row), p, exp_ret in zip(val_df.iterrows(), proba, expected_returns):
            decision, risk_result = _decide_row(row, probability_down=float(p[0]), probability_flat=float(p[1]),
                                                 probability_up=float(p[2]), expected_return=float(exp_ret))
            if decision.decision != DecisionType.POSITIVE_SETUP:
                continue  # long-only — skip NEGATIVE_SETUP and NO_CLEAR_SIGNAL

            entry_price = float(row["Close"])
            atr_pct = row.get("atr_pct")
            if pd.isna(atr_pct) or entry_price <= 0:
                continue
            stop_loss_price = entry_price * (1 - 1.5 * float(atr_pct))  # same 1.5x ATR stop the live orchestrator uses
            realized_return = row["fwd_return"]
            if pd.isna(realized_return):
                continue

            all_trades.append({
                "fold": fold_num, "date": row["date"], "symbol": row["symbol"],
                "entry_price": entry_price, "stop_loss_price": stop_loss_price,
                "expected_return": float(exp_ret), "realized_return": float(realized_return),
                "risk_level": risk_result.risk_level.value, "liquidity_level": risk_result.liquidity_level.value,
            })

        logger.info("Fold %d: %d candidate POSITIVE_SETUP trades from %d validation rows",
                    fold_num, sum(1 for t in all_trades if t["fold"] == fold_num), len(val_df))

    if not all_trades:
        logger.warning("No POSITIVE_DIRECTIONAL_SETUP trades fired across any fold — decision engine found no long signal at all in this sample.")
        trades_df = pd.DataFrame(columns=["date", "symbol", "realized_return"])
    else:
        trades_df = pd.DataFrame(all_trades)

    # ── Portfolio simulation: real capital, real position sizing, real costs ──
    from src.market.position_sizing import size_position

    equity = capital
    equity_curve = []
    if not trades_df.empty:
        for date, day_trades in trades_df.groupby("date"):
            day_capital_cap = equity * MAX_DAILY_DEPLOYMENT_FRACTION
            deployed = 0.0
            day_pnl = 0.0
            n_positions = 0
            for _, t in day_trades.iterrows():
                if deployed >= day_capital_cap:
                    break
                remaining = day_capital_cap - deployed
                sized = size_position(min(equity, remaining), t["entry_price"], t["stop_loss_price"])
                if sized is None:
                    continue
                gross_pnl = sized.position_value * t["realized_return"]
                cost = sized.position_value * ROUND_TRIP_COST_PCT
                day_pnl += gross_pnl - cost
                deployed += sized.position_value
                n_positions += 1
            equity += day_pnl
            equity_curve.append({"date": date, "equity": equity, "day_pnl": day_pnl, "n_positions": n_positions})

    equity_df = pd.DataFrame(equity_curve).sort_values("date") if equity_curve else pd.DataFrame(columns=["date", "equity", "day_pnl", "n_positions"])

    metrics = _compute_portfolio_metrics(equity_df, capital)
    metrics["nifty_buy_and_hold"] = await _nifty_benchmark(equity_df)

    decile_summary = pd.DataFrame()
    if decile_records:
        decile_summary = _decile_spread_test(pd.concat(decile_records, ignore_index=True))

    return {
        "starting_capital": capital,
        "total_candidate_trades": len(trades_df),
        "trading_days": len(equity_df),
        "portfolio_metrics": metrics,
        "decile_spread": decile_summary.to_dict(orient="records"),
        "sample_trades": trades_df.head(10).to_dict(orient="records") if not trades_df.empty else [],
    }


async def _nifty_benchmark(equity_df: pd.DataFrame) -> Optional[dict]:
    """
    What buying and holding NIFTY over the IDENTICAL date range would have
    returned — "beats a naive baseline classifier" (already tested
    elsewhere) and "beats just buying the index" are different questions,
    and only the second one tells you whether the effort in this whole
    system was worth it versus doing nothing.
    """
    if equity_df.empty:
        return None
    from src.pipelines.train_three_models import _load_symbol_history

    nifty = _load_symbol_history("^NSEI")
    if nifty is None or nifty.empty:
        return None

    start_date, end_date = equity_df["date"].min(), equity_df["date"].max()
    window = nifty[(nifty.index >= start_date) & (nifty.index <= end_date)]
    if len(window) < 2:
        return None

    start_price, end_price = float(window["Close"].iloc[0]), float(window["Close"].iloc[-1])
    total_return_pct = (end_price / start_price - 1) * 100
    daily_returns = window["Close"].pct_change().dropna()
    sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)) if daily_returns.std() > 0 else None
    cummax = window["Close"].cummax()
    max_drawdown_pct = float(((window["Close"] - cummax) / cummax).min() * 100)

    return {
        "total_return_pct": round(total_return_pct, 2),
        "annualized_sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "start_date": str(start_date.date()) if hasattr(start_date, "date") else str(start_date),
        "end_date": str(end_date.date()) if hasattr(end_date, "date") else str(end_date),
    }


def _compute_portfolio_metrics(equity_df: pd.DataFrame, starting_capital: float) -> dict:
    if equity_df.empty:
        return {"note": "No trades were ever taken — nothing to compute portfolio metrics from."}

    final_equity = equity_df["equity"].iloc[-1]
    total_return_pct = (final_equity / starting_capital - 1) * 100

    daily_returns = equity_df["equity"].pct_change().dropna()
    # Annualized Sharpe (rf=0, 252 trading days) — only meaningful with
    # enough trading days; reported as None otherwise rather than a noisy number.
    sharpe = None
    if len(daily_returns) >= 30 and daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

    cummax = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - cummax) / cummax
    max_drawdown_pct = float(drawdown.min() * 100)

    win_rate = float((equity_df["day_pnl"] > 0).mean()) if len(equity_df) else None

    return {
        "final_equity": round(float(final_equity), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "annualized_sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate_by_day": round(win_rate, 4) if win_rate is not None else None,
        "avg_daily_positions": round(float(equity_df["n_positions"].mean()), 2),
        "trading_days": len(equity_df),
    }


if __name__ == "__main__":
    import asyncio
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--capital", type=float, default=STARTING_CAPITAL)
    args = parser.parse_args()

    result = asyncio.run(run_backtest(sample_size=args.sample_size, workers=args.workers, capital=args.capital))
    print(json.dumps(result, indent=2, default=str))
