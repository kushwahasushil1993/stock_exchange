"""
Stock-vs-index relative-strength features (§15/§18: stock_return_vs_nifty,
rolling_beta_nifty, rolling_correlation_nifty, nifty_momentum,
nifty_volatility) — computed by aligning a stock's own enriched OHLCV
DataFrame with NIFTY/SENSEX's daily returns by date, purely from already-
elapsed history (no forward alignment possible: a rolling window at row i
only ever looks at rows <= i).
"""
from typing import Optional

import numpy as np
import pandas as pd

ROLLING_WINDOW = 60  # ~3 trading months — long enough for a stable beta/correlation estimate, short enough to track regime changes


def add_index_relative_features(stock_df: pd.DataFrame, index_df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Adds `{index_name}_return`, `return_vs_{index_name}`,
    `rolling_beta_{index_name}`, `rolling_corr_{index_name}` to a copy of
    stock_df (already-enriched with TechnicalFeatures, expects a
    `daily_return` column already present, matching this codebase's
    existing convention). No-op (returns stock_df unchanged) if index_df is
    unavailable — a missing index history should degrade gracefully, not
    crash the whole feature pipeline for stocks that otherwise have
    everything they need.
    """
    if index_df is None or index_df.empty or "Close" not in index_df.columns:
        return stock_df

    df = stock_df.copy()
    idx_ret = index_df["Close"].pct_change().rename(f"{index_name}_return")
    df = df.join(idx_ret, how="left")

    stock_ret = df["Close"].pct_change() if "daily_return" not in df.columns else df["daily_return"] / 100
    df[f"return_vs_{index_name}"] = (stock_ret - df[f"{index_name}_return"]) * 100

    cov = stock_ret.rolling(ROLLING_WINDOW).cov(df[f"{index_name}_return"])
    var = df[f"{index_name}_return"].rolling(ROLLING_WINDOW).var()
    df[f"rolling_beta_{index_name}"] = cov / var.replace(0, np.nan)
    df[f"rolling_corr_{index_name}"] = stock_ret.rolling(ROLLING_WINDOW).corr(df[f"{index_name}_return"])

    return df
