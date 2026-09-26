"""
Market regime classification from NIFTY + SENSEX — computed ONCE per
trading day (not redundantly per stock) and read by every stock's decision
layer. Every threshold below is a fixed, documented constant, not an
arbitrary label — see the docstring on `classify_regime`.
"""
import logging
from datetime import date as date_type
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.database.models import AsyncSessionFactory, MarketRegime, MarketRegimeSnapshot, PriceHistory, Stock

logger = logging.getLogger("market_regime")

NIFTY_SYMBOL = "^NSEI"
SENSEX_SYMBOL = "^BSESN"

# Wilder's own standard ADX thresholds (not this project's invention) —
# ADX > 25 is conventionally "trending", < 20 "non-trending"/range-bound.
ADX_TRENDING = 25
ADX_RANGE_BOUND = 20
# Volatility regime relative to the index's OWN trailing 1-year median ATR%
# — "high"/"low" relative to itself, not an absolute cutoff that would mean
# different things for different volatility eras.
HIGH_VOL_MULTIPLE = 1.5
LOW_VOL_MULTIPLE = 0.67


async def _load_index_history(symbol: str) -> Optional[pd.DataFrame]:
    async with AsyncSessionFactory() as db:
        stock = (await db.execute(select(Stock).where(Stock.symbol == symbol))).scalar_one_or_none()
        if not stock:
            return None
        rows = (
            await db.execute(
                select(PriceHistory.date, PriceHistory.open, PriceHistory.high,
                       PriceHistory.low, PriceHistory.close, PriceHistory.volume)
                .where(PriceHistory.stock_id == stock.id).order_by(PriceHistory.date)
            )
        ).all()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def _index_signals(df: pd.DataFrame) -> dict:
    """ADX, trend direction, and ATR%-vs-its-own-history for one index."""
    from src.features.technical import TechnicalFeatures

    enriched = TechnicalFeatures().compute(df)
    last = enriched.iloc[-1]

    adx = last.get("adx")
    sma20, sma50 = last.get("sma_20"), last.get("sma_50")
    atr_pct = last.get("atr_pct")
    atr_pct_history = enriched["atr_pct"].dropna()
    atr_pct_median = atr_pct_history.tail(252).median() if len(atr_pct_history) >= 60 else None
    momentum = last.get("roc_10")

    trend_dir = None
    if sma20 is not None and sma50 is not None and pd.notna(sma20) and pd.notna(sma50):
        trend_dir = "UP" if sma20 > sma50 else "DOWN"

    return {
        "adx": float(adx) if pd.notna(adx) else None,
        "trend_dir": trend_dir,
        "atr_pct": float(atr_pct) if pd.notna(atr_pct) else None,
        "atr_pct_median": float(atr_pct_median) if atr_pct_median is not None and pd.notna(atr_pct_median) else None,
        "momentum": float(momentum) if pd.notna(momentum) else None,
    }


def classify_regime(nifty: dict, sensex: dict) -> MarketRegime:
    """
    Priority order (documented, not arbitrary):
      1. HIGH_VOLATILITY if either index's ATR% > 1.5x its own trailing
         1-year median — extreme conditions dominate the read regardless
         of trend, since a "trending" market in a volatility spike behaves
         very differently from a normal trend.
      2. TRENDING_UP / TRENDING_DOWN if NIFTY's ADX > 25 (Wilder's
         standard "trending" threshold) and both indices agree on
         direction (SMA20 vs SMA50).
      3. LOW_VOLATILITY if ATR% < 0.67x its own trailing median and ADX <
         20 (Wilder's "non-trending" threshold) — compressed, quiet market.
      4. RANGE_BOUND otherwise.
      5. UNDEFINED if required indicators are missing (never force a
         classification without enough data — matches "don't force a
         prediction when the market regime is unclear").
    """
    if nifty.get("adx") is None or nifty.get("trend_dir") is None:
        return MarketRegime.UNDEFINED

    for idx in (nifty, sensex):
        if idx.get("atr_pct") is not None and idx.get("atr_pct_median"):
            if idx["atr_pct"] > HIGH_VOL_MULTIPLE * idx["atr_pct_median"]:
                return MarketRegime.HIGH_VOLATILITY

    if nifty["adx"] > ADX_TRENDING:
        if sensex.get("trend_dir") is None or sensex["trend_dir"] == nifty["trend_dir"]:
            return MarketRegime.TRENDING_UP if nifty["trend_dir"] == "UP" else MarketRegime.TRENDING_DOWN

    if nifty["adx"] < ADX_RANGE_BOUND and nifty.get("atr_pct") is not None and nifty.get("atr_pct_median"):
        if nifty["atr_pct"] < LOW_VOL_MULTIPLE * nifty["atr_pct_median"]:
            return MarketRegime.LOW_VOLATILITY

    return MarketRegime.RANGE_BOUND


async def compute_and_store_regime(as_of: Optional[date_type] = None) -> Optional[dict]:
    """Computes today's regime from stored NIFTY/SENSEX history and upserts one row into market_regime_snapshots. Returns the computed snapshot dict, or None if NIFTY/SENSEX history isn't available yet."""
    nifty_df = await _load_index_history(NIFTY_SYMBOL)
    sensex_df = await _load_index_history(SENSEX_SYMBOL)
    if nifty_df is None or nifty_df.empty:
        logger.warning("compute_and_store_regime: no NIFTY history — cannot compute regime")
        return None

    nifty_sig = _index_signals(nifty_df)
    sensex_sig = _index_signals(sensex_df) if sensex_df is not None and not sensex_df.empty else {}
    regime = classify_regime(nifty_sig, sensex_sig)
    snapshot_date = as_of or nifty_df.index[-1].date()

    row = {
        "date": snapshot_date, "regime": regime,
        "nifty_trend": nifty_sig.get("trend_dir"), "nifty_momentum": nifty_sig.get("momentum"),
        "nifty_volatility": nifty_sig.get("atr_pct"),
        "sensex_trend": sensex_sig.get("trend_dir"), "sensex_momentum": sensex_sig.get("momentum"),
        "sensex_volatility": sensex_sig.get("atr_pct"),
        "details": {"nifty": nifty_sig, "sensex": sensex_sig},
    }

    async with AsyncSessionFactory() as db:
        stmt = pg_insert(MarketRegimeSnapshot).values(**row)
        stmt = stmt.on_conflict_do_update(index_elements=["date"], set_={k: stmt.excluded[k] for k in row if k != "date"})
        await db.execute(stmt)
        await db.commit()

    return row


async def get_latest_regime() -> Optional[dict]:
    async with AsyncSessionFactory() as db:
        row = (
            await db.execute(
                select(MarketRegimeSnapshot).order_by(MarketRegimeSnapshot.date.desc()).limit(1)
            )
        ).scalar_one_or_none()
    if not row:
        return None
    return {
        "date": str(row.date), "regime": row.regime.value if row.regime else None,
        "nifty_trend": row.nifty_trend, "nifty_momentum": row.nifty_momentum, "nifty_volatility": row.nifty_volatility,
        "sensex_trend": row.sensex_trend, "sensex_momentum": row.sensex_momentum, "sensex_volatility": row.sensex_volatility,
    }
