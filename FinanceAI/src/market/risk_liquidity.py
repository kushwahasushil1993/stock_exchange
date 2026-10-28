"""
Model C: Risk / Liquidity — deliberately rule-based, not trained (§65: "do
not arbitrarily assign risk labels" — a documented threshold is auditable
in a way a black-box model's label isn't). Every threshold here is a fixed
constant, listed once, so they can be reviewed/tuned without hunting
through code.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.database.models import LiquidityLevel, RiskLevel

# ── Risk thresholds (on ATR as % of price — atr_pct, already computed by
#    TechnicalFeatures) ──────────────────────────────────────────────────
RISK_LOW_ATR_PCT = 0.015   # < 1.5% average daily range: LOW_RISK
RISK_HIGH_ATR_PCT = 0.035  # > 3.5%: HIGH_RISK; between the two: MEDIUM_RISK
# A drawdown from the recent 52-week high beyond this bumps risk up one
# level regardless of ATR — a stock already down hard carries more downside
# risk than its recent daily range alone suggests.
RISK_DRAWDOWN_BUMP_PCT = 0.30

# ── Liquidity thresholds (on 20-day average traded value = price * volume,
#    in INR) ─────────────────────────────────────────────────────────────
LIQUIDITY_HIGH_VALUE = 50_00_00_000     # >= 50 crore/day: HIGH
LIQUIDITY_MEDIUM_VALUE = 5_00_00_000    # >= 5 crore/day: MEDIUM
LIQUIDITY_LOW_VALUE = 50_00_000         # >= 50 lakh/day: LOW; below: UNSUITABLE
# A stock trading at a fraction of its own recent average volume today is
# less liquid RIGHT NOW than its historical average suggests.
RELATIVE_VOLUME_UNSUITABLE = 0.15  # today's volume < 15% of 20d average -> demote one level


@dataclass
class RiskLiquidityResult:
    risk_level: RiskLevel
    liquidity_level: LiquidityLevel
    details: dict


def assess_risk_liquidity(enriched: pd.DataFrame) -> RiskLiquidityResult:
    """
    `enriched` is a TechnicalFeatures-enriched OHLCV DataFrame (expects
    atr_pct, pct_from_52w_high, vol_ratio, vol_sma_20, Close, Volume — all
    already computed by the existing feature pipeline). Uses only the last
    row (today), since risk/liquidity are point-in-time assessments.
    """
    last = enriched.iloc[-1]
    atr_pct = last.get("atr_pct")
    drawdown_pct = last.get("pct_from_52w_high")  # negative = below the 52w high
    vol_ratio = last.get("vol_ratio")
    close = last.get("Close")
    vol_sma_20 = last.get("vol_sma_20")

    # --- Risk ---
    if pd.isna(atr_pct):
        risk = RiskLevel.MEDIUM_RISK  # unknown volatility -> don't claim LOW
    elif atr_pct < RISK_LOW_ATR_PCT:
        risk = RiskLevel.LOW_RISK
    elif atr_pct > RISK_HIGH_ATR_PCT:
        risk = RiskLevel.HIGH_RISK
    else:
        risk = RiskLevel.MEDIUM_RISK

    if pd.notna(drawdown_pct) and drawdown_pct < -100 * RISK_DRAWDOWN_BUMP_PCT:
        if risk == RiskLevel.LOW_RISK:
            risk = RiskLevel.MEDIUM_RISK
        elif risk == RiskLevel.MEDIUM_RISK:
            risk = RiskLevel.HIGH_RISK

    # --- Liquidity ---
    avg_daily_value = (float(close) * float(vol_sma_20)) if pd.notna(close) and pd.notna(vol_sma_20) else None
    if avg_daily_value is None:
        liquidity = LiquidityLevel.UNSUITABLE  # can't assess -> don't claim tradability
    elif avg_daily_value >= LIQUIDITY_HIGH_VALUE:
        liquidity = LiquidityLevel.HIGH
    elif avg_daily_value >= LIQUIDITY_MEDIUM_VALUE:
        liquidity = LiquidityLevel.MEDIUM
    elif avg_daily_value >= LIQUIDITY_LOW_VALUE:
        liquidity = LiquidityLevel.LOW
    else:
        liquidity = LiquidityLevel.UNSUITABLE

    if pd.notna(vol_ratio) and vol_ratio < RELATIVE_VOLUME_UNSUITABLE:
        order = [LiquidityLevel.HIGH, LiquidityLevel.MEDIUM, LiquidityLevel.LOW, LiquidityLevel.UNSUITABLE]
        idx = order.index(liquidity)
        liquidity = order[min(idx + 1, len(order) - 1)]

    return RiskLiquidityResult(
        risk_level=risk, liquidity_level=liquidity,
        details={
            "atr_pct": None if pd.isna(atr_pct) else round(float(atr_pct), 5),
            "drawdown_from_52w_high_pct": None if pd.isna(drawdown_pct) else round(float(drawdown_pct), 2),
            "avg_daily_traded_value_inr": None if avg_daily_value is None else round(avg_daily_value, 0),
            "relative_volume_today": None if pd.isna(vol_ratio) else round(float(vol_ratio), 2),
        },
    )
