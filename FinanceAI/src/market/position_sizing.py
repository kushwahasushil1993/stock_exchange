"""
Risk-based ("fixed fractional") position sizing — standard, published risk
management, not this project's invention. A trade risks the same fixed
fraction of total capital regardless of the underlying's volatility: a
position with a tight stop can size up, a position with a wide stop must
size down, so no single trade (however "confident" the model is) can blow
up the portfolio on its own.

This is deliberately separate from the ML models: Model A/B/C decide
WHETHER and in WHICH DIRECTION there's a setup; this module decides HOW
MUCH capital to put behind it once a setup is found — a decision that
should hold regardless of which model produced the signal.
"""
from dataclasses import dataclass
from typing import Optional

# 1% risk per trade is the conventional, conservative end of the standard
# textbook range (1-2%) for retail systematic risk management — not a
# figure tuned on this project's own backtest results.
DEFAULT_RISK_PER_TRADE = 0.01
# Never put more than 20% of capital in one name regardless of how tight
# its stop is — caps concentration risk a very-tight-stop trade could
# otherwise create by sizing up aggressively.
MAX_POSITION_FRACTION = 0.20


@dataclass
class PositionSize:
    shares: int
    position_value: float
    risk_amount: float
    stop_distance_pct: float
    capped_by_max_position: bool


def size_position(
    capital: float, entry_price: float, stop_loss_price: float,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE, max_position_fraction: float = MAX_POSITION_FRACTION,
) -> Optional[PositionSize]:
    """
    Position size = (capital * risk_per_trade) / stop_distance_pct — the
    dollar risk is held constant; the position value moves inversely with
    how far away the stop is. Returns None for a nonsensical/incomplete
    input (non-positive prices, a stop that isn't actually below entry for
    a long, or the computed size rounding to 0 shares) rather than sizing
    a trade off invalid data.
    """
    if capital <= 0 or entry_price <= 0 or stop_loss_price <= 0 or stop_loss_price >= entry_price:
        return None

    stop_distance_pct = (entry_price - stop_loss_price) / entry_price
    risk_amount = capital * risk_per_trade
    position_value = risk_amount / stop_distance_pct

    capped = False
    max_value = capital * max_position_fraction
    if position_value > max_value:
        position_value = max_value
        capped = True

    shares = int(position_value // entry_price)
    if shares <= 0:
        return None

    actual_position_value = shares * entry_price
    actual_risk = shares * (entry_price - stop_loss_price)

    return PositionSize(
        shares=shares,
        position_value=round(actual_position_value, 2),
        risk_amount=round(actual_risk, 2),
        stop_distance_pct=round(stop_distance_pct, 4),
        capped_by_max_position=capped,
    )
