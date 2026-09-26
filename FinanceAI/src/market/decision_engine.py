"""
The decision layer (§21) — combines Model A (direction probabilities),
Model B (expected return), Model C (risk/liquidity), and the current
market regime into one of three outcomes. Deliberately NOT another model:
a fixed, documented set of rules, so every decision is explainable by
pointing at exactly which rule fired.

NO_CLEAR_SIGNAL is a normal, expected result (§44/§72) — not a failure
mode. It fires whenever the evidence doesn't clear the bar in ANY of these
ways, rather than the system manufacturing a confident-sounding answer out
of weak evidence.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from src.database.models import DecisionType, LiquidityLevel, MarketRegime, RiskLevel

# ── Decision thresholds (documented, not arbitrary) ──────────────────────
PROB_EDGE_MIN = 0.10          # probability_up - probability_down must exceed this to call a direction at all
MIN_ABS_EXPECTED_RETURN = 0.003  # 0.3% — below this, a "direction" isn't worth acting on even if probability favors it
UNSUITABLE_LIQUIDITY = {LiquidityLevel.UNSUITABLE}
DISQUALIFYING_RISK_LIQUIDITY = {LiquidityLevel.LOW, LiquidityLevel.UNSUITABLE}  # combined with HIGH_RISK, blocks a signal


@dataclass
class Decision:
    decision: DecisionType
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)


def decide(
    probability_up: float,
    probability_down: float,
    probability_flat: float,
    expected_return: float,
    risk_level: Optional[RiskLevel],
    liquidity_level: Optional[LiquidityLevel],
    market_regime: Optional[MarketRegime],
) -> Decision:
    positive: List[str] = []
    negative: List[str] = []

    if liquidity_level in UNSUITABLE_LIQUIDITY:
        negative.append("liquidity is UNSUITABLE — average traded value too low to act on reliably")
        return Decision(DecisionType.NO_CLEAR_SIGNAL, positive, negative)

    if market_regime == MarketRegime.UNDEFINED:
        negative.append("market regime is UNDEFINED — not enough index history to characterize current conditions")
        return Decision(DecisionType.NO_CLEAR_SIGNAL, positive, negative)

    edge = probability_up - probability_down
    if abs(edge) < PROB_EDGE_MIN:
        negative.append(f"direction probabilities too close (UP {probability_up:.0%} vs DOWN {probability_down:.0%})")
        return Decision(DecisionType.NO_CLEAR_SIGNAL, positive, negative)

    if abs(expected_return) < MIN_ABS_EXPECTED_RETURN:
        negative.append(f"expected return ({expected_return:.2%}) too small to be meaningful")
        return Decision(DecisionType.NO_CLEAR_SIGNAL, positive, negative)

    if risk_level == RiskLevel.HIGH_RISK and liquidity_level in DISQUALIFYING_RISK_LIQUIDITY:
        negative.append(f"high risk combined with {liquidity_level.value.lower()} liquidity")
        return Decision(DecisionType.NO_CLEAR_SIGNAL, positive, negative)

    is_up = edge > 0
    if is_up:
        positive.append(f"probability UP ({probability_up:.0%}) exceeds DOWN ({probability_down:.0%}) by {edge:.0%}")
        positive.append(f"expected return +{expected_return:.2%}")
    else:
        negative.append(f"probability DOWN ({probability_down:.0%}) exceeds UP ({probability_up:.0%}) by {-edge:.0%}")
        negative.append(f"expected return {expected_return:.2%}")

    if market_regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
        aligned = (market_regime == MarketRegime.TRENDING_UP) == is_up
        (positive if aligned else negative).append(
            f"market regime is {market_regime.value} — {'aligned with' if aligned else 'against'} this direction"
        )
    elif market_regime == MarketRegime.HIGH_VOLATILITY:
        negative.append("market regime is HIGH_VOLATILITY — wider uncertainty than usual")

    if risk_level == RiskLevel.HIGH_RISK:
        negative.append("risk level is HIGH")
    elif risk_level == RiskLevel.LOW_RISK:
        positive.append("risk level is LOW")

    if liquidity_level == LiquidityLevel.HIGH:
        positive.append("liquidity is HIGH")
    elif liquidity_level == LiquidityLevel.LOW:
        negative.append("liquidity is LOW")

    decision_type = DecisionType.POSITIVE_SETUP if is_up else DecisionType.NEGATIVE_SETUP
    return Decision(decision_type, positive, negative)
