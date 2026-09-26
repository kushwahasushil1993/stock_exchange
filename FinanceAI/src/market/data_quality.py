"""
OHLC validation and data-quality classification (VALID / SUSPECT / INVALID /
DUPLICATE / QUARANTINED — see DataQualityStatus). Applied per-bar before a
row is allowed into price_history.

Deliberately does NOT treat "unusual" as "invalid" — a stock split, a
circuit-limit day, or a genuine crash can legitimately move a price 20%+ in
a session; none of that violates basic OHLC arithmetic, so none of it gets
quarantined by this module. What DOES get quarantined is arithmetic that
cannot be real regardless of what happened in the market that day (High <
Low, negative prices, etc.) — see VALIDATORS below for the exact rules.

Rows that fail are NEVER discarded — they're written to
PriceHistoryQuarantine (raw values preserved) and logged to
DataQualityEvent, so a human can review whether a rule was too strict
rather than just losing the row.
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Any, Dict, List, Optional

from src.database.models import DataQualityStatus

logger = logging.getLogger("data_quality")


@dataclass
class ValidationResult:
    status: DataQualityStatus
    issues: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.status in (DataQualityStatus.VALID, DataQualityStatus.SUSPECT)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_ohlc_bar(open_: float, high: float, low: float, close: float, volume: Optional[float]) -> ValidationResult:
    """
    Hard (INVALID) rules — arithmetic that cannot be real:
      - any of O/H/L/C missing, non-finite, or <= 0
      - High < Low
      - High < Open, High < Close  (High must be the max)
      - Low > Open, Low > Close    (Low must be the min)
      - Volume < 0

    Soft (SUSPECT) rules — worth flagging but not rejecting, since a real
    corporate action or halt/circuit-limit day can legitimately produce
    these:
      - single-session move > 40% (well beyond NSE's typical circuit limits
        for most stocks, but real for T2T/no-band securities and for
        legitimate split/bonus-adjustment artifacts) — recorded, not
        rejected, exactly per "do not classify unusual movements as fraud
        merely because they are unusual"
      - zero volume on an otherwise-valid bar (illiquid security, or a
        provider gap — worth knowing about, not a reason to discard price
        data that's otherwise internally consistent)
    """
    issues: List[str] = []

    for name, val in (("open", open_), ("high", high), ("low", low), ("close", close)):
        if not _finite(val):
            issues.append(f"{name} is missing or non-finite")
        elif val <= 0:
            issues.append(f"{name} <= 0 ({val})")
    if issues:
        return ValidationResult(DataQualityStatus.INVALID, issues)

    if high < low:
        issues.append(f"high ({high}) < low ({low})")
    if high < open_:
        issues.append(f"high ({high}) < open ({open_})")
    if high < close:
        issues.append(f"high ({high}) < close ({close})")
    if low > open_:
        issues.append(f"low ({low}) > open ({open_})")
    if low > close:
        issues.append(f"low ({low}) > close ({close})")
    if volume is not None and volume < 0:
        issues.append(f"volume < 0 ({volume})")
    if issues:
        return ValidationResult(DataQualityStatus.INVALID, issues)

    soft_issues: List[str] = []
    if low > 0 and abs(close - open_) / open_ > 0.40:
        soft_issues.append(
            f"session move {100 * (close - open_) / open_:.1f}% exceeds 40% — "
            "could be a genuine corporate action/circuit-limit event, not necessarily bad data"
        )
    if volume == 0:
        soft_issues.append("zero volume")

    if soft_issues:
        return ValidationResult(DataQualityStatus.SUSPECT, soft_issues)
    return ValidationResult(DataQualityStatus.VALID, [])


def validate_bar_dict(bar: Dict[str, Any]) -> ValidationResult:
    """Convenience wrapper for a {"Open","High","Low","Close","Volume"}-shaped row (matches YFinanceFetcher's column names)."""
    return validate_ohlc_bar(
        bar.get("Open"), bar.get("High"), bar.get("Low"), bar.get("Close"), bar.get("Volume"),
    )
