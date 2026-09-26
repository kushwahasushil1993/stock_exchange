"""
Fills PredictionOutcome for every Prediction whose target_trading_date has
already passed and has price_history for that date — this is what turns
"the model seems wrong a lot" into a provable, measurable number (§37,
§68-69), and it's the one thing that was completely missing before this
pass: predictions were written once and never checked against what
actually happened.

Never touches the Prediction row itself (§38, immutability) — only ever
inserts a new PredictionOutcome row, once, for a given prediction_id.

Run this BEFORE generating today's new predictions (see
dags/pipeline_dags.py) — yesterday's target_trading_date is today, so
today's freshly-scanned price_history is exactly what's needed to evaluate
it.
"""
import logging
from datetime import date as date_type
from typing import Dict, Optional

from sqlalchemy import and_, select

from src.database.models import (
    AsyncSessionFactory, Prediction, PredictionOutcome, PriceHistory, SignalType, init_db,
)

logger = logging.getLogger("outcome_tracker")

FLAT_THRESHOLD = 0.002  # same dead-zone used in training (src/pipelines/train_three_models.py) — kept consistent so "correct" means the same thing on both sides


def _actual_direction(actual_return: float) -> str:
    if actual_return > FLAT_THRESHOLD:
        return "UP"
    if actual_return < -FLAT_THRESHOLD:
        return "DOWN"
    return "FLAT"


def _predicted_direction(pred: Prediction) -> Optional[str]:
    """Direction implied by a prediction, whichever path produced it (three-model probabilities, or the legacy BUY/SELL/HOLD signal)."""
    if pred.probability_up is not None and pred.probability_down is not None:
        if pred.probability_up > pred.probability_down and pred.probability_up > (pred.probability_flat or 0):
            return "UP"
        if pred.probability_down > pred.probability_up and pred.probability_down > (pred.probability_flat or 0):
            return "DOWN"
        return "FLAT"
    if pred.signal == SignalType.BUY:
        return "UP"
    if pred.signal == SignalType.SELL:
        return "DOWN"
    return "FLAT"


async def track_outcomes(as_of: Optional[date_type] = None) -> Dict:
    """Evaluates every prediction whose target_trading_date <= as_of (default: today) and doesn't have an outcome yet."""
    import datetime as _dt
    as_of = as_of or _dt.datetime.utcnow().date()

    await init_db()
    evaluated = skipped_no_data = 0

    async with AsyncSessionFactory() as db:
        pending = (
            await db.execute(
                select(Prediction)
                .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
                .where(
                    and_(
                        Prediction.target_trading_date.isnot(None),
                        Prediction.target_trading_date <= as_of,
                        PredictionOutcome.id.is_(None),
                    )
                )
            )
        ).scalars().all()

        for pred in pending:
            bar = (
                await db.execute(
                    select(PriceHistory).where(
                        PriceHistory.stock_id == pred.stock_id,
                        PriceHistory.date == pred.target_trading_date,
                    )
                )
            ).scalar_one_or_none()
            if not bar or pred.current_price is None:
                skipped_no_data += 1
                continue

            actual_return = (bar.close - pred.current_price) / pred.current_price
            actual_dir = _actual_direction(actual_return)
            predicted_dir = _predicted_direction(pred)
            prediction_error = (
                abs(pred.expected_return - actual_return) if pred.expected_return is not None else None
            )

            db.add(PredictionOutcome(
                prediction_id=pred.id,
                actual_open=bar.open, actual_high=bar.high, actual_low=bar.low, actual_close=bar.close,
                actual_return=round(actual_return, 5), actual_direction=actual_dir,
                prediction_correct=(predicted_dir == actual_dir) if predicted_dir else None,
                prediction_error=round(prediction_error, 5) if prediction_error is not None else None,
            ))
            evaluated += 1

        if evaluated:
            await db.commit()

    logger.info("Outcome tracker: evaluated %d predictions, %d still waiting on price data", evaluated, skipped_no_data)
    return {"evaluated": evaluated, "skipped_no_data": skipped_no_data, "as_of": str(as_of)}


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(asyncio.run(track_outcomes()))
