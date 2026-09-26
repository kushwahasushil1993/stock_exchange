"""
Corporate action detection — from yfinance's own `Ticker.splits` /
`Ticker.dividends` (real source data reported by the exchange/registrar,
not inferred by looking for suspicious price jumps). yfinance's
auto_adjust=True (used everywhere in this codebase's OHLCV fetches)
already folds splits and dividends into the stored price series correctly
at the source — this module doesn't change that math, it just makes the
event auditable: CorporateAction rows answer "why did this stock's price
series show what it shows on this date" after the fact, instead of that
information being invisible once yfinance has already adjusted for it.

Bonus issues and rights issues are NOT separately reported by yfinance's
API — they typically show up bundled into its `splits` series (a bonus is
economically a stock split), so this module records what it can actually
observe from the data and doesn't fabricate the bonus/rights/merger/
demerger/symbol-change categories which have no corresponding real yfinance
field.
"""
import logging
from typing import List

from sqlalchemy import select

from src.database.models import AsyncSessionFactory, CorporateAction, CorporateActionType, Stock

logger = logging.getLogger("corporate_actions")


def detect_from_yfinance(symbol: str) -> List[dict]:
    """
    Runs synchronously (yfinance is sync) — call from a thread when used
    inside an async pipeline, same convention as the rest of this codebase's
    yfinance calls. Returns raw detected events; persist_corporate_actions()
    handles the DB write.
    """
    import yfinance as yf

    events: List[dict] = []
    try:
        ticker = yf.Ticker(symbol)
        splits = ticker.splits
        for action_date, ratio in splits.items():
            events.append({
                "action_type": CorporateActionType.SPLIT,
                "action_date": action_date.date(),
                "ratio": float(ratio),
                "details": {"raw_ratio": float(ratio)},
            })
        dividends = ticker.dividends
        for action_date, amount in dividends.items():
            events.append({
                "action_type": CorporateActionType.DIVIDEND,
                "action_date": action_date.date(),
                "ratio": float(amount),
                "details": {"amount_per_share": float(amount)},
            })
    except Exception as exc:
        logger.warning("Corporate action detection failed for %s: %s", symbol, exc)
    return events


async def persist_corporate_actions(symbol: str, events: List[dict]) -> int:
    """Idempotent: skips an event already recorded for the same stock/type/date."""
    if not events:
        return 0

    async with AsyncSessionFactory() as db:
        stock = (await db.execute(select(Stock).where(Stock.symbol == symbol))).scalar_one_or_none()
        if not stock:
            return 0

        existing = (
            await db.execute(
                select(CorporateAction.action_type, CorporateAction.action_date)
                .where(CorporateAction.stock_id == stock.id)
            )
        ).all()
        existing_keys = {(t, d) for t, d in existing}

        new_count = 0
        for e in events:
            key = (e["action_type"], e["action_date"])
            if key in existing_keys:
                continue
            db.add(CorporateAction(
                stock_id=stock.id, action_type=e["action_type"], action_date=e["action_date"],
                ratio=e.get("ratio"), details=e.get("details"), source="yfinance",
            ))
            new_count += 1
        if new_count:
            await db.commit()
        return new_count
