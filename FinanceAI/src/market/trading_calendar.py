"""
NSE trading calendar — the authoritative source for "is this a trading
day" / "what's the next trading day", backing Prediction.target_trading_date
so a Friday prediction correctly targets Monday (or later, across a
holiday), never a naive calendar +1 day that lands on a weekend.

Two genuinely different sources of truth here, and this module is explicit
about which is which rather than blurring them:

  1. PAST dates: backfilled EMPIRICALLY from NIFTY 50's own actual price
     history (already in price_history — see backfill_from_market_data()).
     A weekday with no NIFTY bar really was a holiday: this is real
     observed data, not a guess, and it automatically gets every festival/
     lunar-calendar holiday correct for any year we already have NIFTY
     history for, with zero manual holiday-list maintenance.

  2. FUTURE dates: NIFTY has no bar yet, so there is nothing to observe.
     Only a small set of FIXED-DATE national holidays (never move year to
     year, and NSE has observed all of them consistently) are seeded ahead
     of time — Republic Day, Independence Day, Gandhi Jayanti, Christmas.
     Lunar-calendar festivals (Holi, Diwali, Eid, Ganesh Chaturthi,
     Dussehra, Mahashivratri, ...) shift every year and are DELIBERATELY
     NOT guessed at here — NSE publishes the official list every December
     for the following year (nseindia.com circulars), and until those
     dates are entered (via seed_holiday() below, or once that year's
     NIFTY data exists and gets backfilled after the fact), this module
     will incorrectly treat those specific dates as trading days. That is
     an honest, documented gap — not a silent wrong answer dressed up as
     a confident one.
"""
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select, text

from src.database.models import AsyncSessionFactory, PriceHistory, Stock, TradingCalendar

logger = logging.getLogger("trading_calendar")

NIFTY_SYMBOL = "^NSEI"

# Fixed-date national holidays NSE has consistently observed regardless of
# year — the only holidays this module will assert for a FUTURE date it
# hasn't empirically observed yet. (month, day, name)
FIXED_HOLIDAYS = [
    (1, 26, "Republic Day"),
    (8, 15, "Independence Day"),
    (10, 2, "Gandhi Jayanti"),
    (12, 25, "Christmas"),
]


async def backfill_from_market_data(exchange: str = "NSE") -> int:
    """
    For every date NIFTY has a stored price_history bar: mark it a trading
    day. For every WEEKDAY within NIFTY's stored date range that has no
    bar: mark it a holiday (empirically — this is what actually happened,
    not an inference). Idempotent (ON CONFLICT DO NOTHING) — safe to re-run
    after every scan.
    """
    async with AsyncSessionFactory() as db:
        stock = (await db.execute(select(Stock).where(Stock.symbol == NIFTY_SYMBOL))).scalar_one_or_none()
        if not stock:
            logger.warning("backfill_from_market_data: %s not in stocks yet — nothing to backfill from", NIFTY_SYMBOL)
            return 0

        dates = (
            await db.execute(
                select(PriceHistory.date).where(PriceHistory.stock_id == stock.id).order_by(PriceHistory.date)
            )
        ).scalars().all()
        if not dates:
            return 0

        traded = set(dates)
        start, end = min(dates), max(dates)

        rows = []
        d = start
        while d <= end:
            if d.weekday() < 5:  # weekday
                is_trading = d in traded
                rows.append({
                    "date": d, "exchange": exchange, "is_trading_day": is_trading,
                    "holiday_name": None if is_trading else "Observed non-trading day (empirical)",
                    "source": "empirical",
                })
            d += timedelta(days=1)

        if not rows:
            return 0

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        stmt = pg_insert(TradingCalendar).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "exchange"],
            set_={"is_trading_day": stmt.excluded.is_trading_day,
                  "holiday_name": stmt.excluded.holiday_name, "source": stmt.excluded.source},
            where=(TradingCalendar.source == "empirical") | (TradingCalendar.source.is_(None)),
        )
        await db.execute(stmt)
        await db.commit()
        return len(rows)


async def seed_fixed_holidays(year: int, exchange: str = "NSE") -> None:
    """Seeds this year's fixed-date holidays (see FIXED_HOLIDAYS) plus weekends, for dates not already known empirically."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    rows = []
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    fixed = {(m, day): name for m, day, name in FIXED_HOLIDAYS}
    while d <= end:
        if d.weekday() >= 5:
            rows.append({"date": d, "exchange": exchange, "is_trading_day": False,
                         "holiday_name": "Weekend", "source": "weekend"})
        elif (d.month, d.day) in fixed:
            rows.append({"date": d, "exchange": exchange, "is_trading_day": False,
                         "holiday_name": fixed[(d.month, d.day)], "source": "fixed_holiday"})
        d += timedelta(days=1)

    async with AsyncSessionFactory() as db:
        stmt = pg_insert(TradingCalendar).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["date", "exchange"])
        await db.execute(stmt)
        await db.commit()


async def seed_holiday(day: date, name: str, exchange: str = "NSE") -> None:
    """Manually record a specific holiday (e.g. from NSE's published circular for lunar-calendar festivals) — the honest way to fill the gap documented in this module's docstring, rather than guessing."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    async with AsyncSessionFactory() as db:
        stmt = pg_insert(TradingCalendar).values(
            date=day, exchange=exchange, is_trading_day=False, holiday_name=name, source="manual"
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "exchange"],
            set_={"is_trading_day": False, "holiday_name": name, "source": "manual"},
        )
        await db.execute(stmt)
        await db.commit()


async def is_trading_day(day: date, exchange: str = "NSE") -> bool:
    """
    True if `day` is a known trading day. For a date this calendar has no
    record of at all (a future lunar-holiday date, or simply never
    backfilled), falls back to "weekday" as a best-effort default rather
    than raising — but logs a warning, since that default is exactly the
    kind of undetected festival holiday this module's docstring flags.
    """
    async with AsyncSessionFactory() as db:
        row = (
            await db.execute(
                select(TradingCalendar.is_trading_day)
                .where(TradingCalendar.date == day, TradingCalendar.exchange == exchange)
            )
        ).scalar_one_or_none()
    if row is not None:
        return row
    logger.warning(
        "is_trading_day(%s): no calendar record — falling back to weekday check only. "
        "If this is a lunar-calendar festival holiday, it will be WRONGLY treated as a "
        "trading day until seeded via seed_holiday().", day,
    )
    return day.weekday() < 5


async def next_trading_day(after: date, exchange: str = "NSE") -> date:
    """The next trading day strictly after `after` — used for Prediction.target_trading_date, never a naive +1 calendar day."""
    d = after + timedelta(days=1)
    for _ in range(14):  # 14-day search cap — real gaps (even Diwali+weekend) never exceed a few days
        if await is_trading_day(d, exchange):
            return d
        d += timedelta(days=1)
    logger.error("next_trading_day(%s): no trading day found within 14 days — returning weekday fallback %s", after, d)
    return d


async def previous_trading_day(before: date, exchange: str = "NSE") -> date:
    d = before - timedelta(days=1)
    for _ in range(14):
        if await is_trading_day(d, exchange):
            return d
        d -= timedelta(days=1)
    return d
