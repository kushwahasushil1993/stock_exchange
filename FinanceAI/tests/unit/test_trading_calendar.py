"""
§11/§57 — trading calendar. Covers what can genuinely be unit-tested
without a live database (the fixed-holiday data itself, and weekend
arithmetic); the DB-backed lookups (is_trading_day, next_trading_day,
backfill_from_market_data) need a real Postgres connection and belong in
tests/integration/ instead — asserting against a mocked AsyncSessionFactory
here would just test the mock, not the actual SQL.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.market.trading_calendar import FIXED_HOLIDAYS


class TestFixedHolidays:
    def test_every_entry_has_month_day_name(self):
        for entry in FIXED_HOLIDAYS:
            assert len(entry) == 3
            month, day, name = entry
            assert 1 <= month <= 12
            assert 1 <= day <= 31
            assert isinstance(name, str) and name

    def test_dates_are_valid_calendar_dates(self):
        """Every (month, day) must actually exist (e.g. no Feb 30) — constructing a date() raises otherwise."""
        for month, day, _ in FIXED_HOLIDAYS:
            date(2025, month, day)  # raises ValueError on an invalid calendar date

    def test_no_duplicate_dates(self):
        seen = [(m, d) for m, d, _ in FIXED_HOLIDAYS]
        assert len(seen) == len(set(seen))


class TestWeekendArithmetic:
    def test_known_saturday_and_sunday(self):
        assert date(2026, 9, 26).weekday() == 5  # Saturday
        assert date(2026, 9, 27).weekday() == 6  # Sunday

    def test_known_weekday(self):
        assert date(2026, 9, 24).weekday() < 5  # Thursday
