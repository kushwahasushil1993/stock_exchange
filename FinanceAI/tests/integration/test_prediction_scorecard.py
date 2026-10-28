"""
GET /api/v1/predictions/scorecard — the per-prediction "was the model
actually right?" view (src/api/routers/predictions.py).

Exercised against a real (in-memory) database rather than mocks, because
the part that can realistically break is the SQL: the outer join to
PredictionOutcome plus the conditional aggregates that have to count
resolved / correct / wrong / awaiting rows over the *whole* filtered set,
not just the page being returned.
"""
from datetime import date, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.routers.predictions import get_prediction_scorecard
from src.database.models import (
    Base, DecisionType, Prediction, PredictionOutcome, SignalType, Stock,
)

NOW = datetime.utcnow()
YESTERDAY = (NOW - timedelta(days=1)).date()
TOMORROW = (NOW + timedelta(days=1)).date()


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


def _prediction(stock_id: int, **overrides) -> Prediction:
    kwargs = dict(
        stock_id=stock_id,
        model_version="three_model_v2",
        signal=SignalType.BUY,
        confidence=0.61,
        current_price=100.0,
        decision=DecisionType.POSITIVE_SETUP,
        probability_up=0.58, probability_down=0.30, probability_flat=0.12,
        expected_return=0.02, expected_return_lower=0.005, expected_return_upper=0.035,
        target_price=104.0, stop_loss=97.0,
        target_trading_date=YESTERDAY,
        predicted_at=NOW - timedelta(days=2),
    )
    kwargs.update(overrides)
    return Prediction(**kwargs)


@pytest_asyncio.fixture
async def seeded(db):
    """
    Four predictions covering every state the endpoint has to distinguish:
      BHEL.NS  — UP called, went up      → RESOLVED / CORRECT
      SAIL.NS  — UP called, went down    → RESOLVED / WRONG
      NTPC.NS  — targets a future session → AWAITING_OUTCOME / PENDING
      IOC.NS   — legacy row, no target    → NOT_EVALUABLE
    """
    stocks = [
        Stock(symbol="BHEL.NS", company_name="Bharat Heavy Electricals"),
        Stock(symbol="SAIL.NS", company_name="Steel Authority of India"),
        Stock(symbol="NTPC.NS", company_name="NTPC Ltd"),
        Stock(symbol="IOC.NS",  company_name="Indian Oil Corporation"),
    ]
    db.add_all(stocks)
    await db.flush()
    bhel, sail, ntpc, ioc = stocks

    correct_pred = _prediction(bhel.id)
    wrong_pred = _prediction(sail.id)
    pending_pred = _prediction(ntpc.id, target_trading_date=TOMORROW, predicted_at=NOW)
    legacy_pred = _prediction(
        ioc.id, target_trading_date=None, decision=None,
        probability_up=None, probability_down=None, probability_flat=None,
        expected_return=None, expected_return_lower=None, expected_return_upper=None,
        model_version="ensemble_v1",
    )
    db.add_all([correct_pred, wrong_pred, pending_pred, legacy_pred])
    await db.flush()

    db.add_all([
        # Closed at 102.50 — up 2.5% from 100.00, and the 104.00 target was
        # never touched intraday (high 103.00): right call, target not met.
        PredictionOutcome(
            prediction_id=correct_pred.id,
            actual_open=100.5, actual_high=103.0, actual_low=99.8, actual_close=102.5,
            actual_return=0.025, actual_direction="UP",
            prediction_correct=True, prediction_error=0.005,
        ),
        # Closed at 96.00 — down 4%, and it broke the 97.00 stop on the way.
        PredictionOutcome(
            prediction_id=wrong_pred.id,
            actual_open=99.5, actual_high=99.9, actual_low=95.5, actual_close=96.0,
            actual_return=-0.04, actual_direction="DOWN",
            prediction_correct=False, prediction_error=0.06,
        ),
    ])
    await db.commit()
    return {"bhel": bhel, "sail": sail, "ntpc": ntpc, "ioc": ioc}


async def _call(db, **kwargs):
    params = dict(symbol=None, status="all", verdict="all", days=30, page=1, page_size=50)
    params.update(kwargs)
    return await get_prediction_scorecard(db=db, _user=None, **params)


@pytest.mark.asyncio
class TestScorecardStates:
    async def test_every_prediction_state_is_classified(self, db, seeded):
        resp = await _call(db)
        by_symbol = {i.symbol: i for i in resp.items}

        assert by_symbol["BHEL.NS"].status == "RESOLVED"
        assert by_symbol["BHEL.NS"].verdict == "CORRECT"
        assert by_symbol["SAIL.NS"].verdict == "WRONG"
        assert by_symbol["NTPC.NS"].status == "AWAITING_OUTCOME"
        assert by_symbol["NTPC.NS"].verdict == "PENDING"
        assert by_symbol["IOC.NS"].status == "NOT_EVALUABLE"

    async def test_pending_row_reports_no_actuals_rather_than_zeros(self, db, seeded):
        """The whole point of §81: an unevaluated prediction must not render as a 0% move."""
        ntpc = next(i for i in (await _call(db)).items if i.symbol == "NTPC.NS")
        assert ntpc.actual_close is None
        assert ntpc.actual_return is None
        assert ntpc.actual_direction is None
        assert ntpc.direction_correct is None
        assert ntpc.target_price_hit is None

    async def test_resolved_row_carries_the_full_before_and_after(self, db, seeded):
        bhel = next(i for i in (await _call(db)).items if i.symbol == "BHEL.NS")
        assert bhel.predicted_direction == "UP"
        assert bhel.price_at_prediction == 100.0
        assert bhel.expected_return == 0.02
        assert bhel.actual_close == 102.5
        assert bhel.actual_return == 0.025
        assert bhel.actual_direction == "UP"
        assert bhel.target_trading_date == YESTERDAY.isoformat()

    async def test_explanation_names_the_symbol_the_call_and_the_result(self, db, seeded):
        by_symbol = {i.symbol: i for i in (await _call(db)).items}
        assert "BHEL.NS UP" in by_symbol["BHEL.NS"].explanation
        assert "was right" in by_symbol["BHEL.NS"].explanation
        assert "was wrong" in by_symbol["SAIL.NS"].explanation
        assert "not been evaluated yet" in by_symbol["NTPC.NS"].explanation


@pytest.mark.asyncio
class TestScorecardComparisons:
    async def test_target_price_hit_is_false_when_the_high_never_reached_it(self, db, seeded):
        """Right direction is not the same as meeting the stated expectation."""
        bhel = next(i for i in (await _call(db)).items if i.symbol == "BHEL.NS")
        assert bhel.verdict == "CORRECT"
        assert bhel.target_price_hit is False   # high 103.00 < target 104.00
        assert bhel.stop_loss_hit is False      # low 99.80 > stop 97.00

    async def test_stop_loss_hit_is_true_when_the_low_broke_it(self, db, seeded):
        sail = next(i for i in (await _call(db)).items if i.symbol == "SAIL.NS")
        assert sail.stop_loss_hit is True       # low 95.50 <= stop 97.00
        assert sail.target_price_hit is False

    async def test_return_within_interval_tracks_model_bs_stated_bounds(self, db, seeded):
        by_symbol = {i.symbol: i for i in (await _call(db)).items}
        # +2.5% actual falls inside the stated [+0.5%, +3.5%] interval.
        assert by_symbol["BHEL.NS"].return_within_interval is True
        # -4.0% is nowhere near it.
        assert by_symbol["SAIL.NS"].return_within_interval is False

    async def test_legacy_row_without_an_interval_reports_null_not_false(self, db, seeded):
        ioc = next(i for i in (await _call(db)).items if i.symbol == "IOC.NS")
        assert ioc.return_within_interval is None
        assert ioc.expected_return is None


@pytest.mark.asyncio
class TestScorecardSummary:
    async def test_summary_counts_every_state(self, db, seeded):
        s = (await _call(db)).summary
        assert s.predictions_matched == 4
        assert s.resolved == 2
        assert s.correct == 1
        assert s.wrong == 1
        assert s.awaiting_outcome == 1
        assert s.not_evaluable == 1
        assert s.accuracy == 0.5

    async def test_accuracy_is_null_not_zero_when_nothing_is_resolved(self, db, seeded):
        s = (await _call(db, status="pending")).summary
        assert s.resolved == 0
        assert s.accuracy is None
        assert "no accuracy to report" in s.note

    async def test_summary_spans_the_whole_filter_not_just_the_page(self, db, seeded):
        """Paginating must not change the headline accuracy number."""
        page1 = await _call(db, page=1, page_size=1)
        assert len(page1.items) == 1
        assert page1.total == 4
        assert page1.summary.resolved == 2
        assert page1.summary.accuracy == 0.5

    async def test_mean_absolute_error_averages_only_resolved_rows(self, db, seeded):
        s = (await _call(db)).summary
        assert s.mean_absolute_error == pytest.approx((0.005 + 0.06) / 2, abs=1e-6)


@pytest.mark.asyncio
class TestScorecardFilters:
    async def test_symbol_filter_matches_the_exact_ticker(self, db, seeded):
        resp = await _call(db, symbol="BHEL.NS")
        assert [i.symbol for i in resp.items] == ["BHEL.NS"]
        assert resp.summary.predictions_matched == 1

    async def test_symbol_filter_matches_a_bare_ticker_without_the_suffix(self, db, seeded):
        """A user types "BHEL", not "BHEL.NS"."""
        resp = await _call(db, symbol="BHEL")
        assert [i.symbol for i in resp.items] == ["BHEL.NS"]

    async def test_verdict_filter_narrows_to_wrong_calls(self, db, seeded):
        resp = await _call(db, verdict="wrong")
        assert [i.symbol for i in resp.items] == ["SAIL.NS"]
        assert resp.summary.correct == 0
        assert resp.summary.wrong == 1

    async def test_status_resolved_excludes_pending_and_legacy_rows(self, db, seeded):
        resp = await _call(db, status="resolved")
        assert sorted(i.symbol for i in resp.items) == ["BHEL.NS", "SAIL.NS"]

    async def test_days_window_excludes_older_predictions(self, db, seeded):
        old = _prediction(seeded["bhel"].id, predicted_at=NOW - timedelta(days=90))
        db.add(old)
        await db.commit()

        assert (await _call(db, days=30)).summary.predictions_matched == 4
        assert (await _call(db, days=120)).summary.predictions_matched == 5
