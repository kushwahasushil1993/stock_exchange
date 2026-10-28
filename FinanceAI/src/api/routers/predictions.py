"""
Predictions router — trigger model inference and return ranked signals.
Supports both cached (pre-computed) and live on-demand prediction.
"""
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, or_, and_, case
from sqlalchemy.orm import aliased

from src.database.models import get_db, DecisionType, LiquidityLevel, MarketRegime, Prediction, RiskLevel, SignalType, Stock
from src.database.models import redis_cache
from src.api.routers.auth import get_current_user, User
from config.settings import settings
from src.recommendation.engine import TradingAgents

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────────────────
class ThreeModelOut(BaseModel):
    """
    The three-model split's own fields (§20-21) — kept as a separate nested
    object rather than flattened, so it's obvious at a glance which
    predictions actually went through Model A/B/C + the decision engine
    (model_type == "three_model_v2") vs the legacy single-composite path,
    for which every field here is null rather than a fabricated 0.
    """
    decision:              Optional[str] = None
    decision_factors:      Optional[dict] = None
    probability_up:        Optional[float] = None
    probability_down:      Optional[float] = None
    probability_flat:      Optional[float] = None
    expected_return:       Optional[float] = None
    expected_return_lower: Optional[float] = None
    expected_return_upper: Optional[float] = None
    risk_level:            Optional[str] = None
    liquidity_level:       Optional[str] = None
    market_regime:         Optional[str] = None
    target_trading_date:   Optional[str] = None
    model_version:         Optional[str] = None
    feature_version:       Optional[str] = None


def _three_model_out(pred: Prediction) -> Optional[ThreeModelOut]:
    """None for legacy predictions (decision is only ever set on the three-model path) — never fabricate these fields for a row the three-model system didn't actually produce."""
    if pred.decision is None:
        return None
    return ThreeModelOut(
        decision=pred.decision.value,
        decision_factors=pred.decision_factors,
        probability_up=pred.probability_up, probability_down=pred.probability_down, probability_flat=pred.probability_flat,
        expected_return=pred.expected_return, expected_return_lower=pred.expected_return_lower, expected_return_upper=pred.expected_return_upper,
        risk_level=pred.risk_level.value if pred.risk_level else None,
        liquidity_level=pred.liquidity_level.value if pred.liquidity_level else None,
        market_regime=pred.market_regime.value if pred.market_regime else None,
        target_trading_date=pred.target_trading_date.isoformat() if pred.target_trading_date else None,
        model_version=pred.model_version, feature_version=pred.feature_version,
    )


class PredictionOut(BaseModel):
    symbol:              str
    signal:              str
    confidence:          float
    composite_score:     float
    technical_score:     float
    fundamental_score:   float
    sentiment_score:     float
    current_price:       float
    target_price:        float
    stop_loss:           float
    risk_reward_ratio:   float
    predicted_return_5d: float
    feature_importance:  dict
    predicted_at:        str
    three_model:         Optional[ThreeModelOut] = None

    class Config:
        from_attributes = True


class TopPicksResponse(BaseModel):
    buy_picks:       List[PredictionOut]
    sell_picks:      List[PredictionOut]
    total_analyzed:  int
    generated_at:    str


class StockRowOut(BaseModel):
    symbol:              str
    company_name:        str
    sector:               Optional[str]
    industry:             Optional[str]
    market_cap:           Optional[float]
    is_fno_eligible:      bool
    signal:               Optional[str]
    confidence:           Optional[float]
    composite_score:      Optional[float]
    current_price:        Optional[float]
    target_price:         Optional[float]
    stop_loss:             Optional[float]
    predicted_return_5d:  Optional[float]
    predicted_at:          Optional[str]
    three_model:           Optional[ThreeModelOut] = None


class AllStocksResponse(BaseModel):
    items:       List[StockRowOut]
    total:       int
    page:        int
    page_size:   int


class TradeAgentRequest(BaseModel):
    portfolio_symbols: List[str]
    top_n_buys: int = 5


class TradeActionOut(BaseModel):
    symbol: str
    action: str
    confidence: float
    expected_return_5d: float
    composite_score: float
    rationale: str
    rank: int


class TradeAgentPlanResponse(BaseModel):
    sell_actions: List[TradeActionOut]
    buy_actions: List[TradeActionOut]
    total_analyzed: int
    generated_at: str


async def _fetch_latest_prediction_rows(
    db: AsyncSession,
    limit: int = 100,
):
    subq = (
        select(Prediction.stock_id, func.max(Prediction.predicted_at).label("latest"))
        .group_by(Prediction.stock_id)
        .subquery()
    )
    result = await db.execute(
        select(Prediction, Stock.symbol)
        .join(Stock, Prediction.stock_id == Stock.id)
        .join(subq, (Prediction.stock_id == subq.c.stock_id) &
                    (Prediction.predicted_at == subq.c.latest))
        .where(Prediction.confidence >= settings.prediction_confidence_threshold)
        .order_by(desc(Prediction.composite_score))
        .limit(limit)
    )
    return result.fetchall()


# ─── Endpoints ───────────────────────────────────────────────────────────────
@router.get("/top-picks", response_model=TopPicksResponse)
async def get_top_picks(
    top_n:    int = Query(default=10, ge=1, le=50),
    db:       AsyncSession = Depends(get_db),
    _user:    User = Depends(get_current_user),
):
    """
    Return top N BUY and SELL signals from latest model predictions.
    Results are cached in Redis for 15 minutes.
    """
    cache_key = f"top_picks:{top_n}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return json.loads(cached)

    rows = await _fetch_latest_prediction_rows(db=db, limit=100)

    from datetime import datetime
    buy_picks  = []
    sell_picks = []

    for pred, sym in rows:
        p = PredictionOut(
            symbol              = sym,
            signal              = pred.signal.value,
            confidence          = pred.confidence,
            composite_score     = pred.composite_score or 0,
            technical_score     = pred.technical_score or 0,
            fundamental_score   = pred.fundamental_score or 0,
            sentiment_score     = pred.sentiment_score or 0,
            current_price       = pred.current_price or 0,
            target_price        = pred.target_price or 0,
            stop_loss           = pred.stop_loss or 0,
            risk_reward_ratio   = pred.risk_reward_ratio or 0,
            predicted_return_5d = pred.predicted_return_5d or pred.predicted_return_1d or 0,
            feature_importance  = pred.feature_importance or {},
            predicted_at        = pred.predicted_at.isoformat(),
            three_model         = _three_model_out(pred),
        )
        if pred.signal.value == "BUY":
            buy_picks.append(p)
        elif pred.signal.value == "SELL":
            sell_picks.append(p)

    response = TopPicksResponse(
        buy_picks      = buy_picks[:top_n],
        sell_picks     = sell_picks[:top_n],
        total_analyzed = len(rows),
        generated_at   = datetime.utcnow().isoformat(),
    )

    await redis_cache.set(cache_key, response.model_dump_json(), ttl=900)
    return response


@router.get("/all", response_model=AllStocksResponse)
async def list_all_stocks(
    page:        int            = Query(default=1, ge=1),
    page_size:   int            = Query(default=50, ge=1, le=200),
    signal:      Optional[str]  = Query(default=None, pattern="^(BUY|SELL|HOLD)$"),
    search:      Optional[str]  = Query(default=None, description="Filter by symbol or company name"),
    db:          AsyncSession   = Depends(get_db),
    _user:       User           = Depends(get_current_user),
):
    """
    Every stock in the database (not just top picks), each with its latest
    prediction if one exists. Backs the "All Stocks" page — paginated since
    this can be thousands of rows after a full universe scan.
    """
    latest_sub = (
        select(Prediction.stock_id, func.max(Prediction.predicted_at).label("latest"))
        .group_by(Prediction.stock_id)
        .subquery()
    )
    latest_pred = aliased(Prediction)

    base_query = (
        select(Stock, latest_pred)
        .outerjoin(latest_sub, Stock.id == latest_sub.c.stock_id)
        .outerjoin(
            latest_pred,
            (latest_pred.stock_id == latest_sub.c.stock_id) & (latest_pred.predicted_at == latest_sub.c.latest),
        )
    )
    if signal:
        base_query = base_query.where(latest_pred.signal == SignalType(signal))
    if search:
        like = f"%{search}%"
        base_query = base_query.where(or_(Stock.symbol.ilike(like), Stock.company_name.ilike(like)))

    total = (await db.execute(select(func.count()).select_from(base_query.subquery()))).scalar_one()

    rows = (
        await db.execute(
            base_query
            .order_by(desc(latest_pred.composite_score).nulls_last(), Stock.symbol)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()

    items = [
        StockRowOut(
            symbol=stock.symbol,
            company_name=stock.company_name,
            sector=stock.sector,
            industry=stock.industry,
            market_cap=stock.market_cap,
            is_fno_eligible=stock.is_fno_eligible,
            signal=pred.signal.value if pred else None,
            confidence=pred.confidence if pred else None,
            composite_score=pred.composite_score if pred else None,
            current_price=pred.current_price if pred else None,
            target_price=pred.target_price if pred else None,
            stop_loss=pred.stop_loss if pred else None,
            predicted_return_5d=pred.predicted_return_5d if pred else None,
            predicted_at=pred.predicted_at.isoformat() if pred else None,
            three_model=_three_model_out(pred) if pred else None,
        )
        for stock, pred in rows
    ]

    return AllStocksResponse(items=items, total=total, page=page, page_size=page_size)


# ─── Single-symbol detail (§45-51) ────────────────────────────────────────────
class OutcomeOut(BaseModel):
    actual_close:        Optional[float] = None
    actual_return:       Optional[float] = None
    actual_direction:    Optional[str] = None
    prediction_correct:  Optional[bool] = None
    prediction_error:    Optional[float] = None
    evaluated_at:        Optional[str] = None


class LatestPredictionResponse(BaseModel):
    symbol:               str
    company_name:         str
    current_price:        Optional[float]
    signal:               str
    confidence:           float
    target_price:         Optional[float]
    stop_loss:            Optional[float]
    predicted_at:         str
    target_trading_date:  Optional[str]
    three_model:          Optional[ThreeModelOut]
    outcome:              Optional[OutcomeOut]


@router.get("/{symbol}/latest", response_model=LatestPredictionResponse)
async def get_latest_prediction_for_symbol(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    The single latest STORED prediction for one symbol, with full
    three-model detail (decision, calibrated probabilities, expected-return
    interval, risk/liquidity, market regime, decision factors) when it was
    produced by the three-model path — this is what backs StockDetails'
    "what the model actually predicted" panel, as distinct from
    /analyze/stock's live on-demand legacy analysis. Also includes the
    actual outcome (src/pipelines/outcome_tracker.py) once the target
    trading session has resolved, so a stale or wrong prediction is visible
    on the page, not just the confident-looking parts.
    """
    res = await db.execute(select(Stock).where(Stock.symbol == symbol))
    stock = res.scalar_one_or_none()
    if not stock:
        raise HTTPException(status_code=404, detail=f"{symbol} not found")

    pred = (
        await db.execute(
            select(Prediction).where(Prediction.stock_id == stock.id)
            .order_by(desc(Prediction.predicted_at)).limit(1)
        )
    ).scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail=f"No prediction stored yet for {symbol} — run a universe scan or trigger one for this symbol first.")

    from src.database.models import PredictionOutcome
    outcome_row = (
        await db.execute(select(PredictionOutcome).where(PredictionOutcome.prediction_id == pred.id))
    ).scalar_one_or_none()
    outcome = None
    if outcome_row:
        outcome = OutcomeOut(
            actual_close=outcome_row.actual_close, actual_return=outcome_row.actual_return,
            actual_direction=outcome_row.actual_direction, prediction_correct=outcome_row.prediction_correct,
            prediction_error=outcome_row.prediction_error,
            evaluated_at=outcome_row.evaluated_at.isoformat() if outcome_row.evaluated_at else None,
        )

    return LatestPredictionResponse(
        symbol=symbol, company_name=stock.company_name, current_price=pred.current_price,
        signal=pred.signal.value, confidence=pred.confidence,
        target_price=pred.target_price, stop_loss=pred.stop_loss,
        predicted_at=pred.predicted_at.isoformat(),
        target_trading_date=pred.target_trading_date.isoformat() if pred.target_trading_date else None,
        three_model=_three_model_out(pred), outcome=outcome,
    )


# ─── Honest performance reporting (§69/§81) ───────────────────────────────────
class FoldMetricOut(BaseModel):
    fold_number:               int
    validation_start:          str
    validation_end:            str
    validation_rows:           int
    accuracy:                  Optional[float] = None
    majority_class_accuracy:   Optional[float] = None
    brier_score:                Optional[float] = None
    log_loss:                   Optional[float] = None


class ModelHonestyReport(BaseModel):
    model_id:                    int
    model_name:                  str
    version:                     str
    trained_at:                  str
    is_active:                   bool
    training_period_start:       Optional[str]
    training_period_end:         Optional[str]
    folds:                       List[FoldMetricOut]
    calibration_metrics:         Optional[dict]
    return_model_metrics:        Optional[dict]
    # Computed here (not stored anywhere) by comparing the average fold
    # accuracy to the average fold majority-class baseline across the SAME
    # folds — the one honest headline number, rather than a bare accuracy
    # that looks fine in isolation but doesn't beat "always guess the same
    # class."
    beats_baseline_on_average:  Optional[bool] = None


class OutcomeSummaryOut(BaseModel):
    resolved_predictions:      int
    correct:                    int
    accuracy:                   Optional[float] = None
    mean_absolute_error:        Optional[float] = None
    note:                        str


class PerformanceResponse(BaseModel):
    latest_model:      Optional[ModelHonestyReport]
    outcome_summary:   OutcomeSummaryOut


@router.get("/performance", response_model=PerformanceResponse)
async def get_prediction_performance(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    The honest out-of-sample performance report (§69/§81): the latest
    trained model's walk-forward fold-by-fold metrics compared against the
    majority-class baseline computed on those SAME folds (never a bare
    accuracy number — beating a naive "always guess the same class"
    baseline is the actual bar, per §33), plus how live predictions have
    actually resolved so far via PredictionOutcome. This is the platform's
    single place required to "tell the truth when it performs poorly"
    (§81) instead of only ever surfacing a confident-looking number — if
    resolved_predictions is 0, that's reported as exactly that, not hidden
    behind a fabricated accuracy.
    """
    from src.database.models import ModelRegistry, ModelTrainingRun, PredictionOutcome

    model_row = (
        await db.execute(select(ModelRegistry).order_by(desc(ModelRegistry.trained_at)).limit(1))
    ).scalar_one_or_none()

    latest_model = None
    if model_row:
        fold_rows = (
            await db.execute(
                select(ModelTrainingRun).where(ModelTrainingRun.model_id == model_row.id)
                .order_by(ModelTrainingRun.fold_number)
            )
        ).scalars().all()
        folds = []
        accs, baselines = [], []
        for fr in fold_rows:
            m = fr.metrics or {}
            if m.get("accuracy") is not None:
                accs.append(m["accuracy"])
            if m.get("majority_class_accuracy") is not None:
                baselines.append(m["majority_class_accuracy"])
            folds.append(FoldMetricOut(
                fold_number=fr.fold_number,
                validation_start=fr.validation_start.isoformat(), validation_end=fr.validation_end.isoformat(),
                validation_rows=fr.validation_rows or 0,
                accuracy=m.get("accuracy"), majority_class_accuracy=m.get("majority_class_accuracy"),
                brier_score=m.get("brier_score"), log_loss=m.get("log_loss"),
            ))
        beats = None
        if accs and baselines and len(accs) == len(baselines):
            beats = (sum(accs) / len(accs)) > (sum(baselines) / len(baselines))

        latest_model = ModelHonestyReport(
            model_id=model_row.id, model_name=model_row.model_name, version=model_row.version,
            trained_at=model_row.trained_at.isoformat(), is_active=model_row.is_active,
            training_period_start=model_row.training_period_start.isoformat() if model_row.training_period_start else None,
            training_period_end=model_row.training_period_end.isoformat() if model_row.training_period_end else None,
            folds=folds, calibration_metrics=model_row.calibration_metrics,
            return_model_metrics=(model_row.metrics or {}).get("return_model"),
            beats_baseline_on_average=beats,
        )

    outcome_rows = (await db.execute(select(PredictionOutcome))).scalars().all()
    resolved = len(outcome_rows)
    correct = sum(1 for o in outcome_rows if o.prediction_correct)
    errors = [o.prediction_error for o in outcome_rows if o.prediction_error is not None]
    outcome_summary = OutcomeSummaryOut(
        resolved_predictions=resolved,
        correct=correct,
        accuracy=round(correct / resolved, 4) if resolved else None,
        mean_absolute_error=round(sum(errors) / len(errors), 5) if errors else None,
        note=(
            "No predictions have reached their target trading date yet — there is "
            "nothing to evaluate. This is expected right after a fresh scan, not an error."
            if resolved == 0 else
            f"Based on {resolved} resolved prediction(s) so far."
        ),
    )

    return PerformanceResponse(latest_model=latest_model, outcome_summary=outcome_summary)


# ─── "What should I buy today" (§ explicit product ask) ───────────────────────
DISCLAIMER = (
    "This is a probabilistic model output, not financial advice and not a "
    "guarantee of profit. Backtested on real 2023-2026 data, this system's "
    "trading strategy UNDERPERFORMED simply buying and holding NIFTY over "
    "the same period (+2.4% vs +16.2%, Sharpe 0.12 vs 0.53) — see "
    "/predictions/performance and the Model Performance page. Consult a "
    "licensed financial advisor before making any investment decision."
)

LONG_TERM_FORECAST_NOTE = (
    "A 1-year individual-stock price forecast is not something this system "
    "produces, and it is not something any system trained on historical "
    "price/technical data honestly can: over a year, a stock's return is "
    "dominated by information that does not exist yet today — quarterly "
    "results, management decisions, competitive moves, regulatory changes, "
    "macro shocks. No amount of historical-pattern analysis reveals those "
    "in advance. This platform's own models target the NEXT trading "
    "session (1 day) specifically because that is the only horizon where "
    "the input features (price/volume history) are still plausibly "
    "informative — and even at that short horizon, walk-forward validation "
    "found no reliable edge over a naive baseline (see /predictions/performance). "
    "Professional equity research 1-year price targets carry wide error "
    "bars and are frequently wrong for the same fundamental reason; treat "
    "any 1-year stock prediction — from this system or elsewhere — with "
    "that in mind."
)


class BestSignalOut(BaseModel):
    symbol:                     str
    company_name:               str
    direction:                  str  # "UP" or "DOWN" — only POSITIVE_SETUP (UP) is surfaced by the default endpoint today
    current_price:              float
    target_price:               float
    stop_loss:                  float
    expected_return_pct:        float
    expected_return_lower_pct:  float
    expected_return_upper_pct:  float
    probability_up:             float
    probability_down:           float
    risk_level:                 str
    liquidity_level:            str
    market_regime:              Optional[str]
    decision_factors:           dict
    target_trading_date:        Optional[str]
    predicted_at:                str
    fno_rationale:                Optional[str] = None  # e.g. "[BUY FUTURE] ..." — strategy name is embedded in this string as persisted by universe_scan.py


class BestSignalResponse(BaseModel):
    generated_at:              str
    horizon_days:              int
    asset_type:                str
    signals:                   List[BestSignalOut]
    count:                     int
    note:                      str
    long_term_forecast_note:   str
    disclaimer:                str


@router.get("/best-signal", response_model=BestSignalResponse)
async def get_best_signal(
    asset_type: str = Query(default="equity", pattern="^(equity|fno)$"),
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    "Which stock should I buy today" — the platform's own decision engine's
    current POSITIVE_DIRECTIONAL_SETUP calls (Model A+B+C combined,
    calibrated probabilities, real expected-return interval), ranked by
    expected return. NOT a guaranteed pick: an EMPTY list is the common,
    correct answer on most days (§44/§72 — NO_CLEAR_SIGNAL is normal), and
    every response carries the same honesty disclaimer the rest of this
    platform does. asset_type=fno additionally requires F&O eligibility and
    attaches the derived futures/options strategy where one exists.

    Only predictions whose target_trading_date hasn't already passed are
    considered — a POSITIVE_SETUP call for a session that's already over
    isn't "today's" signal, it's stale.
    """
    from datetime import date as date_cls, datetime
    from src.database.models import FNORecommendation
    from src.market.trading_calendar import is_trading_day

    today = date_cls.today()

    latest_sub = (
        select(Prediction.stock_id, func.max(Prediction.predicted_at).label("latest"))
        .group_by(Prediction.stock_id)
        .subquery()
    )
    query = (
        select(Prediction, Stock)
        .join(Stock, Prediction.stock_id == Stock.id)
        .join(latest_sub, (Prediction.stock_id == latest_sub.c.stock_id) & (Prediction.predicted_at == latest_sub.c.latest))
        .where(Prediction.decision == DecisionType.POSITIVE_SETUP)
        .where(Prediction.target_trading_date >= today)
    )
    if asset_type == "fno":
        query = query.where(Stock.is_fno_eligible.is_(True))
    query = query.order_by(desc(Prediction.expected_return)).limit(limit)

    rows = (await db.execute(query)).all()

    fno_by_symbol = {}
    if asset_type == "fno" and rows:
        symbols = [stock.symbol for _pred, stock in rows]
        fno_rows = (
            await db.execute(
                select(FNORecommendation)
                .where(FNORecommendation.symbol.in_(symbols))
                .order_by(desc(FNORecommendation.created_at))
            )
        ).scalars().all()
        for r in fno_rows:
            fno_by_symbol.setdefault(r.symbol, r)  # first (most recent) wins

    signals = []
    for pred, stock in rows:
        fno_rec = fno_by_symbol.get(stock.symbol)
        signals.append(BestSignalOut(
            symbol=stock.symbol, company_name=stock.company_name,
            direction="UP", current_price=pred.current_price,
            target_price=pred.target_price, stop_loss=pred.stop_loss,
            expected_return_pct=round(pred.expected_return * 100, 4),
            expected_return_lower_pct=round(pred.expected_return_lower * 100, 4),
            expected_return_upper_pct=round(pred.expected_return_upper * 100, 4),
            probability_up=pred.probability_up, probability_down=pred.probability_down,
            risk_level=pred.risk_level.value if pred.risk_level else "UNKNOWN",
            liquidity_level=pred.liquidity_level.value if pred.liquidity_level else "UNKNOWN",
            market_regime=pred.market_regime.value if pred.market_regime else None,
            decision_factors=pred.decision_factors or {},
            target_trading_date=pred.target_trading_date.isoformat() if pred.target_trading_date else None,
            predicted_at=pred.predicted_at.isoformat(),
            fno_rationale=fno_rec.rationale if fno_rec else None,
        ))

    trading_today = await is_trading_day(today)
    note = (
        f"Found {len(signals)} stock(s) the decision engine currently rates a positive setup."
        if signals else
        "No stock currently clears the decision engine's bar for a positive setup — this is "
        "the common, expected result (see disclaimer), not a system failure. Check back after "
        "the next scan, or see /predictions/all for every stock's current read, including "
        "the (usually large) NO_CLEAR_SIGNAL majority."
    )
    if not trading_today:
        note += " Note: today is not an NSE trading day."

    return BestSignalResponse(
        generated_at=datetime.utcnow().isoformat(), horizon_days=1, asset_type=asset_type,
        signals=signals, count=len(signals), note=note,
        long_term_forecast_note=LONG_TERM_FORECAST_NOTE, disclaimer=DISCLAIMER,
    )


# ─── Prediction scorecard: what was predicted vs. what actually happened ─────
# This is the per-prediction counterpart to /performance (which only ever
# reports aggregates). §37/§68-69/§81: a prediction is only meaningful once
# it has been checked against the session it was made for, and that check
# has to be inspectable row by row — "the model said BHEL.NS would go UP
# for Tuesday; Tuesday closed -0.6%, so it was wrong" — not just averaged
# away into one accuracy number.
#
# Every scored field here comes from PredictionOutcome, which is written
# once, after the fact, by src/pipelines/outcome_tracker.py. Nothing on
# this endpoint recomputes or back-fills an outcome, so a row that hasn't
# been evaluated yet reports AWAITING_OUTCOME rather than a guess.
class ScorecardItemOut(BaseModel):
    prediction_id:           int
    symbol:                  str
    company_name:            str
    predicted_at:            str
    target_trading_date:     Optional[str] = None
    status:                  str  # RESOLVED | AWAITING_OUTCOME | NOT_EVALUABLE
    verdict:                 str  # CORRECT | WRONG | PENDING | NOT_EVALUABLE

    # ── What the model committed to, before the session happened ──
    predicted_direction:     Optional[str] = None  # UP | DOWN | FLAT
    decision:                Optional[str] = None
    signal:                  Optional[str] = None
    confidence:              Optional[float] = None
    price_at_prediction:     Optional[float] = None
    expected_return:         Optional[float] = None
    expected_return_lower:   Optional[float] = None
    expected_return_upper:   Optional[float] = None
    target_price:            Optional[float] = None
    stop_loss:               Optional[float] = None
    probability_up:          Optional[float] = None
    probability_down:        Optional[float] = None
    probability_flat:        Optional[float] = None
    model_version:           Optional[str] = None

    # ── What the market actually did on that session ──
    actual_open:             Optional[float] = None
    actual_high:             Optional[float] = None
    actual_low:              Optional[float] = None
    actual_close:            Optional[float] = None
    actual_return:           Optional[float] = None
    actual_direction:        Optional[str] = None
    evaluated_at:            Optional[str] = None

    # ── The comparison ──
    direction_correct:       Optional[bool] = None
    return_error:            Optional[float] = None  # |expected_return - actual_return|
    return_within_interval:  Optional[bool] = None   # did actual land inside Model B's stated interval
    target_price_hit:        Optional[bool] = None   # did the session's range reach the target price
    stop_loss_hit:           Optional[bool] = None   # did it reach the stop loss instead
    explanation:             str


class ScorecardSummaryOut(BaseModel):
    predictions_matched:   int
    resolved:              int
    correct:               int
    wrong:                 int
    awaiting_outcome:      int
    not_evaluable:         int
    accuracy:              Optional[float] = None
    mean_absolute_error:   Optional[float] = None
    flat_threshold:        float
    note:                  str


class ScorecardResponse(BaseModel):
    items:        List[ScorecardItemOut]
    summary:      ScorecardSummaryOut
    total:        int
    page:         int
    page_size:    int
    generated_at: str
    disclaimer:   str


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:+.2f}%"


def _rupees(x: Optional[float]) -> str:
    return "—" if x is None else f"₹{x:,.2f}"


def _scorecard_explanation(
    symbol: str, pred: Prediction, outcome, predicted_dir: Optional[str], status: str,
) -> str:
    """One plain sentence a human can check, in the same shape for every row."""
    made_on = pred.predicted_at.date().isoformat()
    target = pred.target_trading_date.isoformat() if pred.target_trading_date else "an unrecorded session"
    called = (
        f"On {made_on} the model called {symbol} {predicted_dir or 'UNKNOWN'} for the {target} "
        f"session (from {_rupees(pred.current_price)}"
        + (f", expecting {_pct(pred.expected_return)}" if pred.expected_return is not None else "")
        + ")"
    )

    if status == "NOT_EVALUABLE":
        return (
            f"{called}. This row was written without a target trading date, so there is no "
            "session to score it against — it can never be marked right or wrong."
        )
    if status == "AWAITING_OUTCOME":
        return (
            f"{called}. That session has not been evaluated yet, so there is nothing to compare "
            "against — the expected state until the outcome tracker runs after the close."
        )

    actual = (
        f"It actually went {outcome.actual_direction} — closed {_rupees(outcome.actual_close)}, "
        f"{_pct(outcome.actual_return)}"
    )
    if outcome.prediction_correct is True:
        return f"{called}. {actual}. The direction call was right."
    if outcome.prediction_correct is False:
        return f"{called}. {actual}. The direction call was wrong."
    return f"{called}. {actual}. No direction could be derived from this row, so it is unscored."


@router.get("/scorecard", response_model=ScorecardResponse)
async def get_prediction_scorecard(
    symbol:    Optional[str] = Query(default=None, description="Filter to one stock, e.g. BHEL.NS (BHEL also matches)"),
    status:    str = Query(default="all", pattern="^(all|resolved|pending)$"),
    verdict:   str = Query(default="all", pattern="^(all|correct|wrong)$"),
    days:      int = Query(default=30, ge=1, le=365, description="Only predictions made in the last N days"),
    page:      int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db:        AsyncSession = Depends(get_db),
    _user:     User = Depends(get_current_user),
):
    """
    Prediction-by-prediction scorecard: for each stored prediction, what the
    model said *before* the session, what the market actually did on that
    session, and whether the call held up.

    A prediction made today targets the next trading session, so it shows up
    as AWAITING_OUTCOME until that session closes and
    src/pipelines/outcome_tracker.py has evaluated it (Airflow runs it daily
    before the next scan; POST /predictions/scorecard/evaluate forces it now).
    Nothing here fabricates an outcome for an unevaluated row.

    `summary` is computed over everything the filters match, not just the
    current page, so the accuracy number doesn't drift as you paginate.
    """
    from datetime import datetime, timedelta

    from src.database.models import PredictionOutcome
    from src.pipelines.outcome_tracker import FLAT_THRESHOLD, _predicted_direction

    cutoff = datetime.utcnow() - timedelta(days=days)
    conds = [Prediction.predicted_at >= cutoff]

    if symbol:
        s = symbol.strip()
        conds.append(or_(
            Stock.symbol.ilike(s),
            Stock.symbol.ilike(f"{s}.%"),          # "BHEL" should still find BHEL.NS
            Stock.company_name.ilike(f"%{s}%"),
        ))
    if status == "resolved":
        conds.append(PredictionOutcome.id.isnot(None))
    elif status == "pending":
        conds.append(PredictionOutcome.id.is_(None))
    if verdict == "correct":
        conds.append(PredictionOutcome.prediction_correct.is_(True))
    elif verdict == "wrong":
        conds.append(PredictionOutcome.prediction_correct.is_(False))

    def _scoped(stmt):
        return (
            stmt.join(Stock, Prediction.stock_id == Stock.id)
            .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
            .where(and_(*conds))
        )

    # Aggregated in SQL, not in Python: a 30-day window over the full
    # ~2,300-stock universe is ~69k rows — far too many to pull just to count.
    matched, resolved, correct, wrong, awaiting, not_evaluable, mae = (
        await db.execute(_scoped(select(
            func.count(Prediction.id),
            func.count(PredictionOutcome.id),
            func.coalesce(func.sum(case((PredictionOutcome.prediction_correct.is_(True), 1), else_=0)), 0),
            func.coalesce(func.sum(case((PredictionOutcome.prediction_correct.is_(False), 1), else_=0)), 0),
            func.coalesce(func.sum(case(
                (and_(PredictionOutcome.id.is_(None), Prediction.target_trading_date.isnot(None)), 1), else_=0,
            )), 0),
            func.coalesce(func.sum(case((Prediction.target_trading_date.is_(None), 1), else_=0)), 0),
            func.avg(PredictionOutcome.prediction_error),
        )))
    ).one()

    rows = (
        await db.execute(
            _scoped(select(Prediction, Stock, PredictionOutcome))
            .order_by(desc(Prediction.predicted_at), Stock.symbol)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()

    items: List[ScorecardItemOut] = []
    for pred, stock, outcome in rows:
        predicted_dir = _predicted_direction(pred)

        if outcome is not None:
            item_status = "RESOLVED"
            item_verdict = (
                "CORRECT" if outcome.prediction_correct is True
                else "WRONG" if outcome.prediction_correct is False
                else "NOT_EVALUABLE"
            )
        elif pred.target_trading_date is None:
            item_status = item_verdict = "NOT_EVALUABLE"
        else:
            item_status, item_verdict = "AWAITING_OUTCOME", "PENDING"

        within = None
        if (
            outcome is not None and outcome.actual_return is not None
            and pred.expected_return_lower is not None and pred.expected_return_upper is not None
        ):
            within = pred.expected_return_lower <= outcome.actual_return <= pred.expected_return_upper

        # Only meaningful for a directional call: on a FLAT (or legacy,
        # directionless) row the target/stop aren't a side the model actually
        # took, so these stay null rather than being invented.
        target_hit = stop_hit = None
        if (
            outcome is not None and predicted_dir in ("UP", "DOWN")
            and outcome.actual_high is not None and outcome.actual_low is not None
        ):
            if pred.target_price is not None:
                target_hit = (
                    outcome.actual_high >= pred.target_price if predicted_dir == "UP"
                    else outcome.actual_low <= pred.target_price
                )
            if pred.stop_loss is not None:
                stop_hit = (
                    outcome.actual_low <= pred.stop_loss if predicted_dir == "UP"
                    else outcome.actual_high >= pred.stop_loss
                )

        items.append(ScorecardItemOut(
            prediction_id=pred.id,
            symbol=stock.symbol,
            company_name=stock.company_name,
            predicted_at=pred.predicted_at.isoformat(),
            target_trading_date=pred.target_trading_date.isoformat() if pred.target_trading_date else None,
            status=item_status,
            verdict=item_verdict,
            predicted_direction=predicted_dir,
            decision=pred.decision.value if pred.decision else None,
            signal=pred.signal.value if pred.signal else None,
            confidence=pred.confidence,
            price_at_prediction=pred.current_price,
            expected_return=pred.expected_return,
            expected_return_lower=pred.expected_return_lower,
            expected_return_upper=pred.expected_return_upper,
            target_price=pred.target_price,
            stop_loss=pred.stop_loss,
            probability_up=pred.probability_up,
            probability_down=pred.probability_down,
            probability_flat=pred.probability_flat,
            model_version=pred.model_version,
            actual_open=outcome.actual_open if outcome else None,
            actual_high=outcome.actual_high if outcome else None,
            actual_low=outcome.actual_low if outcome else None,
            actual_close=outcome.actual_close if outcome else None,
            actual_return=outcome.actual_return if outcome else None,
            actual_direction=outcome.actual_direction if outcome else None,
            evaluated_at=outcome.evaluated_at.isoformat() if outcome and outcome.evaluated_at else None,
            direction_correct=outcome.prediction_correct if outcome else None,
            return_error=outcome.prediction_error if outcome else None,
            return_within_interval=within,
            target_price_hit=target_hit,
            stop_loss_hit=stop_hit,
            explanation=_scorecard_explanation(stock.symbol, pred, outcome, predicted_dir, item_status),
        ))

    if resolved == 0:
        note = (
            f"{matched} prediction(s) match these filters, but none has been evaluated against a "
            "real session yet, so there is no accuracy to report. Right after a scan that is the "
            "correct answer, not an error — every prediction targets the NEXT trading session."
        )
    else:
        note = (
            f"{correct} of {resolved} evaluated prediction(s) called the direction correctly. "
            f"{awaiting} more are still waiting on their target session."
        )

    return ScorecardResponse(
        items=items,
        summary=ScorecardSummaryOut(
            predictions_matched=matched,
            resolved=resolved,
            correct=correct,
            wrong=wrong,
            awaiting_outcome=awaiting,
            not_evaluable=not_evaluable,
            accuracy=round(correct / resolved, 4) if resolved else None,
            mean_absolute_error=round(float(mae), 5) if mae is not None else None,
            flat_threshold=FLAT_THRESHOLD,
            note=note,
        ),
        total=matched,
        page=page,
        page_size=page_size,
        generated_at=datetime.utcnow().isoformat(),
        disclaimer=DISCLAIMER,
    )


@router.post("/scorecard/evaluate")
async def evaluate_pending_predictions(_user: User = Depends(get_current_user)):
    """
    Score every prediction whose target trading session has already passed and
    whose price data is already in the database.

    Airflow runs this exact step daily (dags/pipeline_dags.py → track_outcomes),
    so this endpoint exists only to force it now — e.g. a local run with no
    scheduler. It is idempotent: a PredictionOutcome row is inserted once and
    never rewritten (§38), so a second call evaluates nothing.
    """
    from src.pipelines.outcome_tracker import track_outcomes
    return await track_outcomes()


@router.post("/trade-agents/plan", response_model=TradeAgentPlanResponse)
async def trade_agents_plan(
    req: TradeAgentRequest,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Create trading actions using two agents:
    - SellOnDropAgent: SELL current portfolio symbols with downside signals
    - GrowthStockBuyerAgent: BUY high-growth candidates from the broader universe
    """
    from datetime import datetime

    rows = await _fetch_latest_prediction_rows(db=db, limit=300)
    prediction_payload = []
    for pred, sym in rows:
        prediction_payload.append({
            "symbol": sym,
            "signal": pred.signal.value,
            "confidence": pred.confidence,
            "composite_score": pred.composite_score or 0.0,
            "risk_reward_ratio": pred.risk_reward_ratio or 0.0,
            "predicted_return_5d": pred.predicted_return_5d or pred.predicted_return_1d or 0.0,
        })

    agents = TradingAgents()
    plan = agents.create_trade_plan(
        predictions=prediction_payload,
        portfolio_symbols=req.portfolio_symbols,
        top_n_buys=req.top_n_buys,
    )

    return TradeAgentPlanResponse(
        sell_actions=[
            TradeActionOut(
                symbol=a.symbol,
                action=a.action,
                confidence=a.confidence,
                expected_return_5d=a.expected_return_5d,
                composite_score=a.composite_score,
                rationale=a.rationale,
                rank=a.rank,
            ) for a in plan["sell_actions"]
        ],
        buy_actions=[
            TradeActionOut(
                symbol=a.symbol,
                action=a.action,
                confidence=a.confidence,
                expected_return_5d=a.expected_return_5d,
                composite_score=a.composite_score,
                rationale=a.rationale,
                rank=a.rank,
            ) for a in plan["buy_actions"]
        ],
        total_analyzed=len(rows),
        generated_at=datetime.utcnow().isoformat(),
    )


@router.post("/run/{symbol}", status_code=202)
async def trigger_prediction(
    symbol:           str,
    background_tasks: BackgroundTasks,
    _user:            User = Depends(get_current_user),
    db:               AsyncSession = Depends(get_db),
):
    """Trigger on-demand prediction for a symbol (async background task)."""
    background_tasks.add_task(_run_prediction_task, symbol, db)
    return {"message": f"Prediction job queued for {symbol}", "symbol": symbol}


@router.post("/run-universe", status_code=202)
async def trigger_universe_scan(
    background_tasks: BackgroundTasks,
    _user: User = Depends(get_current_user),
):
    """
    Fetch every NSE-listed stock (~2,300) and every F&O-eligible one among
    them (~210), then run the prediction engine on all of it — no
    parameters needed. Takes a couple of minutes, so it runs as a
    background task; poll GET /run-universe/status for progress, or just
    check /top-picks / /top-buys once it's done.
    """
    from src.pipelines.universe_scan import get_status, run_universe_scan

    current = await get_status()
    if current.get("state") == "running":
        raise HTTPException(
            status_code=409,
            detail={"message": "A universe scan is already running", "status": current},
        )

    background_tasks.add_task(run_universe_scan)
    return {
        "message": "Universe scan started — fetching all NSE stocks + F&O predictions in the background.",
        "status_url": "/api/v1/predictions/run-universe/status",
    }


@router.get("/run-universe/status")
async def universe_scan_status(_user: User = Depends(get_current_user)):
    """Progress of the most recent (or currently running) universe scan."""
    from src.pipelines.universe_scan import get_status
    return await get_status()


async def _run_prediction_task(symbol: str, db: AsyncSession):
    """Background task: fetch data → features → predict → store."""
    try:
        from src.ingestion.data_fetcher import YFinanceFetcher, NSEFetcher, NewsFetcher
        from src.features.technical import TechnicalFeatures
        from src.models.ensemble.predictor import ModelOrchestrator
        from src.sentiment.analyzer import NewsAggregator

        # Fetch OHLCV
        yf = YFinanceFetcher()
        ohlcv = yf.fetch_ohlcv(symbol, period="2y")
        if ohlcv.empty:
            logger.warning("No OHLCV data for %s", symbol)
            return

        # Fetch fundamentals
        fundamentals = yf.fetch_fundamentals(symbol)

        # Sentiment
        news_fetcher = NewsFetcher()
        import asyncio
        articles = await news_fetcher.fetch_all([symbol])
        aggregator = NewsAggregator()
        enriched = aggregator.process_articles(articles)
        sentiment_map = aggregator.aggregate_by_symbol(enriched, [symbol])
        sym_clean = symbol.replace(".NS", "")
        sentiment_score = sentiment_map.get(sym_clean, 0.0)

        # Predict
        orchestrator = ModelOrchestrator()
        result = orchestrator.predict(symbol, ohlcv, fundamentals, sentiment_score)

        # Persist
        stock_res = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = stock_res.scalar_one_or_none()
        if not stock:
            stock = Stock(symbol=symbol, company_name=fundamentals.get("company_name", symbol))
            db.add(stock)
            await db.flush()

        pred = Prediction(
            stock_id          = stock.id,
            model_version     = "1.0.0",
            signal            = result["signal"],
            confidence        = result["confidence"],
            current_price     = result.get("current_price"),
            composite_score   = result["composite_score"],
            technical_score   = result["technical_score"],
            fundamental_score = result["fundamental_score"],
            sentiment_score   = result["sentiment_score"],
            target_price      = result["target_price"],
            stop_loss         = result["stop_loss"],
            risk_reward_ratio = result["risk_reward_ratio"],
            predicted_return_1d = result.get("predicted_return_1d"),
            predicted_return_5d = result.get("predicted_return_5d"),
            feature_importance  = result.get("feature_importance"),
        )
        db.add(pred)
        await db.commit()
        logger.info("Prediction stored for %s: %s (%.2f)", symbol, result["signal"], result["confidence"])

    except Exception as exc:
        logger.error("Prediction task failed for %s: %s", symbol, exc)
