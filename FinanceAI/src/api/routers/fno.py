"""FNO router — options chain analysis, strategy recommendations."""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from src.api.routers.auth import get_current_user, User
from src.database.models import FNORecommendation, Prediction, SignalType, Stock, get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class FNOStrategyOut(BaseModel):
    symbol:        str
    strategy:      str
    asset_type:    str
    signal:        str
    confidence:    float
    iv:            Optional[float]
    pcr:           float
    oi_change_pct: float
    max_pain:      Optional[float]
    strike_price:  Optional[float]
    option_type:   Optional[str]
    expiry:        Optional[str]
    risk_level:    str
    rationale:     str
    option_data_source: str  # "nse" | "yahoo" | "none"


@router.get("/{symbol}/strategy", response_model=FNOStrategyOut)
async def fno_strategy(
    symbol: str,
    _user:  User = Depends(get_current_user),
):
    """
    Analyze F&O data for a symbol and return a strategy recommendation.
    Uses NSE option chain + stock prediction ensemble.
    """
    from src.api.routers.analyze import _fetch_real_option_chain
    from src.ingestion.data_fetcher import YFinanceFetcher
    from src.models.ensemble.predictor import ModelOrchestrator
    from src.recommendation.engine import FNOAnalyzer

    # Stock prediction
    yf = YFinanceFetcher()
    ohlcv = yf.fetch_ohlcv(symbol, period="1y")
    if ohlcv.empty:
        raise HTTPException(404, detail=f"No OHLCV data for {symbol}")

    fund = yf.fetch_fundamentals(symbol)
    pred = ModelOrchestrator().predict(symbol, ohlcv, fund)

    # Option chain — this used to be `asyncio.run(nse.fetch_option_chain(...))`
    # from inside an already-async endpoint, which always raises
    # RuntimeError ("asyncio.run() cannot be called from a running event
    # loop"), silently caught and falling back to Yahoo (which has no
    # options data at all for NSE-listed stocks) or empty dicts, every
    # single time this endpoint was ever called. Shares the real
    # NSE-then-Yahoo-then-neutral fetch logic with /analyze/fno instead of
    # duplicating (and re-breaking) it here.
    option_data = await _fetch_real_option_chain(symbol.replace(".NS", ""), symbol)

    analyzer = FNOAnalyzer()
    rec = analyzer.analyze(symbol, pred, option_data["chain"], option_data["oi"])

    return FNOStrategyOut(
        symbol       = rec.symbol,
        strategy     = rec.strategy,
        asset_type   = rec.asset_type,
        signal       = rec.signal,
        confidence   = rec.confidence,
        iv           = rec.iv,
        pcr          = rec.pcr,
        oi_change_pct= rec.oi_change_pct,
        max_pain     = rec.max_pain,
        strike_price = rec.strike_price,
        option_type  = rec.option_type,
        expiry       = rec.expiry,
        risk_level   = rec.risk_level,
        rationale    = rec.rationale,
        option_data_source = option_data["source"],
    )


class FnoRowOut(BaseModel):
    symbol:              str
    company_name:        str
    sector:              Optional[str]
    signal:              Optional[str]
    confidence:          Optional[float]
    current_price:       Optional[float]
    target_price:        Optional[float]
    stop_loss:            Optional[float]
    predicted_return_5d: Optional[float]
    fno_strategy:        Optional[str]
    fno_confidence:      Optional[float]
    pcr:                 Optional[float]
    iv:                  Optional[float]
    predicted_at:        Optional[str]


class AllFnoResponse(BaseModel):
    items:     List[FnoRowOut]
    total:     int
    page:      int
    page_size: int


@router.get("/all", response_model=AllFnoResponse)
async def list_all_fno(
    page:      int           = Query(default=1, ge=1),
    page_size: int           = Query(default=50, ge=1, le=200),
    signal:    Optional[str] = Query(default=None, pattern="^(BUY|SELL|HOLD)$"),
    search:    Optional[str] = Query(default=None, description="Filter by symbol or company name"),
    db:        AsyncSession  = Depends(get_db),
    _user:     User          = Depends(get_current_user),
):
    """
    Every F&O-eligible stock (NSE's ~210-symbol derivatives list), each with
    its latest stock prediction and latest FNORecommendation (from a
    universe scan) if either exists. Mirrors /predictions/all, scoped to
    Stock.is_fno_eligible.
    """
    latest_pred_sub = (
        select(Prediction.stock_id, func.max(Prediction.predicted_at).label("latest"))
        .group_by(Prediction.stock_id)
        .subquery()
    )
    latest_pred = aliased(Prediction)

    latest_fno_sub = (
        select(FNORecommendation.symbol, func.max(FNORecommendation.created_at).label("latest"))
        .group_by(FNORecommendation.symbol)
        .subquery()
    )
    latest_fno = aliased(FNORecommendation)

    base_query = (
        select(Stock, latest_pred, latest_fno)
        .where(Stock.is_fno_eligible.is_(True))
        .outerjoin(latest_pred_sub, Stock.id == latest_pred_sub.c.stock_id)
        .outerjoin(
            latest_pred,
            (latest_pred.stock_id == latest_pred_sub.c.stock_id) & (latest_pred.predicted_at == latest_pred_sub.c.latest),
        )
        .outerjoin(latest_fno_sub, Stock.symbol == latest_fno_sub.c.symbol)
        .outerjoin(
            latest_fno,
            (latest_fno.symbol == latest_fno_sub.c.symbol) & (latest_fno.created_at == latest_fno_sub.c.latest),
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
        FnoRowOut(
            symbol=stock.symbol,
            company_name=stock.company_name,
            sector=stock.sector,
            signal=pred.signal.value if pred else None,
            confidence=pred.confidence if pred else None,
            current_price=pred.current_price if pred else None,
            target_price=pred.target_price if pred else None,
            stop_loss=pred.stop_loss if pred else None,
            predicted_return_5d=pred.predicted_return_5d if pred else None,
            fno_strategy=(fno.rationale.split("]")[0].lstrip("[") if fno and fno.rationale and fno.rationale.startswith("[") else None),
            fno_confidence=fno.confidence if fno else None,
            pcr=fno.pcr if fno else None,
            iv=fno.iv if fno else None,
            predicted_at=pred.predicted_at.isoformat() if pred else None,
        )
        for stock, pred, fno in rows
    ]

    return AllFnoResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/indices/summary")
async def indices_summary(_user: User = Depends(get_current_user)):
    """Return current Nifty50, Bank Nifty, Nifty IT levels."""
    from src.ingestion.data_fetcher import YFinanceFetcher
    from yfinance.exceptions import YFRateLimitError
    symbols = {"NIFTY50": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK", "NIFTYIT": "^CNXIT"}
    result = {}
    yf = YFinanceFetcher()
    for name, sym in symbols.items():
        try:
            df = yf.fetch_ohlcv(sym, period="5d", interval="1d")
        except YFRateLimitError as exc:
            # fetch_ohlcv already retried with backoff — Yahoo is still
            # rate-limiting this specific symbol. Skip it and return the
            # others rather than crashing the whole response over one
            # index — confirmed live that an uncaught rate-limit exception
            # here took down this endpoint entirely.
            logger.warning("Yahoo rate-limited %s, omitting from indices summary: %s", sym, exc)
            continue
        if not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            change_pct = (last.Close - prev.Close) / prev.Close * 100
            result[name] = {
                "close":      round(last.Close, 2),
                "change_pct": round(change_pct, 2),
                "high":       round(last.High, 2),
                "low":        round(last.Low, 2),
            }
    return result
