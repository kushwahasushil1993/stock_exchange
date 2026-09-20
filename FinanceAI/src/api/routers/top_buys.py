"""
Consolidated "top buy" router — the best BUY-signal stocks, F&O strategies,
and mutual funds in one call, with company/scheme names included.

Stocks and F&O are filtered by a start/end date range on when the signal was
generated (predicted_at / created_at) since those are logged, dated records.
Mutual funds are a live snapshot (AMFI publishes no dated recommendation
history to range-filter), so start_date/end_date don't apply there.
"""
from datetime import date, datetime, time
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routers.auth import get_current_user, User
from src.api.routers.mutual_funds import get_top_mutual_funds
from src.database.models import FNORecommendation, Prediction, SignalType, Stock, get_db

router = APIRouter()


class TopBuyStock(BaseModel):
    symbol:           str
    company_name:     str
    confidence:       float
    composite_score:  float
    target_price:     Optional[float]
    stop_loss:        Optional[float]
    risk_reward_ratio: Optional[float]
    predicted_at:     str


class TopBuyFNO(BaseModel):
    symbol:        str
    company_name:  Optional[str]
    strategy:      str
    confidence:    float
    rationale:     Optional[str]
    created_at:    str


class TopBuyMutualFund(BaseModel):
    scheme_code:     str
    scheme_name:     str
    category:        str
    amc:             str
    composite_score: float
    return_1y:       Optional[float]
    return_3y:       Optional[float]
    return_5y:       Optional[float]
    rank:            int


class TopBuysResponse(BaseModel):
    start_date:    Optional[str]
    end_date:      Optional[str]
    stocks:        List[TopBuyStock]
    fno:           List[TopBuyFNO]
    mutual_funds:  List[TopBuyMutualFund]
    generated_at:  str


def _day_bounds(start_date: Optional[date], end_date: Optional[date]):
    start_dt = datetime.combine(start_date, time.min) if start_date else None
    end_dt = datetime.combine(end_date, time.max) if end_date else None
    return start_dt, end_dt


async def _top_buy_stocks(db: AsyncSession, start_dt, end_dt, top_n: int) -> List[TopBuyStock]:
    conditions = [Prediction.signal == SignalType.BUY]
    if start_dt:
        conditions.append(Prediction.predicted_at >= start_dt)
    if end_dt:
        conditions.append(Prediction.predicted_at <= end_dt)

    # Over-fetch and dedupe in Python (keep each stock's single best-scoring
    # row) rather than a float-equality self-join, which is fragile.
    result = await db.execute(
        select(Prediction, Stock.symbol, Stock.company_name)
        .join(Stock, Prediction.stock_id == Stock.id)
        .where(*conditions)
        .order_by(desc(Prediction.composite_score))
        .limit(top_n * 5)
    )

    seen = set()
    picks: List[TopBuyStock] = []
    for pred, symbol, company_name in result.all():
        if symbol in seen:
            continue
        seen.add(symbol)
        picks.append(TopBuyStock(
            symbol            = symbol,
            company_name      = company_name or symbol,
            confidence        = pred.confidence,
            composite_score   = pred.composite_score or 0.0,
            target_price      = pred.target_price,
            stop_loss         = pred.stop_loss,
            risk_reward_ratio = pred.risk_reward_ratio,
            predicted_at      = pred.predicted_at.isoformat(),
        ))
        if len(picks) >= top_n:
            break
    return picks


async def _top_buy_fno(db: AsyncSession, start_dt, end_dt, top_n: int) -> List[TopBuyFNO]:
    conditions = [FNORecommendation.signal == SignalType.BUY]
    if start_dt:
        conditions.append(FNORecommendation.created_at >= start_dt)
    if end_dt:
        conditions.append(FNORecommendation.created_at <= end_dt)

    result = await db.execute(
        select(FNORecommendation, Stock.company_name)
        .outerjoin(Stock, Stock.symbol == FNORecommendation.symbol)
        .where(*conditions)
        .order_by(desc(FNORecommendation.confidence))
        .limit(top_n * 5)
    )

    seen = set()
    picks: List[TopBuyFNO] = []
    for rec, company_name in result.all():
        if rec.symbol in seen:
            continue
        seen.add(rec.symbol)
        picks.append(TopBuyFNO(
            symbol       = rec.symbol,
            company_name = company_name,
            strategy     = rec.rationale.split("]")[0].lstrip("[") if rec.rationale and rec.rationale.startswith("[") else "BUY FUTURE",
            confidence   = rec.confidence,
            rationale    = rec.rationale,
            created_at   = rec.created_at.isoformat(),
        ))
        if len(picks) >= top_n:
            break
    return picks


@router.get("", response_model=TopBuysResponse)
async def top_buys(
    start_date:   Optional[date] = Query(default=None, description="Filter stock/F&O signals generated on/after this date"),
    end_date:     Optional[date] = Query(default=None, description="Filter stock/F&O signals generated on/before this date"),
    top_n:        int = Query(default=10, ge=1, le=50),
    risk_profile: str = Query(default="MODERATE", pattern="^(CONSERVATIVE|MODERATE|AGGRESSIVE)$"),
    db:           AsyncSession = Depends(get_db),
    _user:        User = Depends(get_current_user),
):
    """
    Top BUY-signal stocks, F&O strategies, and mutual funds in one call —
    each with its name (company_name / scheme_name), not just a ticker.

    start_date/end_date filter stocks and F&O by when the signal was
    generated; mutual funds are always the current live ranking (see module
    docstring for why).
    """
    start_dt, end_dt = _day_bounds(start_date, end_date)

    stocks = await _top_buy_stocks(db, start_dt, end_dt, top_n)
    fno = await _top_buy_fno(db, start_dt, end_dt, top_n)
    mfs = await get_top_mutual_funds(risk_profile=risk_profile, top_n=top_n)

    return TopBuysResponse(
        start_date   = start_date.isoformat() if start_date else None,
        end_date     = end_date.isoformat() if end_date else None,
        stocks       = stocks,
        fno          = fno,
        mutual_funds = [
            TopBuyMutualFund(
                scheme_code     = m.scheme_code,
                scheme_name     = m.scheme_name,
                category        = m.category,
                amc             = m.amc,
                composite_score = m.composite_score,
                return_1y       = m.return_1y,
                return_3y       = m.return_3y,
                return_5y       = m.return_5y,
                rank            = m.rank,
            )
            for m in mfs
        ],
        generated_at = datetime.utcnow().isoformat(),
    )
