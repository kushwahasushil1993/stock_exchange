"""Stocks router — OHLCV data, fundamentals, technical indicators."""
import json
from typing import Optional
from fastapi import APIRouter, Query, Depends, HTTPException
from pydantic import BaseModel
from src.api.routers.auth import get_current_user, User
from src.database.models import redis_cache

router = APIRouter()


class OHLCVBar(BaseModel):
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class FundamentalsOut(BaseModel):
    symbol:               str
    company_name:         Optional[str]
    sector:               Optional[str]
    pe_ratio:             Optional[float]
    pb_ratio:             Optional[float]
    roe:                  Optional[float]
    debt_to_equity:       Optional[float]
    revenue_growth_yoy:   Optional[float]
    profit_margin:        Optional[float]
    dividend_yield:       Optional[float]
    beta:                 Optional[float]
    market_cap:           Optional[float]


@router.get("/{symbol}/ohlcv")
async def get_ohlcv(
    symbol:   str,
    period:   str = Query(default="1y", regex="^(1mo|3mo|6mo|1y|2y|5y)$"),
    interval: str = Query(default="1d", regex="^(1d|1wk|1mo)$"),
    _user:    User = Depends(get_current_user),
):
    cache_key = f"ohlcv:{symbol}:{period}:{interval}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.ingestion.data_fetcher import YFinanceFetcher
    df = YFinanceFetcher().fetch_ohlcv(symbol, period=period, interval=interval)
    if df.empty:
        raise HTTPException(404, f"No data found for {symbol}")

    bars = [
        OHLCVBar(
            date   = str(idx.date()),
            open   = round(row.Open,   2),
            high   = round(row.High,   2),
            low    = round(row.Low,    2),
            close  = round(row.Close,  2),
            volume = int(row.Volume),
        ).model_dump()
        for idx, row in df.iterrows()
    ]
    await redis_cache.set(cache_key, json.dumps(bars), ttl=3600)
    return bars


@router.get("/{symbol}/fundamentals", response_model=FundamentalsOut)
async def get_fundamentals(
    symbol: str,
    _user:  User = Depends(get_current_user),
):
    cache_key = f"fundamentals:{symbol}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.ingestion.data_fetcher import YFinanceFetcher
    data = YFinanceFetcher().fetch_fundamentals(symbol)
    out  = FundamentalsOut(symbol=symbol, **{
        k: data.get(k) for k in FundamentalsOut.model_fields if k != "symbol"
    })
    await redis_cache.set(cache_key, out.model_dump_json(), ttl=86400)
    return out


@router.get("/{symbol}/technicals")
async def get_technicals(
    symbol: str,
    period: str = "1y",
    _user:  User = Depends(get_current_user),
):
    """Return last 30 rows of technical indicators for charting."""
    from src.ingestion.data_fetcher import YFinanceFetcher
    from src.features.technical import TechnicalFeatures

    df = YFinanceFetcher().fetch_ohlcv(symbol, period=period)
    if df.empty:
        raise HTTPException(404, detail=f"No data for {symbol}")

    enriched = TechnicalFeatures().compute(df)
    last30   = enriched.tail(30).fillna(0)
    return last30.reset_index().rename(columns={"Date": "date"}).to_dict(orient="records")
