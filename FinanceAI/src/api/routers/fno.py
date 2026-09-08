"""FNO router — options chain analysis, strategy recommendations."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.api.routers.auth import get_current_user, User

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


@router.get("/{symbol}/strategy", response_model=FNOStrategyOut)
async def fno_strategy(
    symbol: str,
    _user:  User = Depends(get_current_user),
):
    """
    Analyze F&O data for a symbol and return a strategy recommendation.
    Uses NSE option chain + stock prediction ensemble.
    """
    from src.ingestion.data_fetcher import YFinanceFetcher, NSEFetcher
    from src.models.ensemble.predictor import ModelOrchestrator
    from src.recommendation.engine import FNOAnalyzer

    # Stock prediction
    yf = YFinanceFetcher()
    ohlcv = yf.fetch_ohlcv(symbol, period="1y")
    if ohlcv.empty:
        raise HTTPException(404, detail=f"No OHLCV data for {symbol}")

    fund = yf.fetch_fundamentals(symbol)
    pred = ModelOrchestrator().predict(symbol, ohlcv, fund)

    # Option chain
    nse = NSEFetcher()
    try:
        import asyncio
        chain_raw = asyncio.run(nse.fetch_option_chain(symbol.replace(".NS", "")))
        oi_raw    = asyncio.run(nse.fetch_fno_oi(symbol.replace(".NS", "")))
    except Exception:
        # Fallback to Yahoo options
        chain_raw = yf.fetch_options_chain(symbol)
        oi_raw = {}

    analyzer = FNOAnalyzer()
    rec = analyzer.analyze(symbol, pred, chain_raw, oi_raw)

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
    )


@router.get("/indices/summary")
async def indices_summary(_user: User = Depends(get_current_user)):
    """Return current Nifty50, Bank Nifty, Nifty IT levels."""
    from src.ingestion.data_fetcher import YFinanceFetcher
    symbols = {"NIFTY50": "^NSEI", "SENSEX": "^BSESN", "BANKNIFTY": "^NSEBANK", "NIFTYIT": "^CNXIT"}
    result = {}
    yf = YFinanceFetcher()
    for name, sym in symbols.items():
        df = yf.fetch_ohlcv(sym, period="5d", interval="1d")
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
