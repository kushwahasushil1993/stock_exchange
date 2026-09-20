"""
On-demand analysis — give it a stock/F&O/mutual-fund name (or ticker/scheme
code), it fetches fresh data, checks the news, runs the prediction engine,
and answers "buy or not" directly. Unlike /predictions/top-picks (reads
already-computed rows from the DB), everything here runs live per request.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.routers.auth import get_current_user, User
from src.database.models import redis_cache

logger = logging.getLogger(__name__)
router = APIRouter()

_EQUITY_CACHE_KEY = "analyze:equity_universe"
_FNO_CACHE_KEY = "analyze:fno_universe"
_CACHE_TTL = 24 * 3600  # NSE's listed-symbol file changes rarely


# ─── Symbol / fund resolution ────────────────────────────────────────────────
async def _get_equity_universe() -> List[Dict[str, str]]:
    import json

    cached = await redis_cache.get(_EQUITY_CACHE_KEY)
    if cached:
        return json.loads(cached)

    from src.ingestion.data_fetcher import NSEUniverseFetcher
    equities = await NSEUniverseFetcher().fetch_equity_symbols()
    await redis_cache.set(_EQUITY_CACHE_KEY, json.dumps(equities), ttl=_CACHE_TTL)
    return equities


async def _get_fno_universe() -> set:
    import json

    cached = await redis_cache.get(_FNO_CACHE_KEY)
    if cached:
        return set(json.loads(cached))

    from src.ingestion.data_fetcher import NSEUniverseFetcher
    symbols = await NSEUniverseFetcher().fetch_fno_symbols()
    await redis_cache.set(_FNO_CACHE_KEY, json.dumps(symbols), ttl=_CACHE_TTL)
    return set(symbols)


async def _resolve_equity(query: str) -> Dict[str, str]:
    """
    Accept a ticker (RELIANCE / RELIANCE.NS) or a company name (Reliance
    Industries) and resolve it to {symbol, yf_symbol, company_name}.
    """
    q = query.strip()
    q_upper = q.upper().replace(".NS", "")
    universe = await _get_equity_universe()

    for e in universe:
        if e["symbol"] == q_upper:
            return e
    for e in universe:
        if q.lower() in e["company_name"].lower():
            return e

    raise HTTPException(status_code=404, detail=f"No NSE-listed equity matching '{query}'")


# ─── Shared stock analysis ───────────────────────────────────────────────────
async def _analyze_equity(query: str) -> Dict[str, Any]:
    from src.ingestion.data_fetcher import YFinanceFetcher, NewsFetcher
    from src.models.ensemble.predictor import ModelOrchestrator
    from src.sentiment.analyzer import SentimentAnalyzer

    match = await _resolve_equity(query)
    yf_symbol = match["yf_symbol"]

    yf = YFinanceFetcher()
    ohlcv = yf.fetch_ohlcv(yf_symbol, period="5y")
    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"No price data available for {yf_symbol}")
    fund = yf.fetch_fundamentals(yf_symbol)

    # News: search by company name directly rather than the existing
    # symbol-extraction sentiment pipeline, which only recognizes ~50
    # hardcoded large-cap symbols (src/sentiment/analyzer.py NSE_SYMBOLS) —
    # not useful for the other ~2,250 NSE-listed names.
    news = NewsFetcher()
    articles = await news.fetch_newsapi(
        query=f"{match['company_name']} share price",
        days_back=7,
        page_size=20,
    )
    sentiment_score = 0.0
    headlines: List[str] = []
    if articles:
        analyzer = SentimentAnalyzer()
        scores = []
        for a in articles:
            text = f"{a.get('title', '')}. {a.get('description') or ''}"
            scores.append(analyzer.analyze(text)["score"])
        sentiment_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        headlines = [a.get("title", "") for a in articles[:5]]

    pred = ModelOrchestrator().predict(yf_symbol, ohlcv, fund, sentiment_score)

    return {
        "symbol": yf_symbol,
        "company_name": match["company_name"],
        "current_price": pred["current_price"],
        "target_price": pred["target_price"],
        "stop_loss": pred["stop_loss"],
        # The ensemble is a short-term (5-trading-day) model — 5 years of
        # history feeds its features/context, but it doesn't forecast a
        # 5-year return. Labeled explicitly so this isn't overstated.
        "expected_return_5d_pct": pred["predicted_return_5d"],
        "signal": pred["signal"],
        "recommendation": {"BUY": "Buy", "SELL": "Don't buy — sell/avoid", "HOLD": "Hold — no strong edge either way"}[pred["signal"]],
        "confidence": pred["confidence"],
        "risk_reward_ratio": pred["risk_reward_ratio"],
        "technical_score": pred["technical_score"],
        "fundamental_score": pred["fundamental_score"],
        "sentiment_score": sentiment_score,
        "news": {
            "articles_analyzed": len(articles),
            "headlines": headlines,
        },
        "prediction_horizon_note": "expected_return_5d_pct is a 5-trading-day forecast, computed from 5 years of price history — not a 5-year return forecast.",
        "predicted_at": pred["predicted_at"],
        "_pred": pred,  # internal, stripped before response
    }


# ─── Response schemas ─────────────────────────────────────────────────────────
class NewsOut(BaseModel):
    articles_analyzed: int
    headlines: List[str]


class StockAnalysisOut(BaseModel):
    symbol: str
    company_name: str
    current_price: float
    target_price: float
    stop_loss: float
    expected_return_5d_pct: float
    signal: str
    recommendation: str
    confidence: float
    risk_reward_ratio: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    news: NewsOut
    prediction_horizon_note: str
    predicted_at: str


class FNOAnalysisOut(StockAnalysisOut):
    fno_strategy: str
    fno_risk_level: str
    fno_rationale: str


class MutualFundAnalysisOut(BaseModel):
    scheme_code: str
    scheme_name: str
    category: str
    amc: str
    nav: float
    return_1y: Optional[float]
    return_3y: Optional[float]
    return_5y: Optional[float]
    composite_score: float
    recommendation: str
    rationale: str


def _strip_internal(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/stock", response_model=StockAnalysisOut)
async def analyze_stock(
    query: str = Query(..., description="Ticker (RELIANCE / RELIANCE.NS) or company name"),
    _user: User = Depends(get_current_user),
):
    """5-year price history + live news sentiment + ensemble prediction -> buy or not."""
    result = await _analyze_equity(query)
    return StockAnalysisOut(**_strip_internal(result))


@router.get("/fno", response_model=FNOAnalysisOut)
async def analyze_fno(
    query: str = Query(..., description="Ticker or company name — must be F&O-eligible"),
    _user: User = Depends(get_current_user),
):
    """Same as /analyze/stock, plus a derived F&O (futures/options) strategy."""
    from src.recommendation.engine import FNOAnalyzer

    fno_universe = await _get_fno_universe()
    result = await _analyze_equity(query)
    if result["symbol"] not in fno_universe:
        raise HTTPException(
            status_code=400,
            detail=f"{result['symbol']} is not F&O-eligible on NSE (use /analyze/stock instead)",
        )

    fno_rec = FNOAnalyzer().analyze(result["symbol"], result["_pred"], option_chain={}, oi_data={})
    return FNOAnalysisOut(
        **_strip_internal(result),
        fno_strategy=fno_rec.strategy,
        fno_risk_level=fno_rec.risk_level,
        fno_rationale=fno_rec.rationale,
    )


@router.get("/mutual-fund", response_model=MutualFundAnalysisOut)
async def analyze_mutual_fund(
    query: str = Query(..., description="Scheme name (partial match) or exact scheme code"),
    _user: User = Depends(get_current_user),
):
    """Look up one AMFI scheme by name/code, fetch its real return history, score it."""
    from src.ingestion.data_fetcher import AMFIFetcher
    from src.recommendation.engine import MutualFundRecommender

    fetcher = AMFIFetcher()
    nav_df = await fetcher.fetch_all_navs()
    if nav_df.empty:
        raise HTTPException(status_code=502, detail="Could not fetch AMFI NAV data")

    match = nav_df[nav_df["scheme_code"] == query.strip()]
    if match.empty:
        match = nav_df[nav_df["scheme_name"].str.contains(query.strip(), case=False, na=False)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No mutual fund scheme matching '{query}'")
    row = match.iloc[0]

    hist = await fetcher.fetch_historical_nav(str(row["scheme_code"]))
    returns = fetcher.compute_returns(hist)

    recommender = MutualFundRecommender()
    import pandas as pd
    scored_row = pd.Series({**row.to_dict(), **returns, "expense_ratio": None})
    composite_score = round(recommender._score_fund(scored_row), 4)

    # MutualFundRecommender has no built-in discrete decision, so these
    # thresholds are added here. They're deliberately NOT centered on 0.5:
    # _score_fund() weights sharpe_ratio/sortino_ratio/alpha/expense_ratio at
    # 55% combined, and none of those are populated anywhere in this
    # codebase (no source computes them) — those terms default to ~0, so the
    # realistic ceiling on composite_score is ~0.45-0.49, not 1.0. Calibrated
    # against /mutual-funds/recommendations' actual output: the #1-ranked
    # real fund out of ~200 candidates scored 0.3524.
    if composite_score >= 0.30:
        recommendation = "Buy"
    elif composite_score >= 0.20:
        recommendation = "Hold / neutral"
    else:
        recommendation = "Avoid — weak risk-adjusted returns"

    return MutualFundAnalysisOut(
        scheme_code=str(row["scheme_code"]),
        scheme_name=str(row["scheme_name"]),
        category=str(row["category"]),
        amc=str(row["amc"]),
        nav=float(row["nav"]),
        return_1y=returns.get("return_1y"),
        return_3y=returns.get("return_3y"),
        return_5y=returns.get("return_5y"),
        composite_score=composite_score,
        recommendation=recommendation,
        rationale=recommender._build_rationale(scored_row),
    )
