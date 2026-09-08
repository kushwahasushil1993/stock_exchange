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
from sqlalchemy import select, desc

from src.database.models import get_db, Prediction, Stock
from src.database.models import redis_cache
from src.api.routers.auth import get_current_user, User
from config.settings import settings
from src.recommendation.engine import TradingAgents

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────────────────
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

    class Config:
        from_attributes = True


class TopPicksResponse(BaseModel):
    buy_picks:       List[PredictionOut]
    sell_picks:      List[PredictionOut]
    total_analyzed:  int
    generated_at:    str


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
        select(Prediction.stock_id, Prediction.predicted_at.label("latest"))
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
            current_price       = 0,    # fetch from latest OHLCV separately
            target_price        = pred.target_price or 0,
            stop_loss           = pred.stop_loss or 0,
            risk_reward_ratio   = pred.risk_reward_ratio or 0,
            predicted_return_5d = pred.predicted_return_5d or pred.predicted_return_1d or 0,
            feature_importance  = pred.feature_importance or {},
            predicted_at        = pred.predicted_at.isoformat(),
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


async def _run_prediction_task(symbol: str, db: AsyncSession):
    """Background task: fetch data → features → predict → store."""
    try:
        from src.ingestion.data_fetcher import YFinanceFetcher, NSEFetcher
        from src.features.technical import TechnicalFeatures
        from src.models.ensemble.predictor import ModelOrchestrator
        from src.sentiment.analyzer import NewsAggregator, NewsFetcher

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
