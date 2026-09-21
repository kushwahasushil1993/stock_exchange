"""
Full-universe scan: fetch every NSE-listed equity (~2,300 symbols) and every
F&O-eligible stock among them (~210), run the prediction engine on each, and
persist buy/sell signals to Postgres.

Shared by:
  - scripts/run_full_universe_pipeline.py  (CLI, prints progress to stdout)
  - POST /api/v1/predictions/run-universe  (API trigger, no params — the
    scan runs as a background task and reports progress via Redis, since a
    live request can't stay open for the ~2-3 minutes this takes)

Equities get a Stock + Fundamental + Prediction row each. F&O-eligible ones
additionally get an FNORecommendation row — derived from the same
technical/fundamental signal, since NSE's live option-chain API needs a
browser session (see NSE_COOKIE in .env) and isn't fetched here at this
scale; PCR/IV/max-pain default to neutral values, so FNO strategies lean on
signal + confidence rather than real options-market data.
"""
import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, List, Optional

from sqlalchemy import select

from src.database.models import (
    AssetType,
    AsyncSessionFactory,
    Fundamental,
    FNORecommendation,
    Prediction,
    SignalType,
    Stock,
    init_db,
    redis_cache,
)
from src.ingestion.data_fetcher import NSEUniverseFetcher, YFinanceFetcher
from src.models.ensemble.predictor import ModelOrchestrator
from src.recommendation.engine import FNOAnalyzer

logger = logging.getLogger("universe_scan")

MIN_OHLCV_ROWS = 60  # enough history for the technical indicators used downstream
STATUS_KEY = "universe_scan:status"  # redis key: JSON status blob, polled by GET .../status
STATUS_TTL = 24 * 3600
STALE_AFTER_S = 180  # progress updates every 50 symbols land far more often than this
                     # in practice — if "running" hasn't ticked in 3 minutes, the process
                     # that was writing it is dead (e.g. the container got recreated
                     # mid-scan), not actually still working.


async def _ensure_redis() -> None:
    """
    redis_cache is normally connected once by the FastAPI app's lifespan
    hook — fine for the API trigger, but the standalone CLI script never
    goes through that, so connect lazily here instead of assuming it.
    """
    if redis_cache._client is None:
        await redis_cache.connect()


async def _set_status(**fields) -> None:
    await _ensure_redis()
    fields["last_updated"] = datetime.utcnow().isoformat()
    await redis_cache.set(STATUS_KEY, json.dumps(fields), ttl=STATUS_TTL)


async def get_status() -> Dict:
    await _ensure_redis()
    cached = await redis_cache.get(STATUS_KEY)
    if not cached:
        return {"state": "never_run"}

    status = json.loads(cached)
    if status.get("state") == "running":
        # Missing last_updated (a status written before this field existed)
        # is just as unverifiable as a genuinely old timestamp — treat both
        # as stale rather than trusting an unverifiable "running" forever.
        last_updated = status.get("last_updated")
        age_s = (
            (datetime.utcnow() - datetime.fromisoformat(last_updated)).total_seconds()
            if last_updated else STALE_AFTER_S + 1
        )
        if age_s > STALE_AFTER_S:
            status = {**status, "state": "stale",
                      "note": f"No progress update in {round(age_s)}s — the process that was "
                               "running this scan appears to have died (e.g. a container "
                               "restart mid-scan). Safe to trigger a new run."}
    return status


def fetch_and_predict(symbol: str, is_fno: bool) -> Dict:
    """
    Runs in a worker thread — blocking yfinance calls + CPU-bound feature/model
    work. Never raises: failures come back as {"symbol": ..., "error": ...}.
    """
    yf = YFinanceFetcher()
    try:
        ohlcv = yf.fetch_ohlcv(symbol, period="2y")
        if ohlcv.empty or len(ohlcv) < MIN_OHLCV_ROWS:
            return {"symbol": symbol, "error": f"insufficient OHLCV data ({len(ohlcv)} rows)"}

        fund = yf.fetch_fundamentals(symbol)
        orch = ModelOrchestrator()
        pred = orch.predict(symbol, ohlcv, fund, sentiment_score=0.0)

        result = {"symbol": symbol, "fund": fund, "pred": pred, "is_fno": is_fno}
        if is_fno:
            fno_rec = FNOAnalyzer().analyze(symbol, pred, option_chain={}, oi_data={})
            result["fno_rec"] = fno_rec
        return result
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}


async def persist_result(db, result: Dict) -> None:
    symbol = result["symbol"]
    fund = result["fund"]
    pred = result["pred"]

    res = await db.execute(select(Stock).where(Stock.symbol == symbol))
    stock = res.scalar_one_or_none()
    if not stock:
        stock = Stock(symbol=symbol, company_name=fund.get("company_name") or symbol)
        db.add(stock)
        await db.flush()

    stock.company_name = fund.get("company_name") or stock.company_name
    stock.sector = fund.get("sector")
    stock.industry = fund.get("industry")
    stock.market_cap = fund.get("market_cap")
    stock.is_fno_eligible = result["is_fno"]

    fres = await db.execute(select(Fundamental).where(Fundamental.stock_id == stock.id))
    fundamental = fres.scalar_one_or_none()
    if not fundamental:
        fundamental = Fundamental(stock_id=stock.id)
        db.add(fundamental)
    for field in (
        "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "revenue_growth_yoy",
        "profit_margin", "dividend_yield", "beta", "free_cash_flow", "eps_growth",
    ):
        setattr(fundamental, field, fund.get(field))

    db.add(Prediction(
        stock_id=stock.id,
        model_version="universe-scan-v1",
        signal=SignalType(pred["signal"]),
        confidence=pred["confidence"],
        predicted_return_1d=None,
        predicted_return_5d=pred.get("predicted_return_5d"),
        predicted_return_1m=None,
        target_price=pred.get("target_price"),
        stop_loss=pred.get("stop_loss"),
        risk_reward_ratio=pred.get("risk_reward_ratio"),
        sentiment_score=pred.get("sentiment_score"),
        technical_score=pred.get("technical_score"),
        fundamental_score=pred.get("fundamental_score"),
        composite_score=pred.get("composite_score"),
        feature_importance=pred.get("feature_importance") or None,
    ))

    if result.get("is_fno") and "fno_rec" in result:
        r = result["fno_rec"]
        db.add(FNORecommendation(
            symbol=symbol,
            asset_type=AssetType.FNO_FUTURE if r.asset_type == "FUTURE" else AssetType.FNO_OPTION,
            expiry_date=None,
            strike_price=r.strike_price,
            option_type=r.option_type,
            signal=SignalType(r.signal) if r.signal in SignalType.__members__ else SignalType.HOLD,
            iv=r.iv,
            delta=None, theta=None, vega=None,
            oi_change_pct=r.oi_change_pct,
            pcr=r.pcr,
            confidence=r.confidence,
            rationale=f"[{r.strategy}] {r.rationale}",
        ))


async def run_universe_scan(
    limit: Optional[int] = None,
    workers: int = 8,
    universe: str = "all",
    on_progress: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Runs the scan to completion. Updates STATUS_KEY in Redis as it goes (so
    a caller that can't hold the request open — the API trigger — can poll
    progress) and also calls on_progress(dict), if given, for callers that
    can print directly (the CLI script).
    """
    await init_db()
    await _set_status(state="running", scanned=0, total=0, ok=0, failed=0,
                       buy=0, sell=0, hold=0, started_at=datetime.utcnow().isoformat())

    try:
        uf = NSEUniverseFetcher()
        equities = await uf.fetch_equity_symbols()
        fno_symbols = set(await uf.fetch_fno_symbols())

        symbols = [e["yf_symbol"] for e in equities]
        if universe == "fno":
            symbols = [s for s in symbols if s in fno_symbols]
        if limit:
            symbols = symbols[:limit]
        total = len(symbols)

        if on_progress:
            on_progress({"event": "started", "equities": len(equities),
                         "fno_eligible": len(fno_symbols), "scanning": total, "workers": workers})
        await _set_status(state="running", scanned=0, total=total, ok=0, failed=0,
                           buy=0, sell=0, hold=0, started_at=datetime.utcnow().isoformat())

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=workers)

        ok = fail = 0
        buy = sell = hold = 0
        start = time.time()
        BATCH = 50
        batch: List[Dict] = []

        async def flush_batch():
            nonlocal batch
            if not batch:
                return
            async with AsyncSessionFactory() as db:
                for r in batch:
                    await persist_result(db, r)
                await db.commit()
            batch = []

        tasks = [
            loop.run_in_executor(executor, fetch_and_predict, sym, sym in fno_symbols)
            for sym in symbols
        ]
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            result = await coro
            if "error" in result:
                fail += 1
            else:
                ok += 1
                sig = result["pred"]["signal"]
                buy += sig == "BUY"
                sell += sig == "SELL"
                hold += sig == "HOLD"
                batch.append(result)
                if len(batch) >= BATCH:
                    await flush_batch()

            if i % 50 == 0 or i == total:
                elapsed = round(time.time() - start)
                if on_progress:
                    on_progress({"event": "progress", "scanned": i, "total": total,
                                 "ok": ok, "failed": fail, "buy": buy, "sell": sell,
                                 "hold": hold, "elapsed_s": elapsed})
                await _set_status(state="running", scanned=i, total=total, ok=ok, failed=fail,
                                   buy=buy, sell=sell, hold=hold, elapsed_s=elapsed)

        await flush_batch()
        executor.shutdown(wait=True)

        summary = {
            "state": "completed", "scanned": total, "total": total,
            "ok": ok, "failed": fail, "buy": buy, "sell": sell, "hold": hold,
            "elapsed_s": round(time.time() - start),
            "completed_at": datetime.utcnow().isoformat(),
        }
        await _set_status(**summary)
        if on_progress:
            on_progress({"event": "completed", **summary})
        return summary

    except Exception as exc:
        logger.exception("Universe scan failed")
        await _set_status(state="failed", error=str(exc), failed_at=datetime.utcnow().isoformat())
        raise
