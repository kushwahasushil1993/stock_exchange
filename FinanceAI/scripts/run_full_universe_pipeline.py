"""
Full-universe pipeline: fetch every NSE-listed equity (~2,300 symbols) and
every F&O-eligible stock among them (~210), run the prediction engine on
each, and persist buy/sell signals to Postgres.

    python scripts/run_full_universe_pipeline.py                 # full run
    python scripts/run_full_universe_pipeline.py --limit 50       # smoke test
    python scripts/run_full_universe_pipeline.py --workers 16     # more concurrency

Equities get a Stock + Fundamental + Prediction row each (same schema the
watchlist DAG writes to). F&O-eligible ones additionally get an
FNORecommendation row — derived from the same technical/fundamental signal,
since NSE's live option-chain API needs a browser session (see NSE_COOKIE in
.env) and isn't fetched here at this scale; PCR/IV/max-pain default to
neutral values, so FNO strategies lean on signal + confidence rather than
real options-market data.
"""
import argparse
import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select

from config.settings import settings
from src.database.models import (
    AssetType,
    AsyncSessionFactory,
    Fundamental,
    FNORecommendation,
    Prediction,
    SignalType,
    Stock,
    init_db,
)
from src.ingestion.data_fetcher import NSEUniverseFetcher, YFinanceFetcher
from src.models.ensemble.predictor import ModelOrchestrator
from src.recommendation.engine import FNOAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("full_universe_pipeline")

MIN_OHLCV_ROWS = 60  # enough history for the technical indicators used downstream


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


async def run(limit: Optional[int], workers: int, universe: str) -> None:
    print("=" * 70)
    print("  FinanceAI — Full NSE Universe Scan")
    print("=" * 70)

    await init_db()

    uf = NSEUniverseFetcher()
    equities = await uf.fetch_equity_symbols()
    fno_symbols = set(await uf.fetch_fno_symbols())
    print(f"  NSE equities: {len(equities)} | F&O-eligible: {len(fno_symbols)}")

    symbols = [e["yf_symbol"] for e in equities]
    if universe == "fno":
        symbols = [s for s in symbols if s in fno_symbols]
    if limit:
        symbols = symbols[:limit]
    print(f"  Scanning {len(symbols)} symbols with {workers} workers…\n")

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

        if i % 100 == 0 or i == len(symbols):
            elapsed = time.time() - start
            print(f"  [{i}/{len(symbols)}] ok={ok} failed={fail} "
                  f"| BUY={buy} SELL={sell} HOLD={hold} | {elapsed:.0f}s elapsed")

    await flush_batch()
    executor.shutdown(wait=True)

    print("\n" + "=" * 70)
    print(f"  Done in {time.time() - start:.0f}s — {ok} predicted, {fail} skipped/failed")
    print(f"  BUY={buy}  SELL={sell}  HOLD={hold}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N symbols (smoke test)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fetch threads")
    parser.add_argument("--universe", choices=["all", "fno"], default="all",
                         help="'all' = every NSE equity, 'fno' = only F&O-eligible stocks")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.workers, args.universe))
