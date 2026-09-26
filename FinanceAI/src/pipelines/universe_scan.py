"""
Full-universe scan: fetch every NSE-listed equity (~2,585 symbols — main-
board "EQ" plus the "BE"/"BZ" trade-to-trade series, see
NSEUniverseFetcher.EQUITY_SERIES) and every F&O-eligible stock among them
(~210), run the prediction engine on each, and persist buy/sell signals to
Postgres.

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
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import settings
from src.database.models import (
    AssetType,
    AsyncSessionFactory,
    DecisionType,
    Fundamental,
    FNORecommendation,
    Prediction,
    PriceHistory,
    SignalType,
    Stock,
    init_db,
    redis_cache,
)
from src.ingestion.data_fetcher import INDEX_FNO, NSEUniverseFetcher, YFinanceFetcher
from src.models.ensemble.predictor import ModelOrchestrator
from src.recommendation.engine import FNOAnalyzer

logger = logging.getLogger("universe_scan")

MIN_OHLCV_ROWS = 60  # enough history for the technical indicators used downstream
FULL_HISTORY_PERIOD = "5y"  # backfill window for a symbol with no stored history yet
INCREMENTAL_PERIOD = "5d"   # covers weekends/holidays when a symbol already has history

_sync_engine = None  # lazily created; used from worker threads (see fetch_and_predict),
                     # never from the async event loop
STATUS_KEY = "universe_scan:status"  # redis key: JSON status blob, polled by GET .../status
STATUS_TTL = 24 * 3600
STALE_AFTER_S = 180  # progress updates every ~2s land far more often than this in
                     # practice — if "running" hasn't ticked in 3 minutes, the process
                     # that was writing it is dead (e.g. the container got recreated
                     # mid-scan), not actually still working.

LAST_FAILED_KEY = "universe_scan:last_failed"  # redis: JSON {symbol: reason}, from the
                                                # most recent run's still-failing symbols
LAST_FAILED_TTL = 7 * 24 * 3600

# Retry-until-it-works budget: a single run keeps retrying rate-limited/
# transient failures, with exponential backoff between rounds, until either
# nothing is left to retry or this much total time has passed. Genuinely
# bad symbols (insufficient trading history) are never retried — see
# _categorize_error — so this budget is spent entirely on traffic-related
# failures, which usually do clear given enough backoff.
MAX_RETRY_BUDGET_S = 25 * 60
MAX_RETRY_ROUNDS = 6
BASE_COOLDOWN_S = 20
MAX_COOLDOWN_S = 600


class AdaptiveThrottle:
    """
    AIMD (additive-increase / multiplicative-decrease) concurrency
    controller — the same family of algorithm TCP congestion control uses,
    applied to Yahoo request concurrency instead of network bandwidth.

    The old fixed-concurrency approach (constant `workers` all pass, halved
    on each fixed retry round) only ever reacted to rate-limiting after
    already tripping it, then guessed. This tracks a rolling window of
    recent outcomes DURING a pass and adjusts live: a burst of rate-limit
    errors partway through a big pass shrinks concurrency (and adds a small
    delay between dispatches) for the REST of that same pass, not just the
    next retry round; a clean window grows it back. That's what actually
    changes the traffic pattern hitting Yahoo, rather than just retrying
    the same burst pattern slower.

    Starts at a slow-start floor (2), not `initial`, regardless of how high
    `initial`/`workers` is configured — ramping up only via the same
    additive-increase-on-clean-window logic used throughout a run. This
    isn't just caution: firing `initial` (e.g. 8) requests at once as a
    scan's very first act was empirically confirmed to break yfinance's
    session for the rest of that process's lifetime — its one-time cookie/
    crumb bootstrap isn't safe under concurrent access, and once several
    threads race it simultaneously and it fails, EVERY request for the rest
    of that process keeps failing (0% success, even reduced to 1 worker
    with multi-second delays — a fresh process recovers instantly, so it's
    specifically the shared, already-corrupted in-process session state,
    not a real ongoing Yahoo block). A slow start lets that one-time
    bootstrap happen uncontended before concurrency ever ramps up.
    """

    def __init__(self, initial: int, min_workers: int = 1, max_workers: int = 20):
        self.workers = max(min_workers, min(2, initial, max_workers))
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.delay_s = 0.0
        self._window: List[bool] = []  # True = this request was rate-limited

    def record(self, rate_limited: bool) -> None:
        self._window.append(rate_limited)
        if len(self._window) < 10:
            return
        rate = sum(self._window) / len(self._window)
        self._window.clear()
        if rate > 0.15:
            # Multiplicative decrease — back off hard and fast, matching how
            # aggressively rate limiters themselves tend to escalate.
            self.workers = max(self.min_workers, self.workers // 2)
            self.delay_s = min(3.0, self.delay_s + 0.25)
        elif rate == 0.0:
            # Additive increase — only ramp back up once a full window came
            # back completely clean, so we don't oscillate into the limit
            # again immediately after backing off from it.
            self.workers = min(self.max_workers, self.workers + 1)
            self.delay_s = max(0.0, self.delay_s - 0.05)


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


async def _load_last_failed() -> Dict[str, str]:
    """{symbol: reason} still failing at the end of the most recent run —
    used to scan those FIRST on the next run, instead of only discovering
    the same trouble spots again after working through the whole universe."""
    await _ensure_redis()
    cached = await redis_cache.get(LAST_FAILED_KEY)
    return json.loads(cached) if cached else {}


async def _save_last_failed(failed: Dict[str, str]) -> None:
    await _ensure_redis()
    if failed:
        await redis_cache.set(LAST_FAILED_KEY, json.dumps(failed), ttl=LAST_FAILED_TTL)
    else:
        # Nothing left failing — clear it rather than leaving a stale
        # (now-resolved) list around for the next run to needlessly prioritize.
        await redis_cache.delete(LAST_FAILED_KEY)


def _warm_up_yfinance_session() -> bool:
    """
    One single-threaded, no-concurrency-possible request before the scan's
    thread pool ever starts, forcing yfinance's one-time cookie/crumb
    bootstrap to complete with zero chance of the multi-thread race that
    otherwise corrupts it for a whole process's lifetime (see
    AdaptiveThrottle's docstring for how that was found). Retries a few
    times with a short gap — a single 429 here is recoverable; several in
    a row means Yahoo genuinely isn't answering right now, and the caller
    should wait longer before spinning up the real scan at all.
    """
    for attempt in range(3):
        try:
            df = YFinanceFetcher().fetch_ohlcv("RELIANCE.NS", period="5d")
            if not df.empty:
                return True
        except Exception as exc:
            logger.warning("yfinance warm-up attempt %d failed: %s", attempt + 1, exc)
        time.sleep(5 * (attempt + 1))
    return False


def _get_sync_engine():
    """
    A plain sync engine (psycopg2, not asyncpg) for the one read
    fetch_and_predict needs from inside a worker thread — the rest of this
    module is async/asyncpg-based, which can't be called from a thread
    without its own event loop. Read-only, tiny queries (indexed lookup by
    symbol), so a modest pool alongside the thread pool's concurrency is
    plenty.
    """
    global _sync_engine
    if _sync_engine is None:
        sync_url = settings.postgres_url.replace("+asyncpg", "+psycopg2")
        _sync_engine = create_engine(sync_url, pool_pre_ping=True, pool_size=20, max_overflow=10)
    return _sync_engine


def _load_existing_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    """
    Whatever history is already in Postgres for this symbol, shaped exactly
    like YFinanceFetcher.fetch_ohlcv()'s return (DatetimeIndex named Date,
    Open/High/Low/Close/Volume + symbol columns) so it can be concatenated
    with a fresh incremental fetch with no special-casing downstream. None
    if the symbol has no stored history yet (never scanned, or a brand-new
    listing) — the caller does a full backfill in that case.
    """
    engine = _get_sync_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT ph.date, ph.open, ph.high, ph.low, ph.close, ph.volume "
                "FROM price_history ph JOIN stocks s ON s.id = ph.stock_id "
                "WHERE s.symbol = :symbol ORDER BY ph.date"
            ),
            {"symbol": symbol},
        ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df["symbol"] = symbol
    return df


def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance returns a tz-aware DatetimeIndex (e.g. +05:30 for NSE symbols);
    Postgres's `date` column — and _load_existing_ohlcv's reconstruction of
    it — has no timezone concept at all. Concatenating the two as-is raises
    "Cannot compare tz-naive and tz-aware timestamps" the moment anything
    tries to sort or merge them (confirmed directly). Only the calendar
    date matters for daily bars, so just drop the tz rather than converting.
    """
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _get_ohlcv(yf: YFinanceFetcher, symbol: str) -> pd.DataFrame:
    """
    The heart of "don't always re-fetch 5 years of history — only fetch
    what's actually new":
      - no stored history at all (never scanned, or a brand-new listing)
        -> full backfill (5y, or however much Yahoo actually has for a
        recently-listed stock — fetch_ohlcv naturally returns less, nothing
        special needed).
      - stored history that doesn't yet reach back ~5 years (true for
        every symbol scanned before this changed from a 2y window) -> one
        more full-range fetch to backfill the gap. Self-correcting: once
        this succeeds, existing.index.min() sits at ~5y back and this
        branch stops firing for that symbol.
      - stored history that already reaches back ~5 years and is current
        as of today's most recent session -> no Yahoo call at all.
      - stored history that reaches back ~5 years but is stale -> a small
        incremental fetch only, merged onto what's already stored.
    """
    existing = _load_existing_ohlcv(symbol)
    if existing is None or existing.empty:
        return yf.fetch_ohlcv(symbol, period=FULL_HISTORY_PERIOD)

    five_years_ago = datetime.utcnow().date() - timedelta(days=5 * 365)
    if existing.index.min().date() > five_years_ago:
        full = yf.fetch_ohlcv(symbol, period=FULL_HISTORY_PERIOD)
        if full.empty:
            return existing
        combined = pd.concat([existing, _tz_naive(full)])
        return combined[~combined.index.duplicated(keep="last")].sort_index()

    last_date = existing.index.max().date()
    if last_date >= datetime.utcnow().date():
        return existing  # already current as of today — nothing to fetch

    incremental = yf.fetch_ohlcv(symbol, period=INCREMENTAL_PERIOD)
    if incremental.empty:
        return existing  # Yahoo has nothing newer right now (weekend/holiday) — use what we have

    combined = pd.concat([existing, _tz_naive(incremental)])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


# Set once per run_universe_scan() call, before the thread pool starts —
# read (never mutated) by every worker thread. Avoids threading two extra
# parameters through _scan_pass/_prioritize/the pairs-triple plumbing for
# values that are genuinely the same for every symbol in a given run.
_current_market_regime = None
_current_target_trading_date = None
_index_history_cache: Dict[str, Optional[pd.DataFrame]] = {}


def _get_index_history_cached(symbol: str) -> Optional[pd.DataFrame]:
    if symbol not in _index_history_cache:
        _index_history_cache[symbol] = _load_existing_ohlcv(symbol)
    return _index_history_cache[symbol]


def fetch_and_predict(symbol: str, is_fno: bool, series: Optional[str] = None) -> Dict:
    """
    Runs in a worker thread — blocking yfinance calls + CPU-bound feature/model
    work. Never raises: failures come back as {"symbol": ..., "error": ...}.

    Tries the three-model path (src/models/three_model_orchestrator.py —
    calibrated direction + return regression + rule-based risk/liquidity,
    walk-forward validated) first; falls back to the legacy single-composite
    ModelOrchestrator if the three-model artifacts haven't been trained yet
    or fail for this symbol, same "never crash the whole scan over one
    model's issue" convention the rest of this pipeline follows.
    """
    from src.market.risk_liquidity import RiskLiquidityResult  # noqa: F401 (import kept local, mirrors rest of file's style)

    yf = YFinanceFetcher()
    try:
        ohlcv = _get_ohlcv(yf, symbol)
        if ohlcv.empty or len(ohlcv) < MIN_OHLCV_ROWS:
            return {"symbol": symbol, "error": f"insufficient OHLCV data ({len(ohlcv)} rows)"}

        fund = yf.fetch_fundamentals(symbol)

        pred = None
        from src.models.three_model_orchestrator import ThreeModelOrchestrator, artifacts_exist
        if artifacts_exist():
            try:
                nifty_hist = _get_index_history_cached("^NSEI")
                sensex_hist = _get_index_history_cached("^BSESN")
                three_model_pred = ThreeModelOrchestrator().predict(
                    symbol, ohlcv, nifty_df=nifty_hist, sensex_df=sensex_hist,
                    market_regime=_current_market_regime,
                )
                three_model_pred["target_trading_date"] = _current_target_trading_date
                pred = three_model_pred
            except Exception as exc:
                logger.debug("Three-model prediction failed for %s, falling back to legacy: %s", symbol, exc)

        if pred is None:
            orch = ModelOrchestrator()
            pred = orch.predict(symbol, ohlcv, fund, sentiment_score=0.0)

        result = {"symbol": symbol, "fund": fund, "pred": pred, "is_fno": is_fno,
                  "series": series, "ohlcv": ohlcv}
        if is_fno:
            fno_rec = FNOAnalyzer().analyze(symbol, pred, option_chain={}, oi_data={})
            result["fno_rec"] = fno_rec
        return result
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}


def _categorize_error(err: str) -> str:
    """
    Only "rate_limited" and "other" are worth retrying — a symbol with
    genuinely insufficient trading history (a recent IPO, thinly-traded
    small-cap) will fail the exact same way next time, no matter how many
    times it's retried.
    """
    if "Too Many Requests" in err or "Rate limited" in err or "429" in err:
        return "rate_limited"
    if "insufficient OHLCV" in err:
        return "insufficient_data"
    return "other"


async def _persist_price_history(db, stock_id: int, symbol: str, ohlcv) -> None:
    """
    Bulk-upsert daily OHLCV bars so the prediction pipeline (and any future
    training run) can read history straight from Postgres instead of
    re-fetching it from Yahoo every time — nothing persisted it before this.
    ON CONFLICT DO UPDATE so a re-scan refreshes bars Yahoo later revises
    rather than silently skipping them.

    Every bar is validated first (src/market/data_quality.py) — VALID/
    SUSPECT bars land in price_history with their data_quality tagged;
    INVALID bars go to price_history_quarantine instead (original values
    preserved, never silently dropped) plus a DataQualityEvent row. An
    unusual-but-arithmetically-consistent bar (a real split/circuit-limit
    day) is SUSPECT, not INVALID — see validate_ohlc_bar's docstring for
    exactly which rules reject a row outright.
    """
    if ohlcv is None or ohlcv.empty:
        return

    from src.market.data_quality import validate_ohlc_bar
    from src.database.models import DataQualityEvent, DataQualityStatus, PriceHistoryQuarantine

    valid_rows, quarantine_rows, quality_events = [], [], []
    for idx, row in ohlcv.iterrows():
        result = validate_ohlc_bar(row.get("Open"), row.get("High"), row.get("Low"), row.get("Close"), row.get("Volume"))
        bar_date = idx.date()
        if result.is_usable:
            valid_rows.append({
                "stock_id": stock_id, "date": bar_date,
                "open": float(row["Open"]), "high": float(row["High"]),
                "low": float(row["Low"]), "close": float(row["Close"]),
                "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
                "data_quality": result.status,
            })
            if result.status == DataQualityStatus.SUSPECT:
                quality_events.append({"symbol": symbol, "date": bar_date, "status": result.status,
                                        "issue_type": "; ".join(result.issues), "details": {"issues": result.issues}})
        else:
            quarantine_rows.append({
                "symbol": symbol, "date": bar_date,
                "open": row.get("Open"), "high": row.get("High"), "low": row.get("Low"),
                "close": row.get("Close"), "volume": row.get("Volume"),
                "quarantine_reason": "; ".join(result.issues),
            })
            quality_events.append({"symbol": symbol, "date": bar_date, "status": result.status,
                                    "issue_type": "; ".join(result.issues), "details": {"issues": result.issues}})

    if quarantine_rows:
        logger.warning("%s: %d bar(s) failed OHLC validation, quarantined: %s",
                        symbol, len(quarantine_rows), [r["quarantine_reason"] for r in quarantine_rows[:3]])
        await db.execute(pg_insert(PriceHistoryQuarantine).values(quarantine_rows))
    if quality_events:
        await db.execute(pg_insert(DataQualityEvent).values(quality_events))

    rows = valid_rows
    if not rows:
        return
    stmt = pg_insert(PriceHistory).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["stock_id", "date"],
        set_={
            "open": stmt.excluded.open, "high": stmt.excluded.high,
            "low": stmt.excluded.low, "close": stmt.excluded.close,
            "volume": stmt.excluded.volume, "data_quality": stmt.excluded.data_quality,
        },
    )
    await db.execute(stmt)


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

    await _persist_price_history(db, stock.id, symbol, result.get("ohlcv"))

    stock.company_name = fund.get("company_name") or stock.company_name
    stock.sector = fund.get("sector")
    stock.industry = fund.get("industry")
    stock.market_cap = fund.get("market_cap")
    stock.is_fno_eligible = result["is_fno"]
    if result.get("series"):
        stock.series = result["series"]

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

    if "decision" in pred:
        # Three-model path (src/models/three_model_orchestrator.py) — a
        # calibrated direction probability, a regression-based expected
        # return, and a rule-based risk/liquidity read, combined by the
        # transparent decision engine. `signal`/`confidence` are populated
        # too, mapped from `decision`, purely so the existing UI (built
        # against the legacy single-signal shape) keeps working without
        # every page needing a rewrite in the same pass as the model change.
        decision_to_signal = {
            DecisionType.POSITIVE_SETUP: SignalType.BUY,
            DecisionType.NEGATIVE_SETUP: SignalType.SELL,
            DecisionType.NO_CLEAR_SIGNAL: SignalType.HOLD,
        }
        db.add(Prediction(
            stock_id=stock.id,
            model_version="three_model_v2",
            feature_version=pred.get("feature_version"),
            signal=decision_to_signal[pred["decision"]],
            confidence=max(pred["probability_up"], pred["probability_down"]),
            current_price=pred.get("current_price"),
            predicted_return_1d=round(pred["expected_return"] * 100, 4) if pred.get("expected_return") is not None else None,
            target_price=pred.get("target_price"),
            stop_loss=pred.get("stop_loss"),
            probability_up=pred.get("probability_up"),
            probability_down=pred.get("probability_down"),
            probability_flat=pred.get("probability_flat"),
            expected_return=pred.get("expected_return"),
            expected_return_lower=pred.get("expected_return_lower"),
            expected_return_upper=pred.get("expected_return_upper"),
            risk_level=pred.get("risk_level"),
            liquidity_level=pred.get("liquidity_level"),
            market_regime=pred.get("market_regime"),
            decision=pred.get("decision"),
            decision_factors=pred.get("decision_factors"),
            target_trading_date=pred.get("target_trading_date"),
        ))
    else:
        # Legacy single-composite path (src/models/ensemble/predictor.py) —
        # kept as the fallback for symbols the three-model path can't yet
        # handle (e.g. artifacts not trained, or insufficient rows for the
        # newer feature set).
        db.add(Prediction(
            stock_id=stock.id,
            model_version="universe-scan-v1",
            signal=SignalType(pred["signal"]),
            confidence=pred["confidence"],
            current_price=pred.get("current_price"),
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


async def _scan_pass(
    symbol_fno_pairs: List,
    throttle: AdaptiveThrottle,
    progress_state: Dict,
    total_overall: int,
    start_time: float,
    on_progress: Optional[Callable[[Dict], None]],
    deadline: float,
) -> Dict:
    """
    One pass over a list of (symbol, is_fno, series) triples, dispatched in
    waves sized by `throttle`'s CURRENT concurrency — re-checked and
    adjusted after every wave (see AdaptiveThrottle), so a burst of
    rate-limit errors partway through a large pass shrinks concurrency for
    the rest of that same pass, not just the next retry round. `throttle`
    is shared across every pass of a run (not recreated per-round), so it
    keeps whatever it learned about current traffic conditions the whole
    time.

    `deadline` (an absolute time.time() value) is checked between waves —
    NOT just once between whole passes, which is what let a single pass
    over the full ~2,600-symbol universe run for hours past the intended
    retry budget on bad-traffic days (confirmed: real runs took 3-9+ hours
    against a 45-minute budget, because that budget was only ever checked
    after a pass returned, and a slow pass could itself run unbounded).
    Anything not yet attempted when the deadline hits is folded into
    `retryable` exactly like a failure — it gets picked up first on the
    next run via LAST_FAILED_KEY, same as a real failure would.

    Persists successes as it goes (batched); returns aggregate counts plus
    the (symbol, is_fno, series) triples worth retrying (rate-limited/other
    — not insufficient-data, which would just fail identically next time).
    """
    ok = fail = 0
    buy = sell = hold = 0
    retryable = []
    reason_counts: Dict[str, int] = {}
    BATCH = 50
    batch: List[Dict] = []
    scanned_in_pass = 0
    last_report = 0.0

    meta_by_symbol = {sym: (is_fno, series) for sym, is_fno, series in symbol_fno_pairs}
    remaining = list(symbol_fno_pairs)
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=throttle.max_workers)

    async def flush_batch():
        nonlocal batch
        if not batch:
            return
        async with AsyncSessionFactory() as db:
            for r in batch:
                await persist_result(db, r)
            await db.commit()
        batch = []

    async def report(force: bool = False):
        nonlocal last_report
        now = time.time()
        if not force and now - last_report < 2:
            return
        last_report = now
        elapsed = round(now - start_time)
        live = {
            "scanned": progress_state["scanned"] + scanned_in_pass, "total": total_overall,
            "ok": progress_state["ok"] + ok, "failed": progress_state["failed"] + fail,
            "buy": progress_state["buy"] + buy, "sell": progress_state["sell"] + sell,
            "hold": progress_state["hold"] + hold, "elapsed_s": elapsed,
            "concurrency": throttle.workers, "throttle_delay_s": round(throttle.delay_s, 2),
        }
        if on_progress:
            on_progress({"event": "progress", **live})
        await _set_status(state="running", **live)

    try:
        while remaining:
            if time.time() >= deadline:
                logger.warning(
                    "_scan_pass: deadline hit with %d symbols not yet attempted this pass — "
                    "deferring them to the next run's priority queue.", len(remaining),
                )
                for sym, is_fno, series in remaining:
                    retryable.append((sym, is_fno, series))
                remaining = []
                break

            wave_size = max(1, throttle.workers)
            wave, remaining = remaining[:wave_size], remaining[wave_size:]

            tasks = []
            for sym, is_fno, series in wave:
                if throttle.delay_s:
                    await asyncio.sleep(throttle.delay_s)
                tasks.append(loop.run_in_executor(executor, fetch_and_predict, sym, is_fno, series))

            for coro in asyncio.as_completed(tasks):
                result = await coro
                scanned_in_pass += 1
                if "error" in result:
                    fail += 1
                    reason = _categorize_error(result["error"])
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    throttle.record(reason == "rate_limited")
                    if reason in ("rate_limited", "other"):
                        is_fno, series = meta_by_symbol[result["symbol"]]
                        retryable.append((result["symbol"], is_fno, series))
                else:
                    throttle.record(False)
                    ok += 1
                    sig = result["pred"]["signal"]
                    buy += sig == "BUY"
                    sell += sig == "SELL"
                    hold += sig == "HOLD"
                    batch.append(result)
                    if len(batch) >= BATCH:
                        await flush_batch()

            await report()

        await flush_batch()
        await report(force=True)
    finally:
        executor.shutdown(wait=True)

    return {"ok": ok, "fail": fail, "buy": buy, "sell": sell, "hold": hold,
            "retryable": retryable, "reason_counts": reason_counts}


def _prioritize(
    pairs: List[Tuple[str, bool, Optional[str]]], last_failed: Dict[str, str],
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Previously-failed symbols first, then everything else (never-scanned +
    previously-successful) in their original order — so if this run's
    retry budget runs out before reaching the end, the symbols that were
    ALREADY known to be trouble got the most attempts, not the least.
    """
    failed_pairs = [p for p in pairs if p[0] in last_failed]
    rest = [p for p in pairs if p[0] not in last_failed]
    return failed_pairs + rest


async def run_universe_scan(
    limit: Optional[int] = None,
    workers: int = 8,
    universe: str = "all",
    on_progress: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Scans the full universe, previously-failed symbols first (see
    _prioritize / LAST_FAILED_KEY), using an AdaptiveThrottle that live-
    adjusts concurrency based on the actual rate-limit responses it's
    getting back — not a fixed guess. Any symbol still failing after a
    pass keeps getting retried, with exponential backoff between rounds,
    until either nothing's left to retry or MAX_RETRY_BUDGET_S is spent —
    not a fixed 2 rounds. Genuinely bad symbols (insufficient trading
    history — a recent IPO, thinly-traded small-cap) are never retried,
    since they'd fail identically again; they're reported separately from
    "failed" in the summary (excluded_insufficient_data), since retrying
    harder can't fix a real data gap, only a traffic one.

    Whatever is STILL failing when the retry budget runs out is persisted
    to LAST_FAILED_KEY, so the *next* run starts with exactly those symbols
    instead of rediscovering them last.

    Updates STATUS_KEY in Redis as it goes (so a caller that can't hold the
    request open — the API trigger — can poll progress) and also calls
    on_progress(dict), if given, for callers that can print directly (the
    CLI script).
    """
    await init_db()
    await _set_status(state="running", scanned=0, total=0, ok=0, failed=0,
                       buy=0, sell=0, hold=0, started_at=datetime.utcnow().isoformat())

    try:
        uf = NSEUniverseFetcher()
        equities = await uf.fetch_equity_symbols()
        fno_symbols = set(await uf.fetch_fno_symbols())

        series_by_symbol: Dict[str, str] = {e["yf_symbol"]: e.get("series") for e in equities}
        symbols = [e["yf_symbol"] for e in equities]
        if universe == "fno":
            symbols = [s for s in symbols if s in fno_symbols]
        if limit:
            symbols = symbols[:limit]

        # Index F&O (NIFTY, BANKNIFTY, ...) — NSE's most-traded derivatives
        # by far, but not equities at all (no SERIES value — see
        # NSEUniverseFetcher.EQUITY_SERIES), so they'd otherwise never
        # appear anywhere. Always included (they're inherently F&O-only,
        # regardless of the universe= filter) and not subject to --limit,
        # since it's a fixed set of 5.
        index_symbols = [cfg["yf_symbol"] for cfg in INDEX_FNO.values()]
        fno_symbols = fno_symbols | set(index_symbols)
        symbols = symbols + index_symbols
        total = len(symbols)

        warm = await asyncio.get_event_loop().run_in_executor(None, _warm_up_yfinance_session)
        if not warm:
            logger.warning(
                "yfinance warm-up didn't succeed after 3 attempts — Yahoo may be "
                "genuinely unreachable right now, not just rate-limited. Proceeding "
                "anyway; AdaptiveThrottle's slow start still protects the real scan."
            )

        # Computed ONCE for the whole run (not per symbol) — market regime
        # and the actual next trading session (via the real NSE calendar,
        # never a naive +1 calendar day). Read by fetch_and_predict's
        # three-model path through the module-level globals set just below.
        global _current_market_regime, _current_target_trading_date
        try:
            from src.market.regime import compute_and_store_regime
            regime_snapshot = await compute_and_store_regime()
            _current_market_regime = regime_snapshot["regime"] if regime_snapshot else None
        except Exception as exc:
            logger.warning("Market regime computation failed: %s — predictions this run won't have a regime label", exc)
            _current_market_regime = None
        try:
            from src.market.trading_calendar import next_trading_day
            _current_target_trading_date = await next_trading_day(datetime.utcnow().date())
        except Exception as exc:
            logger.warning("Trading calendar lookup failed: %s", exc)
            _current_target_trading_date = None
        _index_history_cache.clear()  # stale from a previous run otherwise

        last_failed = await _load_last_failed()
        pairs = _prioritize(
            [(sym, sym in fno_symbols, series_by_symbol.get(sym)) for sym in symbols],
            last_failed,
        )

        if on_progress:
            on_progress({
                "event": "started", "equities": len(equities), "fno_eligible": len(fno_symbols),
                "scanning": total, "workers": workers,
                "prioritized_from_last_failure": len(last_failed),
            })
        await _set_status(state="running", scanned=0, total=total, ok=0, failed=0,
                           buy=0, sell=0, hold=0, started_at=datetime.utcnow().isoformat())

        start = time.time()
        deadline = start + MAX_RETRY_BUDGET_S
        ok = fail = buy = sell = hold = 0
        reason_totals: Dict[str, int] = {}
        insufficient_data_symbols: List[str] = []
        throttle = AdaptiveThrottle(initial=workers, max_workers=max(workers * 2, 20))

        round_num = 0
        still_failing: Dict[str, str] = {}
        while pairs:
            progress_state = {"scanned": ok + fail, "ok": ok, "failed": fail,
                               "buy": buy, "sell": sell, "hold": hold}
            pass_result = await _scan_pass(pairs, throttle, progress_state, total, start, on_progress, deadline)

            ok += pass_result["ok"]
            fail += pass_result["fail"]
            buy += pass_result["buy"]
            sell += pass_result["sell"]
            hold += pass_result["hold"]
            for reason, n in pass_result["reason_counts"].items():
                reason_totals[reason] = reason_totals.get(reason, 0) + n

            retryable = pass_result["retryable"]
            round_num += 1
            elapsed_total = time.time() - start

            if not retryable:
                still_failing = {}
                break
            if round_num > MAX_RETRY_ROUNDS or elapsed_total > MAX_RETRY_BUDGET_S:
                still_failing = {sym: "rate_limited_or_transient" for sym, _, _ in retryable}
                logger.warning(
                    "%d symbols still failing after %d rounds / %ds — stopping for this run, "
                    "prioritized first next time: %s",
                    len(retryable), round_num, round(elapsed_total),
                    ", ".join(s for s, _, _ in retryable[:20]) + (" ..." if len(retryable) > 20 else ""),
                )
                break

            # These are still counted as "failed" above until a retry round
            # actually succeeds on them — back that out before re-scanning.
            fail -= len(retryable)
            cooldown = min(MAX_COOLDOWN_S, BASE_COOLDOWN_S * (2 ** (round_num - 1)))
            cooldown *= 1 + random.uniform(-0.1, 0.1)  # jitter
            logger.info(
                "Retry round %d: %d symbols, concurrency=%d, cooling down %ds first",
                round_num, len(retryable), throttle.workers, round(cooldown),
            )
            if on_progress:
                on_progress({"event": "retry_round", "round": round_num, "retrying": len(retryable),
                             "cooldown_s": round(cooldown), "concurrency": throttle.workers})
            await asyncio.sleep(cooldown)
            pairs = retryable

        await _save_last_failed(still_failing)
        # insufficient_data was counted into `fail` by _scan_pass (it's a
        # real failure, just not a traffic one) — split it back out so the
        # summary's "failed" reflects only what's still actually stuck
        # after every retry effort, and insufficient-data symbols (which no
        # amount of retrying fixes) are reported as their own category.
        insufficient_n = reason_totals.get("insufficient_data", 0)
        fail_after_retry = len(still_failing)

        logger.info("Universe scan failure breakdown: %s", reason_totals or "none")

        summary = {
            "state": "completed", "scanned": total, "total": total,
            "ok": ok, "failed": fail_after_retry,
            "excluded_insufficient_data": insufficient_n,
            "buy": buy, "sell": sell, "hold": hold,
            "failure_reasons": reason_totals,
            "retry_rounds": round_num,
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
