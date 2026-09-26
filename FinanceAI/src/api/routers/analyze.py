"""
On-demand analysis — give it a stock/F&O/mutual-fund name (or ticker/scheme
code), it fetches fresh data, checks the news, runs the prediction engine,
and answers "buy or not" directly. Unlike /predictions/top-picks (reads
already-computed rows from the DB), everything here runs live per request.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.routers.auth import get_current_user, User
from src.api.routers.predictions import ThreeModelOut
from src.database.models import redis_cache

logger = logging.getLogger(__name__)
router = APIRouter()

_EQUITY_CACHE_KEY = "analyze:equity_universe"
_FNO_CACHE_KEY = "analyze:fno_universe"
_MF_CACHE_KEY = "analyze:mf_universe"
_CACHE_TTL = 24 * 3600  # NSE's listed-symbol file / AMFI's scheme list change rarely


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

    from src.ingestion.data_fetcher import INDEX_FNO

    cached = await redis_cache.get(_FNO_CACHE_KEY)
    if cached:
        return set(json.loads(cached)) | {cfg["yf_symbol"] for cfg in INDEX_FNO.values()}

    from src.ingestion.data_fetcher import NSEUniverseFetcher
    symbols = await NSEUniverseFetcher().fetch_fno_symbols()
    await redis_cache.set(_FNO_CACHE_KEY, json.dumps(symbols), ttl=_CACHE_TTL)
    return set(symbols) | {cfg["yf_symbol"] for cfg in INDEX_FNO.values()}


async def _get_mf_universe() -> List[Dict[str, str]]:
    """Lightweight {scheme_code, scheme_name, category, amc} for ~14k AMFI
    schemes — cached separately from the full NAV DataFrame /analyze/mutual-
    fund fetches, since suggestions only need names, not live NAVs."""
    import json

    cached = await redis_cache.get(_MF_CACHE_KEY)
    if cached:
        return json.loads(cached)

    from src.ingestion.data_fetcher import AMFIFetcher
    nav_df = await AMFIFetcher().fetch_all_navs()
    records = (
        nav_df[["scheme_code", "scheme_name", "category", "amc"]]
        .dropna(subset=["scheme_code", "scheme_name"])
        .to_dict("records")
    )
    await redis_cache.set(_MF_CACHE_KEY, json.dumps(records), ttl=_CACHE_TTL)
    return records


async def _resolve_equity(query: str) -> Dict[str, str]:
    """
    Accept a ticker (RELIANCE / RELIANCE.NS), a company name (Reliance
    Industries), or an index F&O name (NIFTY, Bank Nifty) and resolve it to
    {symbol, yf_symbol, company_name}.
    """
    from src.ingestion.data_fetcher import INDEX_FNO

    q = query.strip()
    q_upper = q.upper().replace(".NS", "")

    if q_upper in INDEX_FNO:
        return {"symbol": q_upper, **INDEX_FNO[q_upper]}
    # The value stored as Stock.symbol / used in frontend links is the
    # yf_symbol ("^NSEI", or "NIFTY_FIN_SERVICE.NS" for FINNIFTY) — a query
    # of that exact value (from clicking a symbol link) needs to match here
    # too. Compared against the un-stripped query — q_upper already had
    # ".NS" removed above, which would wrongly break FINNIFTY's yf_symbol
    # (the only one of the 5 actually shaped like a regular ".NS" ticker).
    q_raw_upper = q.upper()
    for key, cfg in INDEX_FNO.items():
        if q_raw_upper == cfg["yf_symbol"].upper():
            return {"symbol": key, **cfg}
    # Word-set match, not substring — "Bank Nifty" (the common colloquial
    # order) wouldn't otherwise match "Nifty Bank" (Yahoo's official name
    # order); comparing word sets is order-independent.
    q_words = set(q.lower().split())
    for key, cfg in INDEX_FNO.items():
        cfg_words = set(cfg["company_name"].lower().split())
        if q_words and q_words.issubset(cfg_words):
            return {"symbol": key, **cfg}

    universe = await _get_equity_universe()
    for e in universe:
        if e["symbol"] == q_upper:
            return e
    for e in universe:
        if q.lower() in e["company_name"].lower():
            return e

    raise HTTPException(status_code=404, detail=f"No NSE-listed equity or index matching '{query}'")


def _risk_reward_ratio(current_price: float, target_price: float, stop_loss: float) -> float:
    upside = abs(target_price - current_price)
    downside = abs(current_price - stop_loss)
    return round(upside / downside, 2) if downside > 0 else 0.0


async def _run_three_model(yf_symbol: str, ohlcv, fund: Dict[str, Any]) -> Dict[str, Any]:
    """
    The properly walk-forward-validated three-model system (calibrated
    direction probabilities, expected-return regression, rule-based risk/
    liquidity, transparent decision engine) — the SAME system
    src/pipelines/universe_scan.py uses for the daily scan, now also
    backing on-demand analysis. Raises on any failure (insufficient data,
    stale/mismatched artifacts) — the caller (_run_prediction) is expected
    to catch this and fall back to the legacy model, same convention as
    universe_scan.py's fetch_and_predict.
    """
    from src.database.models import MarketRegime
    from src.market.regime import get_latest_regime
    from src.market.sector_index import get_sector_index_symbol
    from src.models.three_model_orchestrator import ThreeModelOrchestrator
    from src.pipelines.universe_scan import _load_existing_ohlcv, _tz_naive

    # ohlcv here is a fresh, tz-AWARE live yfinance fetch (unlike
    # universe_scan.py's own OHLCV, already tz-stripped by the time it
    # reaches this point) — nifty/sensex/sector history loaded from
    # Postgres below is tz-naive (the column has no timezone concept at
    # all), and joining tz-aware with tz-naive raises outright. Confirmed
    # directly, not assumed — same class of bug this codebase already hit
    # and fixed once for universe_scan.py's own merge logic.
    ohlcv = _tz_naive(ohlcv)

    loop = asyncio.get_event_loop()
    sector_symbol = get_sector_index_symbol(fund.get("sector"))
    nifty_df, sensex_df, sector_df = await asyncio.gather(
        loop.run_in_executor(None, _load_existing_ohlcv, "^NSEI"),
        loop.run_in_executor(None, _load_existing_ohlcv, "^BSESN"),
        loop.run_in_executor(None, _load_existing_ohlcv, sector_symbol) if sector_symbol else asyncio.sleep(0, result=None),
    )

    regime_snapshot = await get_latest_regime()
    market_regime = MarketRegime(regime_snapshot["regime"]) if regime_snapshot and regime_snapshot.get("regime") else None

    pred = await loop.run_in_executor(
        None,
        lambda: ThreeModelOrchestrator().predict(
            yf_symbol, ohlcv, nifty_df=nifty_df, sensex_df=sensex_df, sector_df=sector_df, market_regime=market_regime,
        ),
    )

    from src.database.models import DecisionType
    decision_to_signal = {DecisionType.POSITIVE_SETUP: "BUY", DecisionType.NEGATIVE_SETUP: "SELL", DecisionType.NO_CLEAR_SIGNAL: "HOLD"}

    return {
        "current_price": pred["current_price"], "target_price": pred["target_price"], "stop_loss": pred["stop_loss"],
        "signal": decision_to_signal[pred["decision"]],
        "confidence": round(max(pred["probability_up"], pred["probability_down"]), 4),
        "expected_return_pct": round(pred["expected_return"] * 100, 4),
        "return_horizon_days": 1,
        "technical_score": None,  # the three-model system doesn't compute a single composite technical score — never fabricate one
        "fundamental_score": None,
        "predicted_at": pred["predicted_at"],
        "three_model": {
            "decision": pred["decision"].value,
            "decision_factors": pred["decision_factors"],
            "probability_up": pred["probability_up"], "probability_down": pred["probability_down"], "probability_flat": pred["probability_flat"],
            "expected_return": pred["expected_return"], "expected_return_lower": pred["expected_return_lower"], "expected_return_upper": pred["expected_return_upper"],
            "risk_level": pred["risk_level"].value if pred["risk_level"] else None,
            "liquidity_level": pred["liquidity_level"].value if pred["liquidity_level"] else None,
            "market_regime": pred["market_regime"].value if pred["market_regime"] else None,
            "target_trading_date": None,
            "model_version": pred.get("model_type"),
            "feature_version": pred.get("feature_version"),
        },
    }


async def _run_prediction(yf_symbol: str, ohlcv, fund: Dict[str, Any], sentiment_score: float) -> Dict[str, Any]:
    """
    Tries the three-model system first, falling back to the legacy
    single-composite ModelOrchestrator only if the three-model artifacts
    aren't trained yet or fail for this symbol. This endpoint used to call
    ONLY the legacy model unconditionally — which is also what made it
    break outright the moment the legacy model's separately-trained
    artifact fell out of sync with the shared feature pipeline (a real
    incident: swing-structure/sector features were added to the shared
    XGBOOST_FEATURES list, the three-model artifacts were retrained to
    match, but this endpoint's legacy model was not, and every call here
    started raising "Feature shape mismatch"). Trying the properly-
    retrained three-model system first removes that maintenance trap
    rather than just papering over its latest occurrence.

    Returns a dict with a STABLE shape regardless of which path ran:
    current_price/target_price/stop_loss/signal/confidence/
    expected_return_pct/return_horizon_days/technical_score(nullable)/
    fundamental_score(nullable)/predicted_at/three_model(nullable dict).
    """
    from src.models.three_model_orchestrator import artifacts_exist

    if artifacts_exist():
        try:
            return await _run_three_model(yf_symbol, ohlcv, fund)
        except Exception as exc:
            logger.debug("Three-model prediction failed for %s, falling back to legacy: %s", yf_symbol, exc)

    from src.models.ensemble.predictor import ModelOrchestrator
    pred = ModelOrchestrator().predict(yf_symbol, ohlcv, fund, sentiment_score)
    return {
        "current_price": pred["current_price"], "target_price": pred["target_price"], "stop_loss": pred["stop_loss"],
        "signal": pred["signal"], "confidence": pred["confidence"],
        "expected_return_pct": pred["predicted_return_5d"], "return_horizon_days": 5,
        "technical_score": pred["technical_score"], "fundamental_score": pred["fundamental_score"],
        "predicted_at": pred["predicted_at"], "three_model": None,
    }


# ─── Shared stock analysis ───────────────────────────────────────────────────
async def _analyze_equity(query: str) -> Dict[str, Any]:
    from src.ingestion.data_fetcher import YFinanceFetcher, NewsFetcher
    from src.sentiment.analyzer import SentimentAnalyzer

    match = await _resolve_equity(query)
    yf_symbol = match["yf_symbol"]

    from src.pipelines.universe_scan import MIN_OHLCV_ROWS

    yf = YFinanceFetcher()
    ohlcv = yf.fetch_ohlcv(yf_symbol, period="5y")
    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"No price data available for {yf_symbol}")
    if len(ohlcv) < MIN_OHLCV_ROWS:
        # Same threshold the universe scan uses to skip a symbol outright
        # (src/pipelines/universe_scan.py) — below this, technical indicators
        # compute as NaN, which the prediction model then propagates through
        # rather than raising, and NaN isn't valid JSON (crashes the response
        # with a 500 instead of a clean error). Confirmed hit by FINNIFTY
        # (NIFTY_FIN_SERVICE.NS), which Yahoo has only ~1 row of history for.
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient price history for {yf_symbol} "
                f"({len(ohlcv)} rows, need at least {MIN_OHLCV_ROWS}) — "
                "prediction requires more data than is currently available."
            ),
        )
    fund = yf.fetch_fundamentals(yf_symbol)

    # News: search by company name directly rather than the existing
    # symbol-extraction sentiment pipeline, which only recognizes ~50
    # hardcoded large-cap symbols (src/sentiment/analyzer.py NSE_SYMBOLS) —
    # not useful for the other ~2,250 NSE-listed names.
    #
    # An unquoted "{company_name} share price" query (the old approach)
    # confirmed pulling in completely unrelated articles for real
    # symbols — a Baozun share-buyback story, a Rocket Lab acquisition —
    # matched only because they contained the word "share". NewsAPI treats
    # unquoted queries as a loose OR/relevance match across individual
    # words, not "this article is actually about this company". Fixed two
    # ways: an exact-phrase query (dropping the "Limited"/"Ltd" suffix,
    # since headlines almost always drop it too — keeping it would make
    # the phrase match too narrow) plus a hard post-filter requiring the
    # company name to actually appear in the article text, independent of
    # whatever NewsAPI itself decided was a match.
    import re as _re
    core_name = _re.sub(r"\s+(Limited|Ltd\.?|Pvt\.?|Private)\s*$", "", match["company_name"], flags=_re.IGNORECASE).strip()
    news = NewsFetcher()
    raw_articles = await news.fetch_newsapi(
        query=f'"{core_name}"',
        days_back=7,
        page_size=20,
    )
    core_name_lower = core_name.lower()
    articles = [
        a for a in raw_articles
        if core_name_lower in f"{a.get('title', '')} {a.get('description') or ''}".lower()
    ]
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

    pred = await _run_prediction(yf_symbol, ohlcv, fund, sentiment_score)

    return {
        "symbol": yf_symbol,
        "company_name": match["company_name"],
        "current_price": pred["current_price"],
        "target_price": pred["target_price"],
        "stop_loss": pred["stop_loss"],
        "expected_return_pct": pred["expected_return_pct"],
        "return_horizon_days": pred["return_horizon_days"],
        "signal": pred["signal"],
        "recommendation": {"BUY": "Buy", "SELL": "Don't buy — sell/avoid", "HOLD": "Hold — no strong edge either way"}[pred["signal"]],
        "confidence": pred["confidence"],
        "risk_reward_ratio": _risk_reward_ratio(pred["current_price"], pred["target_price"], pred["stop_loss"]),
        "technical_score": pred["technical_score"],
        "fundamental_score": pred["fundamental_score"],
        "sentiment_score": sentiment_score,
        "news": {
            "articles_analyzed": len(articles),
            "headlines": headlines,
        },
        "prediction_horizon_note": (
            f"expected_return_pct is a {pred['return_horizon_days']}-trading-day forecast, computed from 5 years "
            "of price history — not a multi-year return forecast."
        ),
        "predicted_at": pred["predicted_at"],
        "three_model": pred["three_model"],
        "_pred": pred,  # internal, stripped before response — used by analyze_fno's FNOAnalyzer call
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
    expected_return_pct: float
    return_horizon_days: int
    signal: str
    recommendation: str
    confidence: float
    risk_reward_ratio: float
    # Only populated on the legacy fallback path — the three-model system
    # doesn't compute a single composite technical/fundamental score, and
    # this never fabricates one just to fill the field.
    technical_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    sentiment_score: float
    news: NewsOut
    prediction_horizon_note: str
    predicted_at: str
    # Populated only when the three-model system actually produced this
    # prediction (None on the legacy fallback path) — see _run_prediction.
    three_model: Optional[ThreeModelOut] = None


def _nse_chain_to_analyzer_shape(nse_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    NSE's option-chain-equities API returns its own JSON schema (records.data
    -> list of {strikePrice, CE: {...}, PE: {...}}) — translate it into the
    {"calls": [...], "puts": [...], "expiry": ...} shape FNOAnalyzer expects
    (modeled on yfinance's option_chain() DataFrame columns: strike,
    openInterest, impliedVolatility as a 0-1 fraction — NSE reports IV as a
    0-100 percentage, so it's divided down to match).
    """
    records = nse_json.get("records", {}) or {}
    data = records.get("data", []) or []
    calls, puts = [], []
    for row in data:
        strike = row.get("strikePrice")
        ce, pe = row.get("CE"), row.get("PE")
        if ce:
            calls.append({
                "strike": strike,
                "openInterest": ce.get("openInterest", 0) or 0,
                "impliedVolatility": (ce.get("impliedVolatility") or 0) / 100,
            })
        if pe:
            puts.append({
                "strike": strike,
                "openInterest": pe.get("openInterest", 0) or 0,
                "impliedVolatility": (pe.get("impliedVolatility") or 0) / 100,
            })
    expiry_dates = records.get("expiryDates") or []
    return {"calls": calls, "puts": puts, "expiry": expiry_dates[0] if expiry_dates else None}


async def _fetch_real_option_chain(symbol_clean: str, yf_symbol: str) -> Dict[str, Any]:
    """
    Best-effort real option chain + OI data, in order:
      1. NSE's live API — needs a valid browser-captured session cookie
         (NSE_COOKIE in .env; Akamai bot-detection blocks a plain server
         request otherwise). Cookies expire within hours.
      2. Yahoo Finance — costs nothing to try, but in practice Yahoo doesn't
         publish options chain data for NSE-listed (India) equities at all,
         so this is expected to come back empty for every Indian symbol.
      3. Neither available — FNOAnalyzer still runs on signal + confidence
         from our own prediction engine, just without real IV/PCR/max-pain.

    Always returns {"chain": {...}, "oi": {...}, "source": "nse"|"yahoo"|"none"}
    so callers can tell real market data from a neutral-default fallback
    instead of presenting one as if it were the other.
    """
    from src.ingestion.data_fetcher import NSEFetcher, YFinanceFetcher

    nse = NSEFetcher()
    try:
        raw_chain = await nse.fetch_option_chain(symbol_clean)
        if raw_chain and raw_chain.get("records", {}).get("data"):
            oi_raw = await nse.fetch_fno_oi(symbol_clean)
            return {"chain": _nse_chain_to_analyzer_shape(raw_chain), "oi": oi_raw or {}, "source": "nse"}
    except Exception as exc:
        logger.warning("NSE option chain fetch failed for %s: %s", symbol_clean, exc)

    try:
        loop = asyncio.get_event_loop()
        yf_chain = await loop.run_in_executor(None, YFinanceFetcher().fetch_options_chain, yf_symbol)
        if yf_chain:
            return {"chain": yf_chain, "oi": {}, "source": "yahoo"}
    except Exception as exc:
        logger.warning("Yahoo option chain fetch failed for %s: %s", symbol_clean, exc)

    return {"chain": {}, "oi": {}, "source": "none"}


class FNOAnalysisOut(StockAnalysisOut):
    fno_strategy: str
    fno_risk_level: str
    fno_rationale: str
    strike_price: Optional[float]
    option_type: Optional[str]
    expiry: Optional[str]
    iv: Optional[float]
    pcr: float
    oi_change_pct: float
    max_pain: Optional[float]
    option_data_source: str  # "nse" | "yahoo" | "none" — see _fetch_real_option_chain


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


class SuggestOut(BaseModel):
    value: str   # what to pass back into ?query= on /stock, /fno, or /mutual-fund
    label: str   # what to show the user


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/suggest", response_model=List[SuggestOut])
async def suggest(
    asset_type: str = Query(..., alias="type", pattern="^(stock|fno|mutual-fund)$"),
    query: str = Query(..., min_length=1),
    limit: int = Query(default=8, ge=1, le=20),
    _user: User = Depends(get_current_user),
):
    """
    Typeahead for the Analyze page's search box — searches server-side
    against the cached NSE equity/F&O list or AMFI scheme list (~2,300 and
    ~14,000 entries respectively; too much to ship to the browser for
    client-side filtering) and returns a short ranked list of matches.
    """
    q = query.strip().lower()
    if not q:
        return []

    if asset_type in ("stock", "fno"):
        universe = await _get_equity_universe()
        if asset_type == "fno":
            from src.ingestion.data_fetcher import INDEX_FNO

            fno_symbols = await _get_fno_universe()
            universe = [e for e in universe if e["yf_symbol"] in fno_symbols]
            # Index F&O aren't in the equity universe at all (they're
            # derivatives, not SERIES=='EQ' listings) — add them as
            # candidates directly so NIFTY/BANKNIFTY/etc. actually show up
            # in the F&O tab's autocomplete.
            universe = universe + [
                {"symbol": key, "yf_symbol": cfg["yf_symbol"], "company_name": cfg["company_name"]}
                for key, cfg in INDEX_FNO.items()
            ]

        matches = [
            e for e in universe
            if q in e["symbol"].lower() or q in e["company_name"].lower()
        ]
        matches.sort(key=lambda e: (
            0 if e["symbol"].lower().startswith(q) or e["company_name"].lower().startswith(q) else 1,
            e["company_name"],
        ))
        return [
            SuggestOut(value=e["yf_symbol"], label=f"{e['company_name']} ({e['symbol']})")
            for e in matches[:limit]
        ]

    # mutual-fund
    universe = await _get_mf_universe()
    matches = [
        m for m in universe
        if q in m["scheme_name"].lower() or q in m["scheme_code"].lower() or q in m["amc"].lower()
    ]
    matches.sort(key=lambda m: (
        0 if m["scheme_name"].lower().startswith(q) else 1,
        m["scheme_name"],
    ))
    # AMFI lists separate rows per plan/option (Direct/Regular, Growth/IDCW)
    # under the same scheme_name — /analyze/mutual-fund resolves by
    # scheme_name substring match anyway (takes the first hit), so dedupe
    # here rather than showing 3-4 near-identical entries per fund.
    seen = set()
    out = []
    for m in matches:
        if m["scheme_name"] in seen:
            continue
        seen.add(m["scheme_name"])
        out.append(SuggestOut(value=m["scheme_name"], label=f"{m['scheme_name']} — {m['amc']}"))
        if len(out) >= limit:
            break
    return out


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
    """
    Same as /analyze/stock, plus a derived F&O (futures/options) strategy —
    using real option chain data (strike-level OI/IV, PCR, max pain) when
    it's actually available (see _fetch_real_option_chain), not silently
    substituted neutral defaults presented as if they were real.
    """
    from src.recommendation.engine import FNOAnalyzer

    fno_universe = await _get_fno_universe()
    result = await _analyze_equity(query)
    if result["symbol"] not in fno_universe:
        raise HTTPException(
            status_code=400,
            detail=f"{result['symbol']} is not F&O-eligible on NSE (use /analyze/stock instead)",
        )

    symbol_clean = result["symbol"].replace(".NS", "")
    option_data = await _fetch_real_option_chain(symbol_clean, result["symbol"])

    fno_rec = FNOAnalyzer().analyze(
        result["symbol"], result["_pred"], option_data["chain"], option_data["oi"],
    )
    return FNOAnalysisOut(
        **_strip_internal(result),
        fno_strategy=fno_rec.strategy,
        fno_risk_level=fno_rec.risk_level,
        fno_rationale=fno_rec.rationale,
        strike_price=fno_rec.strike_price,
        option_type=fno_rec.option_type,
        expiry=fno_rec.expiry,
        iv=fno_rec.iv,
        pcr=fno_rec.pcr,
        oi_change_pct=fno_rec.oi_change_pct,
        max_pain=fno_rec.max_pain,
        option_data_source=option_data["source"],
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
