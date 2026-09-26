"""
Database layer: PostgreSQL (async), Redis cache, InfluxDB for OHLCV time-series.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, DateTime, Date, Boolean, JSON, ForeignKey, Text, Enum, UniqueConstraint, BigInteger
from datetime import datetime
from typing import Optional, List
import enum
import redis.asyncio as aioredis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

from config.settings import settings


# ─── SQLAlchemy Base ────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─── Enums ──────────────────────────────────────────────────────────────────
class SignalType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AssetType(str, enum.Enum):
    EQUITY = "EQUITY"
    FNO_FUTURE = "FNO_FUTURE"
    FNO_OPTION = "FNO_OPTION"
    MUTUAL_FUND = "MUTUAL_FUND"
    INDEX = "INDEX"


class RiskProfile(str, enum.Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


class PaymentMethod(str, enum.Enum):
    UPI = "UPI"
    CARD = "CARD"


class PaymentStatus(str, enum.Enum):
    CREATED = "CREATED"
    AUTHORIZED = "AUTHORIZED"
    CAPTURED = "CAPTURED"
    FAILED = "FAILED"


class DataQualityStatus(str, enum.Enum):
    """
    Every ingested row gets classified, never silently trusted or silently
    discarded. QUARANTINED rows are moved to PriceHistoryQuarantine (raw
    data preserved) rather than dropped — an unusual move can be a real
    stock split/circuit-limit day, not corruption, so nothing gets deleted
    just for failing one rule.
    """
    VALID = "VALID"
    SUSPECT = "SUSPECT"
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    QUARANTINED = "QUARANTINED"


class CorporateActionType(str, enum.Enum):
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    DIVIDEND = "DIVIDEND"
    RIGHTS = "RIGHTS"
    MERGER = "MERGER"
    DEMERGER = "DEMERGER"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"


class MarketRegime(str, enum.Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNDEFINED = "UNDEFINED"


class RiskLevel(str, enum.Enum):
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"


class LiquidityLevel(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNSUITABLE = "UNSUITABLE"


class DecisionType(str, enum.Enum):
    POSITIVE_SETUP = "POSITIVE_DIRECTIONAL_SETUP"
    NEGATIVE_SETUP = "NEGATIVE_DIRECTIONAL_SETUP"
    NO_CLEAR_SIGNAL = "NO_CLEAR_SIGNAL"


class DataFreshness(str, enum.Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    EOD = "EOD"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


# ─── ORM Models ─────────────────────────────────────────────────────────────
class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(200))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[float]] = mapped_column(Float)
    # NSE listing series: "EQ" (normal main-board trading) vs "BE"/"BZ"
    # (trade-to-trade — under surveillance for volatility/illiquidity/
    # compliance; still real NSE equities, just thinner, more volatile
    # trading worth flagging distinctly in the UI). Null for anything
    # scanned before this column existed.
    series: Mapped[Optional[str]] = mapped_column(String(4))
    is_fno_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType), default=AssetType.EQUITY)
    # Survivorship-bias documentation (§35): we can't retroactively source a
    # history of NIFTY/BSE constituents-over-time from any source available
    # here, but at minimum a delisted/renamed stock stays visible as such
    # instead of just vanishing from the universe with no record.
    listing_date: Mapped[Optional[datetime]] = mapped_column(Date)
    is_delisted: Mapped[bool] = mapped_column(Boolean, default=False)
    delisted_date: Mapped[Optional[datetime]] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="stock")
    fundamentals: Mapped[Optional["Fundamental"]] = relationship("Fundamental", back_populates="stock", uselist=False)
    price_history: Mapped[List["PriceHistory"]] = relationship("PriceHistory", back_populates="stock")
    corporate_actions: Mapped[List["CorporateAction"]] = relationship("CorporateAction", back_populates="stock")


class PriceHistory(Base):
    """
    Daily OHLCV bars, persisted once per symbol so the prediction pipeline
    reads from the DB instead of re-fetching 5y of history from Yahoo on
    every single scan/analysis call (slow, rate-limit-prone, and the reason
    nothing was ever actually stored/reusable before this table existed).

    Only rows that pass OHLC validation (src/market/data_quality.py) land
    here — anything that fails goes to PriceHistoryQuarantine instead,
    original values preserved, never silently dropped or overwritten.
    """
    __tablename__ = "price_history"
    __table_args__ = (UniqueConstraint("stock_id", "date", name="uq_price_history_stock_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    date: Mapped[datetime] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(30), default="yfinance")
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # yfinance's auto_adjust=True (used throughout this codebase) means
    # every bar stored here is already split/dividend-adjusted at the
    # source — there is no separate raw series stored. True unless a future
    # ingestion path explicitly stores raw prices.
    is_adjusted: Mapped[bool] = mapped_column(Boolean, default=True)
    data_quality: Mapped[DataQualityStatus] = mapped_column(Enum(DataQualityStatus), default=DataQualityStatus.VALID)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="price_history")


class PriceHistoryQuarantine(Base):
    """
    Rows that failed OHLC validation (src/market/data_quality.py) — kept in
    full, never deleted, so a human can review whether it was a real
    corporate-action/circuit-limit day or genuine bad data from the source.
    """
    __tablename__ = "price_history_quarantine"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), index=True)
    date: Mapped[Optional[datetime]] = mapped_column(Date, index=True)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), default="yfinance")
    quarantine_reason: Mapped[str] = mapped_column(Text)
    quarantined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DataQualityEvent(Base):
    """Audit log of every validation decision, not just the quarantined ones — lets you see how much SUSPECT/DUPLICATE traffic a symbol generates over time, not just outright failures."""
    __tablename__ = "data_quality_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), index=True)
    date: Mapped[Optional[datetime]] = mapped_column(Date)
    status: Mapped[DataQualityStatus] = mapped_column(Enum(DataQualityStatus))
    issue_type: Mapped[Optional[str]] = mapped_column(String(100))
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CorporateAction(Base):
    """
    A detected split/bonus/dividend/rights event (from yfinance's own
    actions/splits/dividends data — real source data, not inferred from
    price jumps). yfinance's auto_adjust=True already folds these into the
    stored price series correctly; this table exists so the adjustment is
    auditable rather than invisible ("why did this stock 'drop' 50% on this
    date" should have an answer on file).
    """
    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    action_type: Mapped[CorporateActionType] = mapped_column(Enum(CorporateActionType))
    action_date: Mapped[datetime] = mapped_column(Date, index=True)
    ratio: Mapped[Optional[float]] = mapped_column(Float)  # e.g. split ratio, or dividend amount
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(30), default="yfinance")
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="corporate_actions")


class TradingCalendar(Base):
    """
    The authoritative source for "is this a trading day" / "what's the next
    trading day" — a queryable table, not a hardcoded list buried in code,
    so it can be corrected without a deploy. Past dates are backfilled
    empirically from NIFTY's own actual trading history (a weekday with no
    NIFTY bar really was a holiday — real data, not a guess); future dates
    rely on a small set of fixed-date national holidays plus weekends —
    see src/market/trading_calendar.py for exactly what is and isn't
    covered, and why lunar-calendar festival holidays for future years are
    deliberately NOT guessed at.
    """
    __tablename__ = "trading_calendar"

    date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(10), primary_key=True, default="NSE")
    is_trading_day: Mapped[bool] = mapped_column(Boolean)
    holiday_name: Mapped[Optional[str]] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(30))  # "empirical" | "fixed_holiday" | "weekend"


class MarketRegimeSnapshot(Base):
    """One row per trading day — the market-wide context every stock's decision layer reads, computed once rather than redundantly per stock."""
    __tablename__ = "market_regime_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(Date, unique=True, index=True)
    regime: Mapped[MarketRegime] = mapped_column(Enum(MarketRegime))
    nifty_trend: Mapped[Optional[str]] = mapped_column(String(20))
    nifty_momentum: Mapped[Optional[float]] = mapped_column(Float)
    nifty_volatility: Mapped[Optional[float]] = mapped_column(Float)
    sensex_trend: Mapped[Optional[str]] = mapped_column(String(20))
    sensex_momentum: Mapped[Optional[float]] = mapped_column(Float)
    sensex_volatility: Mapped[Optional[float]] = mapped_column(Float)
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Fundamental(Base):
    __tablename__ = "fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), unique=True)
    pe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    pb_ratio: Mapped[Optional[float]] = mapped_column(Float)
    roe: Mapped[Optional[float]] = mapped_column(Float)
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float)
    revenue_growth_yoy: Mapped[Optional[float]] = mapped_column(Float)
    profit_margin: Mapped[Optional[float]] = mapped_column(Float)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Float)
    beta: Mapped[Optional[float]] = mapped_column(Float)
    free_cash_flow: Mapped[Optional[float]] = mapped_column(Float)
    eps_growth: Mapped[Optional[float]] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="fundamentals")


class Prediction(Base):
    """
    Immutable once written (§38) — a later model version must never rewrite
    an old row; it writes a new one. `model_version` (free-text) is kept
    for backward compatibility with rows written before model_id existed;
    new rows should populate both.
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_registry.id"))
    feature_version: Mapped[Optional[str]] = mapped_column(String(50))
    signal: Mapped[SignalType] = mapped_column(Enum(SignalType))
    confidence: Mapped[float] = mapped_column(Float)
    current_price: Mapped[Optional[float]] = mapped_column(Float)
    predicted_return_1d: Mapped[Optional[float]] = mapped_column(Float)
    predicted_return_5d: Mapped[Optional[float]] = mapped_column(Float)
    predicted_return_1m: Mapped[Optional[float]] = mapped_column(Float)
    target_price: Mapped[Optional[float]] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    risk_reward_ratio: Mapped[Optional[float]] = mapped_column(Float)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    technical_score: Mapped[Optional[float]] = mapped_column(Float)
    fundamental_score: Mapped[Optional[float]] = mapped_column(Float)
    composite_score: Mapped[Optional[float]] = mapped_column(Float)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON)
    # Model A: Direction (calibrated) — probabilities should sum to ~1.0.
    probability_up: Mapped[Optional[float]] = mapped_column(Float)
    probability_down: Mapped[Optional[float]] = mapped_column(Float)
    probability_flat: Mapped[Optional[float]] = mapped_column(Float)
    # Model B: Expected return (regression), with an interval, not a point
    # guarantee.
    expected_return: Mapped[Optional[float]] = mapped_column(Float)
    expected_return_lower: Mapped[Optional[float]] = mapped_column(Float)
    expected_return_upper: Mapped[Optional[float]] = mapped_column(Float)
    # Model C: Risk / liquidity — descriptive, threshold-based (see
    # src/market/risk_liquidity.py for the exact, documented thresholds).
    risk_level: Mapped[Optional[RiskLevel]] = mapped_column(Enum(RiskLevel))
    liquidity_level: Mapped[Optional[LiquidityLevel]] = mapped_column(Enum(LiquidityLevel))
    # Context the decision layer used, kept alongside the prediction so it's
    # auditable after the fact rather than only reconstructable from
    # separately-timestamped tables.
    market_regime: Mapped[Optional[MarketRegime]] = mapped_column(Enum(MarketRegime))
    decision: Mapped[Optional[DecisionType]] = mapped_column(Enum(DecisionType))
    decision_factors: Mapped[Optional[dict]] = mapped_column(JSON)  # {"positive": [...], "negative": [...]}
    news_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)  # article ids/headlines actually used, for leakage audit
    # The actual next trading session (via TradingCalendar), not naive
    # calendar_date + 1 — a Friday prediction's target is Monday (or
    # later, across a holiday), never Saturday.
    target_trading_date: Mapped[Optional[datetime]] = mapped_column(Date, index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="predictions")
    outcome: Mapped[Optional["PredictionOutcome"]] = relationship("PredictionOutcome", back_populates="prediction", uselist=False)


class PredictionOutcome(Base):
    """
    Filled in AFTER the target trading session actually happens (see
    src/pipelines/outcome_tracker.py) — this is what makes "the model is
    wrong more often than right" a measurable, provable statement instead
    of an impression. Never touches the Prediction row itself.
    """
    __tablename__ = "prediction_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True, index=True)
    actual_open: Mapped[Optional[float]] = mapped_column(Float)
    actual_high: Mapped[Optional[float]] = mapped_column(Float)
    actual_low: Mapped[Optional[float]] = mapped_column(Float)
    actual_close: Mapped[Optional[float]] = mapped_column(Float)
    actual_return: Mapped[Optional[float]] = mapped_column(Float)
    actual_direction: Mapped[Optional[str]] = mapped_column(String(10))  # UP | DOWN | FLAT
    prediction_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    prediction_error: Mapped[Optional[float]] = mapped_column(Float)  # |expected_return - actual_return|
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="outcome")


class FNORecommendation(Base):
    __tablename__ = "fno_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), index=True)
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType))
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    strike_price: Mapped[Optional[float]] = mapped_column(Float)
    option_type: Mapped[Optional[str]] = mapped_column(String(4))  # CE / PE
    signal: Mapped[SignalType] = mapped_column(Enum(SignalType))
    iv: Mapped[Optional[float]] = mapped_column(Float)   # Implied volatility
    delta: Mapped[Optional[float]] = mapped_column(Float)
    theta: Mapped[Optional[float]] = mapped_column(Float)
    vega: Mapped[Optional[float]] = mapped_column(Float)
    oi_change_pct: Mapped[Optional[float]] = mapped_column(Float)
    pcr: Mapped[Optional[float]] = mapped_column(Float)   # Put-Call Ratio
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MutualFund(Base):
    __tablename__ = "mutual_funds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scheme_code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    scheme_name: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(100))
    amc: Mapped[str] = mapped_column(String(200))
    nav: Mapped[Optional[float]] = mapped_column(Float)
    aum: Mapped[Optional[float]] = mapped_column(Float)
    expense_ratio: Mapped[Optional[float]] = mapped_column(Float)
    return_1y: Mapped[Optional[float]] = mapped_column(Float)
    return_3y: Mapped[Optional[float]] = mapped_column(Float)
    return_5y: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float)
    alpha: Mapped[Optional[float]] = mapped_column(Float)
    beta: Mapped[Optional[float]] = mapped_column(Float)
    composite_score: Mapped[Optional[float]] = mapped_column(Float)
    risk_profile: Mapped[RiskProfile] = mapped_column(Enum(RiskProfile), default=RiskProfile.MODERATE)
    recommendation_rank: Mapped[Optional[int]] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(100))
    url: Mapped[str] = mapped_column(String(1000), unique=True)
    content: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    symbols_mentioned: Mapped[Optional[list]] = mapped_column(JSON)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(20))
    relevance_score: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    """
    Real model versioning (§39) — this table existed before but had zero
    references anywhere in the codebase; every training run silently
    overwrote models_store/ with no record of what was trained, on what
    data, or how well it validated out-of-sample. Now the source of truth
    for champion/challenger promotion (§70): `is_active` marks the current
    production model; a challenger is only promoted after walk-forward
    validation shows a real out-of-sample improvement — never on training
    score alone.
    """
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(50))
    model_type: Mapped[str] = mapped_column(String(50))  # lstm | xgboost | lgbm | ensemble
    symbol: Mapped[Optional[str]] = mapped_column(String(20))  # null for the pooled cross-sectional model
    artifact_path: Mapped[str] = mapped_column(String(500))
    feature_version: Mapped[Optional[str]] = mapped_column(String(50))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON)
    training_period_start: Mapped[Optional[datetime]] = mapped_column(Date)
    training_period_end: Mapped[Optional[datetime]] = mapped_column(Date)
    validation_period_start: Mapped[Optional[datetime]] = mapped_column(Date)
    validation_period_end: Mapped[Optional[datetime]] = mapped_column(Date)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)  # accuracy/precision/recall/f1/roc_auc/pr_auc/log_loss/brier
    calibration_metrics: Mapped[Optional[dict]] = mapped_column(JSON)  # reliability curve points, calibrated-vs-raw
    baseline_metrics: Mapped[Optional[dict]] = mapped_column(JSON)  # previous-day/market-direction/majority-class comparisons
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_degraded: Mapped[bool] = mapped_column(Boolean, default=False)  # §41 — flagged, never hidden, on live drift
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    training_runs: Mapped[List["ModelTrainingRun"]] = relationship("ModelTrainingRun", back_populates="model")


class ModelTrainingRun(Base):
    """One row per walk-forward fold (§31) for a given model version — lets the expanding-window results be inspected fold-by-fold, not just as one averaged number."""
    __tablename__ = "model_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"), index=True)
    fold_number: Mapped[int] = mapped_column(Integer)
    train_start: Mapped[datetime] = mapped_column(Date)
    train_end: Mapped[datetime] = mapped_column(Date)
    validation_start: Mapped[datetime] = mapped_column(Date)
    validation_end: Mapped[datetime] = mapped_column(Date)
    train_rows: Mapped[Optional[int]] = mapped_column(Integer)
    validation_rows: Mapped[Optional[int]] = mapped_column(Integer)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped["ModelRegistry"] = relationship("ModelRegistry", back_populates="training_runs")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    risk_profile: Mapped[RiskProfile] = mapped_column(Enum(RiskProfile), default=RiskProfile.MODERATE)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    payments: Mapped[List["PaymentTransaction"]] = relationship(
        "PaymentTransaction", back_populates="user"
    )


class PaymentTransaction(Base):
    __tablename__ = "payment_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    gateway: Mapped[str] = mapped_column(String(30), default="RAZORPAY")
    order_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    payment_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    signature: Mapped[Optional[str]] = mapped_column(String(255))
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(10), default="INR")
    method: Mapped[PaymentMethod] = mapped_column(Enum(PaymentMethod))
    status: Mapped[PaymentStatus] = mapped_column(Enum(PaymentStatus), default=PaymentStatus.CREATED)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="payments")


def _asyncpg_url(url: str) -> str:
    """
    Managed Postgres add-ons (Render, Railway, Heroku, ...) hand back a plain
    postgresql:// (or the older postgres://) connection string with no
    driver — create_async_engine requires the +asyncpg dialect explicitly.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix) and "+asyncpg" not in url.split("://", 1)[0]:
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


# ─── Engine & Session Factory ────────────────────────────────────────────────
engine = create_async_engine(
    _asyncpg_url(settings.postgres_url),
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionFactory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncSession:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ─── Redis Client ────────────────────────────────────────────────────────────
class RedisCache:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = aioredis.from_url(
            settings.redis_url, encoding="utf-8", decode_responses=True
        )

    async def get(self, key: str):
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int = 300):
        await self._client.setex(key, ttl, value)

    async def delete(self, key: str):
        await self._client.delete(key)

    async def delete_pattern(self, pattern: str) -> int:
        """
        Deletes every key matching a glob pattern (e.g. "top_picks:*") —
        for cache entries keyed by a parameter (top_n, symbol, ...) where
        the exact key(s) written aren't known by the invalidating caller.
        Uses SCAN (cursor-based), not KEYS — doesn't block Redis on a large
        keyspace. Returns the number of keys deleted.
        """
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                deleted += await self._client.delete(*keys)
            if cursor == 0:
                break
        return deleted

    async def close(self):
        await self._client.aclose()


redis_cache = RedisCache()


# ─── InfluxDB Client ─────────────────────────────────────────────────────────
async def get_influx_client() -> InfluxDBClientAsync:
    return InfluxDBClientAsync(
        url=settings.influx_url,
        token=settings.influx_token,
        org=settings.influx_org,
    )
