"""
Database layer: PostgreSQL (async), Redis cache, InfluxDB for OHLCV time-series.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, DateTime, Boolean, JSON, ForeignKey, Text, Enum
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


# ─── ORM Models ─────────────────────────────────────────────────────────────
class Stock(Base):
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(200))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[float]] = mapped_column(Float)
    is_fno_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType), default=AssetType.EQUITY)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="stock")
    fundamentals: Mapped[Optional["Fundamental"]] = relationship("Fundamental", back_populates="stock", uselist=False)


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
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_id: Mapped[int] = mapped_column(ForeignKey("stocks.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    signal: Mapped[SignalType] = mapped_column(Enum(SignalType))
    confidence: Mapped[float] = mapped_column(Float)
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
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    stock: Mapped["Stock"] = relationship("Stock", back_populates="predictions")


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
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(50))
    model_type: Mapped[str] = mapped_column(String(50))  # lstm | xgboost | ensemble
    symbol: Mapped[Optional[str]] = mapped_column(String(20))
    artifact_path: Mapped[str] = mapped_column(String(500))
    metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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


# ─── Engine & Session Factory ────────────────────────────────────────────────
engine = create_async_engine(
    settings.postgres_url,
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
