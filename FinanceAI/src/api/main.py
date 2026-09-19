"""
FastAPI application entry point.
Registers all routers, middleware, startup/shutdown hooks.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

from config.settings import settings
from src.database.models import init_db, redis_cache
from src.api.routers import stocks, fno, mutual_funds, predictions, auth, health
# from src.api.routers import payments  # Razorpay integration — disabled until needed
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.logging_mw import RequestLoggingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 FinanceAI starting up…")
    await init_db()
    await redis_cache.connect()
    logger.info("✅ Database and Redis connected")
    yield
    logger.info("🛑 FinanceAI shutting down…")
    await redis_cache.close()


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinanceAI – AI-Powered Stock & MF Intelligence",
    description=(
        "Production-grade ML system for Indian equity markets:\n"
        "- Stock return predictions (Equity, FNO)\n"
        "- Mutual fund recommendations\n"
        "- Real-time sentiment analysis\n"
        "- Ensemble model (LSTM + XGBoost + LightGBM)"
    ),
    version=settings.version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)
app.add_middleware(RequestLoggingMiddleware)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(auth.router,         prefix="/api/v1/auth",         tags=["Authentication"])
app.include_router(health.router,       prefix="/api/v1",              tags=["Health"])
app.include_router(stocks.router,       prefix="/api/v1/stocks",       tags=["Stocks"])
app.include_router(predictions.router,  prefix="/api/v1/predictions",  tags=["Predictions"])
# app.include_router(payments.router,     prefix="/api/v1/payments",     tags=["Payments"])  # Razorpay integration — disabled until needed
app.include_router(fno.router,          prefix="/api/v1/fno",          tags=["FNO"])
app.include_router(mutual_funds.router, prefix="/api/v1/mutual-funds", tags=["Mutual Funds"])


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        log_level="info",
    )
