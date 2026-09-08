"""Health check router."""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "service": "FinanceAI"}


@router.get("/ready")
async def readiness():
    """Readiness probe: check DB and Redis."""
    from src.database.models import engine, redis_cache
    checks = {}
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = str(e)

    try:
        await redis_cache._client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = str(e)

    healthy = all(v == "ok" for v in checks.values())
    return {"status": "ready" if healthy else "degraded", "checks": checks}
