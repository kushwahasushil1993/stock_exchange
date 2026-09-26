"""
Market-wide context (§47-48) — NIFTY/SENSEX regime classification, computed
once per trading day by src/market/regime.py and read here rather than
recomputed per request. Backs the regime badge shown across the Dashboard
and stock detail pages, and a short regime history for a NIFTY/SENSEX page.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import MarketRegimeSnapshot, get_db
from src.api.routers.auth import get_current_user, User

logger = logging.getLogger(__name__)
router = APIRouter()


class RegimeSnapshotOut(BaseModel):
    date:               str
    regime:              str
    nifty_trend:          Optional[str] = None
    nifty_momentum:       Optional[float] = None
    nifty_volatility:     Optional[float] = None
    sensex_trend:         Optional[str] = None
    sensex_momentum:      Optional[float] = None
    sensex_volatility:    Optional[float] = None


def _out(row: MarketRegimeSnapshot) -> RegimeSnapshotOut:
    return RegimeSnapshotOut(
        date=row.date.isoformat(), regime=row.regime.value if row.regime else "UNDEFINED",
        nifty_trend=row.nifty_trend, nifty_momentum=row.nifty_momentum, nifty_volatility=row.nifty_volatility,
        sensex_trend=row.sensex_trend, sensex_momentum=row.sensex_momentum, sensex_volatility=row.sensex_volatility,
    )


@router.get("/regime", response_model=Optional[RegimeSnapshotOut])
async def get_current_regime(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Most recent market regime snapshot, or null if none has been computed yet (e.g. before the first universe scan)."""
    row = (
        await db.execute(select(MarketRegimeSnapshot).order_by(desc(MarketRegimeSnapshot.date)).limit(1))
    ).scalar_one_or_none()
    return _out(row) if row else None


@router.get("/regime/history", response_model=List[RegimeSnapshotOut])
async def get_regime_history(
    days: int = Query(default=90, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Recent regime history, most recent first — for a regime timeline on a NIFTY/SENSEX page."""
    rows = (
        await db.execute(select(MarketRegimeSnapshot).order_by(desc(MarketRegimeSnapshot.date)).limit(days))
    ).scalars().all()
    return [_out(r) for r in rows]
