"""Admin router — user management and system stats. Superuser-only."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routers.auth import get_current_superuser
from src.database.models import (
    FNORecommendation,
    MutualFund,
    Prediction,
    Stock,
    User,
    get_db,
)
from src.pipelines.universe_scan import get_status as get_universe_scan_status

router = APIRouter()


class AdminUserOut(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    risk_profile: str
    is_active: bool
    is_superuser: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class StatsOut(BaseModel):
    total_users: int
    total_stocks_tracked: int
    total_predictions: int
    total_fno_recommendations: int
    total_mutual_funds_cataloged: int
    universe_scan: dict


@router.get("/users", response_model=List[AdminUserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


@router.patch("/users/{user_id}", response_model=AdminUserOut)
async def update_user(
    user_id: int,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_superuser),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.is_active is not None:
        if user.id == admin.id and not payload.is_active:
            raise HTTPException(status_code=400, detail="You can't deactivate your own account")
        user.is_active = payload.is_active
    if payload.is_superuser is not None:
        if user.id == admin.id and not payload.is_superuser:
            raise HTTPException(status_code=400, detail="You can't remove your own admin access")
        user.is_superuser = payload.is_superuser

    await db.flush()
    return user


@router.get("/stats", response_model=StatsOut)
async def stats(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_superuser),
):
    async def count(model) -> int:
        result = await db.execute(select(func.count()).select_from(model))
        return result.scalar_one()

    return StatsOut(
        total_users=await count(User),
        total_stocks_tracked=await count(Stock),
        total_predictions=await count(Prediction),
        total_fno_recommendations=await count(FNORecommendation),
        total_mutual_funds_cataloged=await count(MutualFund),
        universe_scan=await get_universe_scan_status(),
    )
