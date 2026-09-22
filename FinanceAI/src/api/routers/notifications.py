"""
Notification endpoints. Currently just an admin-gated manual trigger for the
WhatsApp daily-predictions message (src/notifications/whatsapp.py) — the
real, production trigger is the "notify_whatsapp" task in
dags/pipeline_dags.py's daily_prediction_pipeline DAG, which fires
automatically after every scan. This endpoint exists so the integration can
be tested/resent on demand without waiting for the next scheduled run.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.routers.auth import get_current_superuser
from src.database.models import User

router = APIRouter()


class SendWhatsAppOut(BaseModel):
    buy_count: int
    sell_count: int
    recipients: Dict[str, Any]
    sent_at: str


@router.post("/whatsapp/send-daily", response_model=SendWhatsAppOut)
async def send_daily_whatsapp(
    _admin: User = Depends(get_current_superuser),
):
    """
    Manually (re)sends today's top BUY/SELL picks over WhatsApp right now,
    to whatever numbers are in WHATSAPP_RECIPIENT_NUMBERS. Admin-only —
    this sends a real outbound message per configured recipient.
    """
    from src.notifications.whatsapp import send_daily_predictions

    try:
        return await send_daily_predictions()
    except RuntimeError as exc:
        # Not-configured / no-recipients — a client-fixable 400, not a 500.
        raise HTTPException(status_code=400, detail=str(exc))
