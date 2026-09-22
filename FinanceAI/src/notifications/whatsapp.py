"""
Daily stock predictions over WhatsApp, via Meta's WhatsApp Cloud API.

Setup required on Meta's side before this can send anything (none of this
can be done from code — it's account/dashboard setup on developers.facebook.com):

  1. A Meta Business account + a WhatsApp Business app (developers.facebook.com
     -> My Apps -> Create App -> "Business" -> add the "WhatsApp" product).
  2. A phone number registered to that app (Meta's free test number works for
     up to 5 pre-verified recipients; a real number is needed to message
     anyone else, and requires Business verification).
  3. An access token (a short-lived token from the app dashboard is fine for
     testing; production needs a permanent token from a System User under
     Business Settings -> System Users, with whatsapp_business_messaging
     permission) and the Phone Number ID (shown on the same dashboard page)
     -> set WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID in .env.docker.
  4. A pre-approved MESSAGE TEMPLATE. This is the part that's easy to miss:
     WhatsApp Business accounts can only send free-form text to a number
     that has messaged them in the last 24 hours. An automated daily
     message like this one — nothing incoming, sent on a schedule — is a
     "business-initiated" message, which Meta requires to use a template
     that's been submitted and approved in advance (usually within a few
     hours, sometimes up to 24h). Create one at developers.facebook.com ->
     WhatsApp -> Message Templates -> Create Template, matching exactly:

       Name:     daily_stock_predictions   (or set WHATSAPP_TEMPLATE_NAME)
       Category: Utility
       Language: English (US)              (or set WHATSAPP_TEMPLATE_LANG)
       Body:     📊 FinanceAI Daily Predictions ({{1}})

                 {{2}}

                 Valid until next scan: {{3}}

     The three {{n}} placeholders are filled in by send_daily_predictions()
     below (today's date, the formatted pick list, and the next scan date)
     — the template's own wording must match what's approved, but the
     placeholder *content* is fully dynamic per send.
  5. WHATSAPP_RECIPIENT_NUMBERS in .env.docker — comma-separated, E.164
     format (e.g. "+919876543210"). Using the free test number, each one
     must also be added as a verified recipient in the Meta dashboard first.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from sqlalchemy import desc, func, select

from config.settings import settings
from src.database.models import AsyncSessionFactory, Prediction, SignalType, Stock

logger = logging.getLogger("whatsapp_notifier")

GRAPH_API_VERSION = "v20.0"
TOP_BUY_N = 10
TOP_SELL_N = 5


class WhatsAppNotifier:
    """Thin wrapper around Meta's WhatsApp Cloud API 'send template message' call."""

    def __init__(self, access_token: Optional[str] = None, phone_number_id: Optional[str] = None):
        self.access_token = access_token or settings.whatsapp_access_token
        self.phone_number_id = phone_number_id or settings.whatsapp_phone_number_id

    @property
    def configured(self) -> bool:
        return bool(self.access_token and self.phone_number_id)

    async def send_template_message(
        self, to: str, template_name: str, language: str, body_params: List[str],
    ) -> Dict[str, Any]:
        """
        Sends one WhatsApp template message. `body_params` fill the
        template's {{1}}, {{2}}, ... placeholders in order, as plain text.
        Raises RuntimeError on a non-2xx response (with Meta's own error
        body, which is usually specific enough to act on directly — e.g.
        "template not found", "recipient not in allowed list", "re-engagement
        message" for an unapproved/mismatched template).
        """
        if not self.configured:
            raise RuntimeError(
                "WhatsApp not configured — set WHATSAPP_ACCESS_TOKEN and "
                "WHATSAPP_PHONE_NUMBER_ID (see src/notifications/whatsapp.py "
                "module docstring for full setup steps)."
            )

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language},
                "components": [
                    {
                        "type": "body",
                        "parameters": [{"type": "text", "text": p} for p in body_params],
                    }
                ],
            },
        }
        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"WhatsApp send to {to} failed ({resp.status}): {data}")
                return data


def _next_scan_date(after: datetime) -> datetime:
    """
    Next weekday (Mon-Fri) after `after` — matches daily_prediction_pipeline's
    "0 10 * * 1-5" schedule (dags/pipeline_dags.py). Not schedule-aware beyond
    that (doesn't know about market holidays), just the weekday cadence.
    """
    nxt = after + timedelta(days=1)
    while nxt.weekday() >= 5:  # 5=Saturday, 6=Sunday
        nxt += timedelta(days=1)
    return nxt


async def _fetch_top_picks(top_n_buy: int, top_n_sell: int):
    """Latest prediction per stock, split into top BUY / top SELL by composite_score."""
    async with AsyncSessionFactory() as db:
        latest_sub = (
            select(Prediction.stock_id, func.max(Prediction.predicted_at).label("latest"))
            .group_by(Prediction.stock_id)
            .subquery()
        )
        result = await db.execute(
            select(Prediction, Stock.symbol, Stock.company_name)
            .join(Stock, Prediction.stock_id == Stock.id)
            .join(latest_sub, (Prediction.stock_id == latest_sub.c.stock_id) &
                               (Prediction.predicted_at == latest_sub.c.latest))
            .where(Prediction.confidence.isnot(None))
            .order_by(desc(Prediction.composite_score))
        )
        rows = result.fetchall()

    buys = [r for r in rows if r[0].signal == SignalType.BUY][:top_n_buy]
    sells = sorted(
        [r for r in rows if r[0].signal == SignalType.SELL],
        key=lambda r: r[0].composite_score or 0,
    )[:top_n_sell]
    return buys, sells


def _format_pick_list(buys, sells) -> str:
    lines = []
    if buys:
        lines.append("🟢 *BUY*")
        for i, (pred, symbol, _name) in enumerate(buys, start=1):
            target = f", Target ₹{pred.target_price:.2f}" if pred.target_price else ""
            lines.append(f"{i}. {symbol} — {pred.confidence * 100:.0f}% conf{target}")
    if sells:
        lines.append("")
        lines.append("🔴 *SELL*")
        for i, (pred, symbol, _name) in enumerate(sells, start=1):
            lines.append(f"{i}. {symbol} — {pred.confidence * 100:.0f}% conf")
    if not buys and not sells:
        lines.append("No BUY/SELL signals from today's scan — all HOLD.")
    return "\n".join(lines)


async def send_daily_predictions(
    top_n_buy: int = TOP_BUY_N,
    top_n_sell: int = TOP_SELL_N,
    recipients: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fetches today's top BUY + SELL picks and sends them as a WhatsApp
    template message to every configured recipient. Returns a per-recipient
    result summary; never raises for an individual recipient's failure (one
    bad number shouldn't block the others) — but does raise if WhatsApp
    isn't configured at all, or if there are no recipients to send to.
    """
    notifier = WhatsAppNotifier()
    if not notifier.configured:
        raise RuntimeError(
            "WhatsApp not configured — set WHATSAPP_ACCESS_TOKEN and "
            "WHATSAPP_PHONE_NUMBER_ID in .env.docker."
        )

    recipients = recipients or [
        n.strip() for n in settings.whatsapp_recipient_numbers.split(",") if n.strip()
    ]
    if not recipients:
        raise RuntimeError("No recipients configured — set WHATSAPP_RECIPIENT_NUMBERS in .env.docker.")

    buys, sells = await _fetch_top_picks(top_n_buy, top_n_sell)
    today = datetime.utcnow()
    next_scan = _next_scan_date(today)
    body_params = [
        today.strftime("%d %b %Y"),
        _format_pick_list(buys, sells),
        next_scan.strftime("%d %b %Y"),
    ]

    results = {}
    for to in recipients:
        try:
            resp = await notifier.send_template_message(
                to=to,
                template_name=settings.whatsapp_template_name,
                language=settings.whatsapp_template_lang,
                body_params=body_params,
            )
            results[to] = {"ok": True, "response": resp}
            logger.info("WhatsApp daily predictions sent to %s", to)
        except Exception as exc:
            results[to] = {"ok": False, "error": str(exc)}
            logger.error("WhatsApp send to %s failed: %s", to, exc)

    return {
        "buy_count": len(buys), "sell_count": len(sells),
        "recipients": results,
        "sent_at": today.isoformat(),
    }
