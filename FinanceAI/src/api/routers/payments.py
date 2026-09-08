"""Payments router — Razorpay order creation and payment verification."""
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Optional

import razorpay
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from src.api.routers.auth import User, get_current_user
from src.database.models import (
    PaymentMethod,
    PaymentStatus,
    PaymentTransaction,
    get_db,
)

router = APIRouter()


class PaymentMethodOut(BaseModel):
    code: str
    title: str
    description: str


class CreateOrderRequest(BaseModel):
    amount: float = Field(gt=0)
    method: str = Field(pattern="^(UPI|CARD)$")
    description: Optional[str] = None
    notes: Dict[str, str] = Field(default_factory=dict)


class CreateOrderResponse(BaseModel):
    gateway: str
    key_id: str
    order_id: str
    amount: int
    currency: str
    method: str
    checkout_options: Dict[str, List[str]]
    created_at: str


class VerifyPaymentRequest(BaseModel):
    order_id: str
    payment_id: str
    signature: str
    status: str = Field(default="CAPTURED", pattern="^(AUTHORIZED|CAPTURED|FAILED)$")


class PaymentOut(BaseModel):
    order_id: str
    payment_id: Optional[str]
    amount: float
    currency: str
    method: str
    status: str
    created_at: str
    updated_at: str


def _get_razorpay_client() -> razorpay.Client:
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(
            status_code=500,
            detail="Razorpay keys are not configured",
        )
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


def _verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    payload = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(
        settings.razorpay_key_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.get("/methods", response_model=List[PaymentMethodOut])
async def payment_methods(_user: User = Depends(get_current_user)):
    return [
        PaymentMethodOut(
            code="UPI",
            title="UPI",
            description="Pay via UPI apps like GPay, PhonePe, Paytm",
        ),
        PaymentMethodOut(
            code="CARD",
            title="Credit/Debit Card",
            description="Pay using Visa, Mastercard, RuPay cards",
        ),
    ]


@router.post("/orders", response_model=CreateOrderResponse, status_code=201)
async def create_order(
    req: CreateOrderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    method = PaymentMethod(req.method)
    amount_paise = int(round(req.amount * 100))

    client = _get_razorpay_client()
    order = client.order.create({
        "amount": amount_paise,
        "currency": settings.payment_currency,
        "payment_capture": 1,
        "notes": req.notes,
    })

    tx = PaymentTransaction(
        user_id=current_user.id,
        gateway="RAZORPAY",
        order_id=order["id"],
        amount=req.amount,
        currency=settings.payment_currency,
        method=method,
        status=PaymentStatus.CREATED,
        metadata_json={
            "description": req.description,
            "notes": req.notes,
            "provider_order_payload": order,
        },
    )
    db.add(tx)
    await db.flush()

    checkout_options = {"supported": ["UPI", "CARD"]}
    return CreateOrderResponse(
        gateway="RAZORPAY",
        key_id=settings.razorpay_key_id,
        order_id=order["id"],
        amount=amount_paise,
        currency=settings.payment_currency,
        method=method.value,
        checkout_options=checkout_options,
        created_at=datetime.utcnow().isoformat(),
    )


@router.post("/verify", response_model=PaymentOut)
async def verify_payment(
    req: VerifyPaymentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    res = await db.execute(
        select(PaymentTransaction).where(PaymentTransaction.order_id == req.order_id)
    )
    tx = res.scalar_one_or_none()
    if not tx or tx.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Payment order not found")

    if not _verify_signature(req.order_id, req.payment_id, req.signature):
        tx.status = PaymentStatus.FAILED
        tx.payment_id = req.payment_id
        tx.signature = req.signature
        tx.updated_at = datetime.utcnow()
        await db.flush()
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    status_map = {
        "AUTHORIZED": PaymentStatus.AUTHORIZED,
        "CAPTURED": PaymentStatus.CAPTURED,
        "FAILED": PaymentStatus.FAILED,
    }
    tx.status = status_map[req.status]
    tx.payment_id = req.payment_id
    tx.signature = req.signature
    tx.updated_at = datetime.utcnow()
    await db.flush()

    return PaymentOut(
        order_id=tx.order_id,
        payment_id=tx.payment_id,
        amount=tx.amount,
        currency=tx.currency,
        method=tx.method.value,
        status=tx.status.value,
        created_at=tx.created_at.isoformat(),
        updated_at=tx.updated_at.isoformat(),
    )


@router.get("/transactions", response_model=List[PaymentOut])
async def my_transactions(
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    res = await db.execute(
        select(PaymentTransaction)
        .where(PaymentTransaction.user_id == current_user.id)
        .order_by(desc(PaymentTransaction.created_at))
        .limit(limit)
    )
    rows = res.scalars().all()
    return [
        PaymentOut(
            order_id=r.order_id,
            payment_id=r.payment_id,
            amount=r.amount,
            currency=r.currency,
            method=r.method.value,
            status=r.status.value,
            created_at=r.created_at.isoformat(),
            updated_at=r.updated_at.isoformat(),
        )
        for r in rows
    ]
