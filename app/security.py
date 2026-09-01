"""Keamanan server-to-server: verifikasi shared secret header.

Laravel (Main API, sudah lewat auth Sanctum) mem-forward request internal ke
FastAPI sembari mengirim header `X-AI-Secret`. FastAPI memvalidasi dengan
`hmac.compare_digest` (konstan-waktu) agar endpoint hanya menerima panggilan
internal — bukan dari publik internet.
"""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import settings

HEADER_NAME = "x-ai-secret"


def is_valid_secret(value: str) -> bool:
    expected = settings.shared_secret
    if not expected:
        # Tanpa shared secret terpasang, tolak agar aman secara default.
        return False
    return hmac.compare_digest(str(value), expected)


def require_secret(x_ai_secret: str = Header(default="", alias=HEADER_NAME)) -> None:
    if not is_valid_secret(x_ai_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: invalid or missing AI shared secret",
        )
