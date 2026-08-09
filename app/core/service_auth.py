"""
app/core/service_auth.py — VIT inter-service HMAC authentication.

Ported from vitnetwork/app/core/service_auth.py so vit-ai can validate
incoming service tokens issued by vitnetwork (and any other VIT service).

Token format:  "<service_name>.<minute_bucket>.<hmac_sha256_hex>"
TTL:           2 minutes (accepts current + previous minute bucket)
Secret:        SERVICE_TOKEN_SECRET env var (32+ byte random string)

Usage (validating incoming calls in vit-ai):
    from app.core.service_auth import verify_service_token
    # Called automatically via security.py verify_auth dependency

Usage (generating tokens — for future vit-ai → other-service calls):
    from app.core.service_auth import make_service_headers
    headers = make_service_headers("vit-ai")
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SECRET: Optional[str] = None


def _get_secret() -> str:
    global _SECRET
    if _SECRET is None:
        _SECRET = os.getenv("SERVICE_TOKEN_SECRET", "")
        if not _SECRET:
            logger.warning(
                "[service_auth] SERVICE_TOKEN_SECRET not set — "
                "HMAC token validation disabled for generation; verification will fail unless configured"
            )
    return _SECRET


def _sign(service_name: str, minute_bucket: int) -> str:
    secret = _get_secret()
    if not secret:
        return "dev-no-secret"
    payload = f"{service_name}:{minute_bucket}".encode()
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def generate_service_token(service_name: str = "vit-ai") -> str:
    """Generate a token valid for ~2 minutes."""
    bucket = int(time.time()) // 60
    return f"{service_name}.{bucket}.{_sign(service_name, bucket)}"


def verify_service_token(token: str) -> bool:
    """Validate a service token — accepts current and previous minute bucket."""
    secret = _get_secret()
    if not secret:
        # No secret configured — fail closed for verification to avoid accepting arbitrary tokens
        logger.warning("[service_auth] SERVICE_TOKEN_SECRET not set — refusing to validate service tokens")
        return False

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False
        service_name, bucket_str, sig = parts
        bucket = int(bucket_str)
    except (ValueError, AttributeError):
        return False

    now_bucket = int(time.time()) // 60
    for valid_bucket in (now_bucket, now_bucket - 1):
        if bucket == valid_bucket and hmac.compare_digest(
            sig, _sign(service_name, valid_bucket)
        ):
            return True
    return False


def make_service_headers(service_name: str = "vit-ai") -> dict[str, str]:
    """Return headers to include in outgoing internal service calls."""
    return {
        "X-VIT-Service-Token": generate_service_token(service_name),
        "X-VIT-Source-Service": service_name,
    }
