"""
app/core/redis_client.py — Shared async Redis client for vit-ai persistence.

Provides a module-level singleton. Falls back gracefully when REDIS_URL is
absent or unreachable so every caller can treat None as "no persistence".
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


async def get_redis():
    """Return the shared async Redis client, or None if unavailable."""
    global _client
    if _client is not None:
        return _client

    url = os.getenv("REDIS_URL", "")
    if not url:
        logger.warning("[redis] REDIS_URL not set — state will not persist across restarts")
        return None

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=3)
        await r.ping()
        _client = r
        logger.info("[redis] Connected — vit-ai persistence layer active")
    except Exception as exc:
        logger.warning("[redis] Unavailable (%s) — in-memory only mode", exc)
        _client = None

    return _client


async def close_redis():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
