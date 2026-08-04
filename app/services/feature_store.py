"""
app/services/feature_store.py — Feature registry with Redis persistence.

In-memory dict is the read cache; every write is also committed to Redis.
Call restore_from_redis() at lifespan startup.
"""
import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Dict, List, Optional

from app.schemas.feature import Feature, FeatureCreate

logger = logging.getLogger(__name__)
_REDIS_PREFIX = "vit:ai:feature"


class FeatureStore:
    def __init__(self):
        self.features: Dict[str, Feature] = {}

    # ── Startup restore ───────────────────────────────────────────────────────

    async def restore_from_redis(self) -> int:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return 0
        try:
            ids = await r.smembers(f"{_REDIS_PREFIX}:index")
            count = 0
            for fid in ids:
                raw = await r.get(f"{_REDIS_PREFIX}:{fid}")
                if raw:
                    self.features[fid] = Feature(**json.loads(raw))
                    count += 1
            if count:
                logger.info("[feature_store] Restored %d features from Redis", count)
            return count
        except Exception as exc:
            logger.warning("[feature_store] Redis restore failed: %s", exc)
            return 0

    # ── Private Redis write ───────────────────────────────────────────────────

    async def _persist(self, feature: Feature) -> None:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return
        try:
            await r.set(f"{_REDIS_PREFIX}:{feature.id}", feature.model_dump_json())
            await r.sadd(f"{_REDIS_PREFIX}:index", feature.id)
        except Exception as exc:
            logger.warning("[feature_store] Redis write failed for %s: %s", feature.id, exc)

    async def _delete_from_redis(self, feature_id: str) -> None:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return
        try:
            await r.delete(f"{_REDIS_PREFIX}:{feature_id}")
            await r.srem(f"{_REDIS_PREFIX}:index", feature_id)
        except Exception as exc:
            logger.warning("[feature_store] Redis delete failed for %s: %s", feature_id, exc)

    def _fire(self, coro) -> None:
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, feature_in: FeatureCreate) -> Feature:
        feature = Feature(**feature_in.model_dump())
        self.features[feature.id] = feature
        self._fire(self._persist(feature))
        return feature

    def get_all(self) -> List[Feature]:
        return list(self.features.values())

    def get_by_id(self, feature_id: str) -> Optional[Feature]:
        return self.features.get(feature_id)

    def delete(self, feature_id: str) -> bool:
        if feature_id in self.features:
            del self.features[feature_id]
            self._fire(self._delete_from_redis(feature_id))
            return True
        return False


feature_store = FeatureStore()
