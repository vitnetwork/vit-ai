"""
app/services/dataset_registry.py — Dataset registry with Redis persistence.

In-memory dict is the read cache; every write is also committed to Redis.
Call restore_from_redis() at lifespan startup.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional

from app.schemas.dataset import Dataset, DatasetCreate

logger = logging.getLogger(__name__)
_REDIS_PREFIX = "vit:ai:dataset"


class DatasetRegistry:
    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}

    # ── Startup restore ───────────────────────────────────────────────────────

    async def restore_from_redis(self) -> int:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return 0
        try:
            ids = await r.smembers(f"{_REDIS_PREFIX}:index")
            count = 0
            for did in ids:
                raw = await r.get(f"{_REDIS_PREFIX}:{did}")
                if raw:
                    self.datasets[did] = Dataset(**json.loads(raw))
                    count += 1
            if count:
                logger.info("[dataset_registry] Restored %d datasets from Redis", count)
            return count
        except Exception as exc:
            logger.warning("[dataset_registry] Redis restore failed: %s", exc)
            return 0

    # ── Private Redis write ───────────────────────────────────────────────────

    async def _persist(self, dataset: Dataset) -> None:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return
        try:
            await r.set(f"{_REDIS_PREFIX}:{dataset.id}", dataset.model_dump_json())
            await r.sadd(f"{_REDIS_PREFIX}:index", dataset.id)
        except Exception as exc:
            logger.warning("[dataset_registry] Redis write failed for %s: %s", dataset.id, exc)

    async def _delete_from_redis(self, dataset_id: str) -> None:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return
        try:
            await r.delete(f"{_REDIS_PREFIX}:{dataset_id}")
            await r.srem(f"{_REDIS_PREFIX}:index", dataset_id)
        except Exception as exc:
            logger.warning("[dataset_registry] Redis delete failed for %s: %s", dataset_id, exc)

    def _fire(self, coro) -> None:
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, dataset_in: DatasetCreate) -> Dataset:
        dataset = Dataset(**dataset_in.model_dump())
        self.datasets[dataset.id] = dataset
        self._fire(self._persist(dataset))
        return dataset

    def get_all(self) -> List[Dataset]:
        return list(self.datasets.values())

    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        return self.datasets.get(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            self._fire(self._delete_from_redis(dataset_id))
            return True
        return False


dataset_registry = DatasetRegistry()
