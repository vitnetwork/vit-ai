"""
app/services/training.py — Training job management with Redis persistence.

In-memory dict is the read cache; every write is also committed to Redis so
jobs survive container restarts. Call restore_from_redis() at startup.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional

from app.schemas.training import TrainingJob, TrainingJobCreate

logger = logging.getLogger(__name__)
_REDIS_PREFIX = "vit:ai:training"


class TrainingManager:
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}

    # ── Startup restore ───────────────────────────────────────────────────────

    async def restore_from_redis(self) -> int:
        """Load all persisted jobs into memory. Called once at lifespan startup."""
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return 0
        try:
            job_ids = await r.smembers(f"{_REDIS_PREFIX}:index")
            count = 0
            for job_id in job_ids:
                raw = await r.get(f"{_REDIS_PREFIX}:{job_id}")
                if raw:
                    self.jobs[job_id] = TrainingJob(**json.loads(raw))
                    count += 1
            if count:
                logger.info("[training] Restored %d jobs from Redis", count)
            return count
        except Exception as exc:
            logger.warning("[training] Redis restore failed: %s", exc)
            return 0

    # ── Private Redis write ───────────────────────────────────────────────────

    async def _persist(self, job: TrainingJob) -> None:
        from app.core.redis_client import get_redis
        r = await get_redis()
        if not r:
            return
        try:
            await r.set(f"{_REDIS_PREFIX}:{job.id}", job.model_dump_json())
            await r.sadd(f"{_REDIS_PREFIX}:index", job.id)
        except Exception as exc:
            logger.warning("[training] Redis write failed for %s: %s", job.id, exc)

    def _fire_persist(self, job: TrainingJob) -> None:
        """Schedule a non-blocking Redis write from a sync context."""
        try:
            asyncio.get_running_loop().create_task(self._persist(job))
        except RuntimeError:
            pass  # No event loop (tests/CLI) — skip

    # ── Public API ────────────────────────────────────────────────────────────

    def create_job(self, job_in: TrainingJobCreate) -> TrainingJob:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = TrainingJob(id=job_id, **job_in.model_dump())
        self.jobs[job_id] = job
        self._fire_persist(job)
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        return list(self.jobs.values())

    def update_status(
        self, job_id: str, status: str, log: Optional[str] = None
    ) -> Optional[TrainingJob]:
        job = self.get_job(job_id)
        if not job:
            return None
        job.status = status
        if log:
            job.logs.append(log)
        job.updated_at = datetime.now(UTC)
        self._fire_persist(job)
        return job


training_manager = TrainingManager()
