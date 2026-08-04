import uuid
from typing import List, Dict, Optional
from app.schemas.training import TrainingJob, TrainingJobCreate
from datetime import datetime, UTC

class TrainingManager:
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}

    def create_job(self, job_in: TrainingJobCreate) -> TrainingJob:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = TrainingJob(id=job_id, **job_in.model_dump())
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        return list(self.jobs.values())

    def update_status(self, job_id: str, status: str, log: Optional[str] = None) -> Optional[TrainingJob]:
        job = self.get_job(job_id)
        if not job:
            return None
        job.status = status
        if log:
            job.logs.append(log)
        job.updated_at = datetime.now(UTC)
        return job

training_manager = TrainingManager()
