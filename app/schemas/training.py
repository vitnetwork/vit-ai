from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List

class TrainingJobBase(BaseModel):
    model_id: str
    dataset_id: str
    params: Dict[str, Any] = {}

    model_config = ConfigDict(protected_namespaces=())

class TrainingJob(TrainingJobBase):
    id: str
    status: str = "queued"  # queued, running, completed, failed
    logs: List[str] = []
    result_metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class TrainingJobCreate(TrainingJobBase):
    pass
