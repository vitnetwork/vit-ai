from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, UTC
from typing import Optional, Dict, Any

class DatasetBase(BaseModel):
    id: str
    name: str
    version: str
    description: str
    checksum: str
    metadata: Optional[Dict[str, Any]] = None

class DatasetCreate(DatasetBase):
    pass

class Dataset(DatasetBase):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
