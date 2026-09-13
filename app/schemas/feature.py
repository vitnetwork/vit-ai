from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List

class FeatureBase(BaseModel):
    id: str
    name: str
    type: str  # numeric, categorical, text, etc.
    description: str
    metadata: Optional[Dict[str, Any]] = None

class FeatureCreate(FeatureBase):
    pass

class Feature(FeatureBase):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
