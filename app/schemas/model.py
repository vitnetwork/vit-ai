from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List

class ModelVersionBase(BaseModel):
    version: str
    status: str = "active"
    storage_id: Optional[str] = None  # Reference to vit-storage or local path
    metadata: Dict[str, Any] = {}

class ModelVersionCreate(ModelVersionBase):
    pass

class ModelVersion(ModelVersionBase):
    model_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(protected_namespaces=())

class ModelBase(BaseModel):
    id: str
    name: str
    description: str = ""
    # Optional fields — not required for internal VIT models loaded from disk
    model_type: Optional[str] = None
    capabilities: List[str] = []
    provider: str = "internal"
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}

class Model(ModelBase):
    versions: List[ModelVersion] = []
    active_version: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ModelCreate(ModelBase):
    initial_version: str = "0.1.0"
    storage_id: Optional[str] = None

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    active_version: Optional[str] = None
