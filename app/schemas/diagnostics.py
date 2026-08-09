from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class ModelDiagnostics(BaseModel):
    model_id: str
    name: str
    version: Optional[str] = None
    framework: Optional[str] = None
    provider: Optional[str] = None
    registered: bool = False
    configured: bool = False
    artifact_available: bool = False
    artifact_source: Optional[str] = None
    loaded: bool = False
    inference_ready: bool = False
    health: str = "unknown"
    failed: bool = False
    load_error: Optional[str] = None
    attempted_paths: Optional[List[str]] = None
    load_time: Optional[float] = None
    last_loaded_at: Optional[str] = None
    last_inference_at: Optional[str] = None
    inference_count: int = 0
    inference_failures: int = 0

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class AIStatusSummary(BaseModel):
    status: str
    version: str
    models_registered: int
    models_loaded: int
    models_inference_ready: int
    models_failed: int
    storage_status: str
    database_status: str
    uptime: Optional[float] = None
    last_successful_inference: Optional[str] = None
    total_inference_count: int = 0
    failed_inference_count: int = 0

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class AIDiagnostics(BaseModel):
    summary: AIStatusSummary
    models: List[ModelDiagnostics]
    components: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
