"""Legacy compatibility module for older pickled models that expected services.ml_service."""

from app.services.base_model import StandardizedModel
from app.services.registry import registry
from app.services.ensemble import ensemble_engine
from app.services.inference import inference_pipeline

__all__ = [
    "StandardizedModel",
    "registry",
    "ensemble_engine",
    "inference_pipeline",
]
