"""Compatibility shim for legacy pickles that import the top-level services package."""

from app.services.base_model import StandardizedModel
from app.services.ensemble import ensemble_engine
from app.services.inference import inference_pipeline
from app.services.registry import registry
from app.services.providers import InternalProvider, AdHocProvider, EnsembleProvider

__all__ = [
    "StandardizedModel",
    "ensemble_engine",
    "inference_pipeline",
    "registry",
    "InternalProvider",
    "AdHocProvider",
    "EnsembleProvider",
]
