import logging
import os
import time
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class AIKernel:
    def __init__(self):
        self.status = "operational"
        self.providers = settings.SUPPORTED_PROVIDERS
        self.start_time = time.time()

    def get_status(self) -> Dict[str, Any]:
        # Import here to avoid circular import; use the shared singleton registry
        from app.services.registry import registry
        loaded = registry.loaded_model_count()
        diagnostics = registry.get_diagnostics()
        summary = registry.inference_summary()
        return {
            "status": self.status,
            "version": settings.VERSION,
            "models_registered": len(diagnostics),
            "models_loaded": loaded,
            "models_inference_ready": registry.inference_ready_count(),
            "models_failed": registry.failed_model_count(),
            "storage_status": "configured" if settings.VIT_STORAGE_URL else "disabled",
            "database_status": "configured" if os.getenv("REDIS_URL") else "disabled",
            "uptime": round(time.time() - self.start_time, 3),
            "last_successful_inference": summary.get("last_successful_inference"),
            "total_inference_count": summary.get("total_inference_count", 0),
            "failed_inference_count": summary.get("failed_inference_count", 0),
        }

    def get_providers(self) -> List[str]:
        return self.providers

    def get_diagnostics(self) -> Dict[str, Any]:
        from app.services.registry import registry
        status = self.get_status()
        diagnostics = registry.get_diagnostics()
        summary = registry.inference_summary()
        # Provider metrics (synchronous snapshot of counters maintained by providers)
        provider_metrics = {}
        try:
            from app.services.inference import inference_pipeline
            for name, prov in getattr(inference_pipeline, "providers", {}).items():
                provider_metrics[name] = {
                    "provider_class": prov.__class__.__name__,
                    "initialized": bool(getattr(prov, "_is_initialized", False)),
                    "requests_processed": int(getattr(prov, "_request_count", 0)),
                }
        except Exception:
            provider_metrics = {}

        # Storage reachability quick-check (best-effort, short timeout)
        storage_reachable = None
        storage_url = settings.VIT_STORAGE_URL
        if storage_url:
            try:
                import httpx
                with httpx.Client(timeout=1.0) as client:
                    r = client.get(storage_url)
                    storage_reachable = r.status_code == 200
            except Exception:
                storage_reachable = False

        return {
            "summary": status,
            "models": diagnostics,
            "components": {
                "model_registry": {
                    "status": "healthy" if diagnostics else "degraded",
                    "registered": len(diagnostics),
                    "loaded": registry.loaded_model_count(),
                    "inference_ready": registry.inference_ready_count(),
                    "failed": registry.failed_model_count(),
                },
                "storage": {
                    "status": "configured" if storage_url else "disabled",
                    "url": storage_url,
                    "reachable": storage_reachable,
                },
                "database": {
                    "status": "configured" if os.getenv("REDIS_URL") else "disabled"
                },
                "pipeline": {
                    "total_inference_count": summary.get("total_inference_count", 0),
                    "failed_inference_count": summary.get("failed_inference_count", 0),
                    "providers": provider_metrics,
                }
            }
        }

kernel = AIKernel()
