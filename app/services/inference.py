import time
import uuid
import logging
from typing import Dict, Any, Optional
from app.schemas.inference import InferenceRequest, InferenceResponse, EnsemblePrediction
from app.services.registry import registry
from app.services.ensemble import ensemble_engine
from app.services.providers import InternalProvider, EnsembleProvider, AdHocProvider

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self):
        self.providers = {
            "internal": InternalProvider(),
            "ensemble": EnsembleProvider(ensemble_engine),
            "adhoc": AdHocProvider()
        }

    async def process(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()

        # 1. Resolve Model and Provider
        model = registry.get_by_id(request.model_id)
        provider_type = "adhoc"
        if model:
            provider_type = model.provider
            provider_type = {
                "vit-internal": "internal",
                "internal": "internal",
                "vit-ensemble": "ensemble",
                "ensemble": "ensemble",
                "vit-adhoc": "adhoc",
                "adhoc": "adhoc",
            }.get(provider_type, provider_type)
        else:
            logger.warning(f"Model {request.model_id} not found in registry. Using ad-hoc provider.")

        provider = self.providers.get(provider_type, self.providers["adhoc"])

        # 2. Dispatch Inference
        prediction_details = None
        result_data = await provider.predict(request.model_id, request.payload)

        # 3. Post-process based on result type
        if "prediction" in result_data and isinstance(result_data["prediction"], dict):
            try:
                prediction_details = EnsemblePrediction(**result_data["prediction"])
                result = "Ensemble prediction completed"
            except Exception:
                result = result_data
        else:
            result = result_data

        latency = time.time() - start_time
        # Record metrics for this inference
        try:
            from app.metrics.updater import record_inference
            success = not (isinstance(result, dict) and result.get("status") == "error")
            record_inference(request.model_id, latency, success=success)
        except Exception:
            pass

        return InferenceResponse(
            request_id=str(uuid.uuid4()),
            model_id=request.model_id,
            result=result,
            latency=latency,
            metadata={
                "provider": provider_type,
                "version": getattr(model, "active_version", "unknown"),
                "engine_status": ensemble_engine.status
            },
            prediction_details=prediction_details
        )

inference_pipeline = InferencePipeline()
