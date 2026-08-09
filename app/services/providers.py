import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from app.utils.math import normalise, vig_free, market_to_xg
from app.services.registry import registry

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider and its resources."""
        pass

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with external APIs or local security keys."""
        pass

    @abstractmethod
    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute inference on the specified model."""
        pass

    @abstractmethod
    async def embeddings(self, text: str, model_id: str) -> Dict[str, Any]:
        """Generate vector representations for the input text."""
        pass

    @abstractmethod
    async def stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream inference chunks from the provider."""
        yield {}

    @abstractmethod
    async def shutdown(self) -> bool:
        """Gracefully shut down and release resources."""
        pass

    @abstractmethod
    async def metrics(self) -> Dict[str, Any]:
        """Retrieve performance and usage metrics."""
        pass

    # Backwards compatibility helper
    async def predict(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility mapping to standard infer function."""
        return await self.infer(model_id, payload)


class InternalProvider(ModelProvider):
    def __init__(self):
        self._is_initialized = False
        self._request_count = 0

    async def initialize(self) -> bool:
        self._is_initialized = True
        logger.info("InternalProvider initialized.")
        return True

    async def authenticate(self) -> bool:
        return True

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._request_count += 1
        try:
            from app.metrics.updater import record_provider_request
            record_provider_request("internal")
        except Exception:
            pass
        model = registry.get_by_id(model_id)
        if not model or not model.active_version:
            return {"status": "error", "message": "Model not found or inactive"}

        artifact = registry.get_artifact(model_id, model.active_version)
        if not artifact:
            logger.error(
                "No artifact loaded for %s v%s. "
                "Ensure .pkl files are bundled in the Docker image under MODEL_DIR.",
                model_id, model.active_version,
            )
            return {
                "status": "error",
                "message": (
                    f"Model '{model_id}' artifact not loaded. "
                    "The service is operating in DEGRADED mode — rebuild the Docker image "
                    "with MODEL_DIR populated."
                ),
                "model_id": model_id,
            }

        try:
            result = artifact.predict(payload)
            if isinstance(result, dict):
                result.setdefault("provider", "internal")
            return result
        except Exception as exc:
            logger.error("InternalProvider inference failed for %s: %s", model_id, exc)
            return {
                "status": "error",
                "message": str(exc),
                "provider": "internal",
                "model_id": model_id,
            }

    async def embeddings(self, text: str, model_id: str) -> Dict[str, Any]:
        # Delegate to the deterministic hash embedding service
        from app.services.embedding import embedding_service
        return await embedding_service.generate(text, model_id)

    async def stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        result = await self.infer(model_id, payload)
        yield result

    async def shutdown(self) -> bool:
        self._is_initialized = False
        logger.info("InternalProvider shut down.")
        return True

    async def metrics(self) -> Dict[str, Any]:
        return {
            "is_initialized": self._is_initialized,
            "requests_processed": self._request_count,
            "provider_type": "internal"
        }


class EnsembleProvider(ModelProvider):
    def __init__(self, ensemble_engine):
        self.ensemble_engine = ensemble_engine
        self._is_initialized = False
        self._request_count = 0

    async def initialize(self) -> bool:
        self._is_initialized = True
        logger.info("EnsembleProvider initialized.")
        return True

    async def authenticate(self) -> bool:
        return True

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._request_count += 1
        try:
            from app.metrics.updater import record_provider_request
            record_provider_request("ensemble")
        except Exception:
            pass
        return await self.ensemble_engine.orchestrate(payload)

    async def embeddings(self, text: str, model_id: str) -> Dict[str, Any]:
        from app.services.embedding import embedding_service
        return await embedding_service.generate(text, model_id)

    async def stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        result = await self.infer(model_id, payload)
        yield result

    async def shutdown(self) -> bool:
        self._is_initialized = False
        logger.info("EnsembleProvider shut down.")
        return True

    async def metrics(self) -> Dict[str, Any]:
        return {
            "is_initialized": self._is_initialized,
            "requests_processed": self._request_count,
            "provider_type": "ensemble"
        }


class AdHocProvider(ModelProvider):
    """
    Fallback provider used when a requested model_id is not found in the registry.

    Routes to the EnsembleEngine when market_odds are present in the payload,
    so sports-prediction requests always get a meaningful result rather than a
    hardcoded 0.0 stub.  For non-sports payloads (chat, classify, summarize,
    text embeddings) it returns a structured response that clearly identifies
    the ad-hoc path.
    """

    def __init__(self):
        self._is_initialized = False
        self._request_count = 0

    async def initialize(self) -> bool:
        self._is_initialized = True
        return True

    async def authenticate(self) -> bool:
        return True

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._request_count += 1
        try:
            from app.metrics.updater import record_provider_request
            record_provider_request("adhoc")
        except Exception:
            pass

        # If market odds are present, route to the ensemble engine which can produce
        # a real probabilistic prediction from the implied probabilities alone.
        if payload.get("market_odds"):
            try:
                from app.services.ensemble import ensemble_engine
                result = await ensemble_engine.orchestrate(payload)
                result["provider"] = "adhoc_ensemble_fallback"
                result["fallback_reason"] = f"model_id '{model_id}' not in registry; routed to ensemble"
                return result
            except Exception as exc:
                logger.warning("AdHocProvider ensemble fallback failed: %s", exc)

        # For text/NLP requests (prompt present), return a structured response
        # that at minimum echoes the prompt back so callers know the request was received.
        if payload.get("prompt"):
            prompt_text = str(payload["prompt"])[:200]
            return {
                "status": "success",
                "provider": "adhoc",
                "model_id": model_id,
                "result": f"[ad-hoc] Received: {prompt_text}",
                "note": (
                    f"Model '{model_id}' is not registered. "
                    "Register it via POST /api/v1/models or use 'ensemble_v1' for sports predictions."
                ),
            }

        # Generic fallback — return a low-confidence prediction
        return {
            "status": "success",
            "prediction": 0.5,
            "confidence": 0.0,
            "provider": "adhoc",
            "model_id": model_id,
            "note": f"Model '{model_id}' not found. Using neutral prediction.",
        }

    async def embeddings(self, text: str, model_id: str) -> Dict[str, Any]:
        from app.services.embedding import embedding_service
        return await embedding_service.generate(text, model_id)

    async def stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        result = await self.infer(model_id, payload)
        yield result

    async def shutdown(self) -> bool:
        self._is_initialized = False
        return True

    async def metrics(self) -> Dict[str, Any]:
        return {
            "is_initialized": self._is_initialized,
            "requests_processed": self._request_count,
            "provider_type": "adhoc"
        }
