import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.core.config import settings
# Import the module-level singleton so lifespan and endpoints share the same instance
from app.services.registry import registry as model_registry
from fastapi import Response

try:
  from prometheus_client import CONTENT_TYPE_LATEST
except Exception:
  CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

logging.basicConfig(
  level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
  format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("vit-ai")


def get_registry():
  """Return the process-wide ModelRegistry singleton."""
  return model_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info("VIT AI Service v%s starting up…", settings.APP_VERSION)
  logger.info(
      "MODEL_DIR=%s  VIT_STORAGE_URL=%s  VIT_NETWORK_URL=%s",
      settings.MODEL_DIR,
      settings.VIT_STORAGE_URL,
      settings.VIT_NETWORK_URL,
  )

  # Bootstrap all 16 VIT ensemble models into the module-level registry singleton.
  # endpoints.py imports this same singleton so models are immediately visible.
  # Restore persisted state from Redis before bootstrapping models
  from app.services.training import training_manager
  from app.services.feature_store import feature_store
  from app.services.dataset_registry import dataset_registry
  jobs_restored     = await training_manager.restore_from_redis()
  features_restored = await feature_store.restore_from_redis()
  datasets_restored = await dataset_registry.restore_from_redis()
  logger.info(
      "Redis restore: %d jobs, %d features, %d datasets",
      jobs_restored, features_restored, datasets_restored,
  )

  loaded = model_registry.bootstrap_vit_models()
  app.state.models_loaded = loaded

  if loaded == 0:
      logger.warning(
          "⚠  DEGRADED: 0 models loaded. Inference endpoints will raise errors until "
          "MODEL_DIR is set correctly and .pkl files are bundled in the Docker image. "
          "Rebuild with: COPY models /app/models in Dockerfile."
      )
  else:
      logger.info("✓  %d models ready. VIT AI Service is OPERATIONAL.", loaded)

  yield

  from app.core.redis_client import close_redis
  await close_redis()
  logger.info("VIT AI Service shutting down.")


app = FastAPI(
  title="VIT AI Service",
  description="13-model ensemble powering the VIT Intelligence Oracle",
  version=settings.APP_VERSION,
  lifespan=lifespan,
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/metrics")
async def metrics():
  """Prometheus metrics endpoint exporting registry and provider metrics."""
  try:
    from app.metrics.collector import collect_metrics
    data = collect_metrics()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
  except Exception as exc:
    return Response(content=f"# metrics error: {exc}\n", media_type="text/plain")


@app.get("/ping")
async def ping():
  return {"status": "ok", "service": "vit-ai"}


@app.get("/health")
async def health():
  # Detect whether lifespan startup has completed (models_loaded is set during lifespan)
  startup_complete = hasattr(app.state, "models_loaded")
  if not startup_complete:
      from fastapi.responses import JSONResponse as _JSONResponse
      return _JSONResponse(
          status_code=503,
          content={
              "status":      "warming",
              "retry_after": 15,
              "version":     settings.APP_VERSION,
          },
      )
  loaded = app.state.models_loaded
  status = "healthy" if loaded > 0 else "degraded"
  return {
      "status":        status,
      "version":       settings.APP_VERSION,
      "models_loaded": loaded,
  }


@app.get("/version")
async def version():
  return {"version": settings.APP_VERSION}
