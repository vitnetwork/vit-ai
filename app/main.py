import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.core.config import settings
# Import the module-level singleton so lifespan and endpoints share the same instance
from app.services.registry import registry as model_registry

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
  logger.info("MODEL_DIR=%s  VIT_STORAGE_URL=%s", settings.MODEL_DIR, settings.VIT_STORAGE_URL)

  # Bootstrap all 16 VIT ensemble models into the module-level registry singleton.
  # endpoints.py imports this same singleton so models are immediately visible.
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


@app.get("/ping")
async def ping():
  return {"status": "ok", "service": "vit-ai"}


@app.get("/health")
async def health():
  loaded = getattr(app.state, "models_loaded", 0)
  status = "healthy" if loaded > 0 else "degraded"
  return {
      "status": status,
      "version": settings.APP_VERSION,
      "models_loaded": loaded,
  }


@app.get("/version")
async def version():
  return {"version": settings.APP_VERSION}
