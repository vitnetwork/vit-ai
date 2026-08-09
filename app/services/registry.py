from typing import List, Dict, Optional, Any
from app.schemas.model import Model, ModelCreate, ModelUpdate, ModelVersion, ModelVersionCreate
from app.services.base_model import StandardizedModel
from datetime import datetime, UTC
import logging

logger = logging.getLogger(__name__)

VIT_CORE_MODELS = [
  {"id": "xgb_v1",          "name": "XGBoost Ensemble",      "version": "1.0", "type": "classification"},
  {"id": "lstm_v1",          "name": "LSTM Sequential",        "version": "1.0", "type": "sequential"},
  {"id": "transformer_v1",   "name": "Transformer Attention",  "version": "1.0", "type": "transformer"},
  {"id": "rf_v1",            "name": "Random Forest",          "version": "1.0", "type": "classification"},
  {"id": "gbm_v1",           "name": "Gradient Boosting",      "version": "1.0", "type": "classification"},
  {"id": "bayes_v1",         "name": "Bayesian Network",       "version": "1.0", "type": "probabilistic"},
  {"id": "logistic_v1",      "name": "Logistic Regression",    "version": "1.0", "type": "classification"},
  {"id": "elo_v1",           "name": "ELO Rating System",      "version": "1.0", "type": "rating"},
  {"id": "poisson_v1",       "name": "Poisson Regressor",      "version": "1.0", "type": "regression"},
  {"id": "dixon_coles_v1",   "name": "Dixon-Coles",            "version": "1.0", "type": "probabilistic"},
  {"id": "hybrid_v1",        "name": "Hybrid Ensemble",        "version": "1.0", "type": "ensemble"},
  {"id": "market_v1",        "name": "Market Odds Model",      "version": "1.0", "type": "market"},
  {"id": "ensemble_v1",      "name": "Master Ensemble",        "version": "1.0", "type": "ensemble"},
  {"id": "btts_v2",          "name": "BTTS Model v2",          "version": "2.0", "type": "classification"},
  {"id": "over_under_v2",    "name": "Over/Under v2",          "version": "2.0", "type": "classification"},
  {"id": "correct_score_v2", "name": "Correct Score v2",       "version": "2.0", "type": "classification"},
]


class ModelRegistry:
  def __init__(self):
      self.models: Dict[str, Model] = {}
      self.loaded_artifacts: Dict[str, StandardizedModel] = {}

  def bootstrap_vit_models(self) -> int:
      loaded = 0
      for spec in VIT_CORE_MODELS:
          try:
              model_in = ModelCreate(
                  id=spec["id"], name=spec["name"],
                  description=f"VIT core model — {spec['type']}",
                  model_type=spec["type"], initial_version=spec["version"], storage_id=None,
              )
              self.register(model_in)
              artifact = self.get_artifact(spec["id"], spec["version"])
              if artifact and artifact.is_loaded:
                  loaded += 1
          except Exception as exc:
              logger.warning("Bootstrap failed for %s: %s", spec["id"], exc)
      if loaded == 0:
          logger.error(
              "STARTUP ASSERTION FAILED: 0/%d models loaded. "
              "Verify MODEL_DIR and that .pkl files are bundled in the Docker image.",
              len(VIT_CORE_MODELS),
          )
      else:
          logger.info("STARTUP: %d/%d VIT models loaded.", loaded, len(VIT_CORE_MODELS))
      return loaded

  def register(self, model_in: ModelCreate) -> Model:
      version = ModelVersion(model_id=model_in.id, version=model_in.initial_version,
                             storage_id=model_in.storage_id, status="active")
      model = Model(**model_in.model_dump(exclude={"initial_version", "storage_id"}),
                    versions=[version], active_version=model_in.initial_version)
      self.models[model.id] = model
      storage_id = model_in.storage_id or f"local://models/{model.id}"
      self.load_model_artifact(model.id, version.version, storage_id)
      return model

  def load_model_artifact(self, model_id: str, version: str, storage_id: str):
      artifact = StandardizedModel(model_id=model_id, model_version=version, storage_id=storage_id)
      artifact.load()
      self.loaded_artifacts[f"{model_id}:{version}"] = artifact

  def get_artifact(self, model_id: str, version: str) -> Optional[StandardizedModel]:
      return self.loaded_artifacts.get(f"{model_id}:{version}")

  def get_all(self) -> List[Model]:
      return list(self.models.values())

  def get_by_id(self, model_id: str) -> Optional[Model]:
      return self.models.get(model_id)

  def loaded_model_count(self) -> int:
      return sum(1 for a in self.loaded_artifacts.values() if a.is_loaded)

  def inference_ready_count(self) -> int:
      return sum(1 for a in self.loaded_artifacts.values() if a.inference_ready)

  def failed_model_count(self) -> int:
      return sum(1 for a in self.loaded_artifacts.values() if a.is_loaded and not a.inference_ready)

  def inference_summary(self) -> Dict[str, Any]:
      total = 0
      failed = 0
      last_success = None
      for artifact in self.loaded_artifacts.values():
          total += artifact.inference_count
          failed += artifact.inference_failures
          ts = getattr(artifact, "last_successful_inference_at", None)
          if ts:
              last_success = max(last_success, ts) if last_success else ts
      return {
          "total_inference_count": total,
          "failed_inference_count": failed,
          "last_successful_inference": last_success,
      }

  def get_diagnostics(self) -> List[Dict[str, Any]]:
      diagnostics = []
      for model_id, model in self.models.items():
          version_obj = None
          if model.active_version:
              version_obj = next((v for v in model.versions if v.version == model.active_version), None)
          if version_obj is None and model.versions:
              version_obj = model.versions[0]

          artifact = self.get_artifact(model_id, version_obj.version) if version_obj else None
          configured = bool(version_obj and version_obj.storage_id)
          diagnostics.append({
              "model_id": model.id,
              "name": model.name,
              "version": version_obj.version if version_obj else None,
              "framework": model.model_type,
              "provider": model.provider,
              "registered": True,
              "configured": configured,
              "artifact_available": bool(artifact and artifact._artifact is not None),
              "artifact_source": artifact.metadata().get("load_source") if artifact else None,
              "loaded": bool(artifact and artifact.is_loaded),
              "inference_ready": bool(artifact and artifact.inference_ready),
              "health": artifact.metadata().get("health") if artifact else "unknown",
              "failed": bool(artifact and (artifact._metadata.get("load_error") or (artifact.is_loaded and not artifact.inference_ready))),
              "load_error": artifact.metadata().get("load_error") if artifact else None,
              "attempted_paths": artifact.metadata().get("attempted_paths") if artifact else None,
              "load_time": artifact.metadata().get("load_time") if artifact else None,
              "last_loaded_at": artifact.metadata().get("last_loaded_at") if artifact else None,
              "last_inference_at": artifact.metadata().get("last_inference_at") if artifact else None,
              "inference_count": artifact.inference_count if artifact else 0,
              "inference_failures": artifact.inference_failures if artifact else 0,
          })
      return diagnostics

  def add_version(self, model_id: str, version_in: ModelVersionCreate) -> Optional[ModelVersion]:
      model = self.get_by_id(model_id)
      if not model:
          return None
      version = ModelVersion(model_id=model_id, **version_in.model_dump())
      model.versions.append(version)
      model.updated_at = datetime.now(UTC)
      self.load_model_artifact(model_id, version.version,
                               version.storage_id or f"local://models/{model_id}-{version.version}")
      return version

  def update(self, model_id: str, model_in: ModelUpdate) -> Optional[Model]:
      if model_id not in self.models:
          return None
      model = self.models[model_id]
      for field, value in model_in.model_dump(exclude_unset=True).items():
          setattr(model, field, value)
      model.updated_at = datetime.now(UTC)
      return model

  def delete(self, model_id: str) -> bool:
      if model_id in self.models:
          del self.models[model_id]
          for k in [k for k in self.loaded_artifacts if k.startswith(f"{model_id}:")]:
              del self.loaded_artifacts[k]
          return True
      return False

# Module-level singleton — imported as 'from app.services.registry import registry'
registry = ModelRegistry()
