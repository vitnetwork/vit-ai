import io
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseModelInterface(ABC):
  @abstractmethod
  def load(self) -> bool:
      pass

  @abstractmethod
  def unload(self) -> bool:
      pass

  @abstractmethod
  def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      pass

  @abstractmethod
  def batch_predict(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
      pass

  @abstractmethod
  def explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      pass

  @abstractmethod
  def validate(self, payload: Dict[str, Any]) -> bool:
      pass

  @abstractmethod
  def version(self) -> str:
      pass

  @abstractmethod
  def metadata(self) -> Dict[str, Any]:
      pass

  @abstractmethod
  def health_check(self) -> bool:
      pass


class StandardizedModel(BaseModelInterface):
  """
  VIT standardized model wrapper.

  Load priority:
    1. Local .pkl file resolved from MODEL_DIR env var
    2. Reports is_loaded=False and raises on predict() if no artifact found
  """

  def __init__(self, model_id, model_version, storage_id=None, metadata_dict=None):
      self.model_id = model_id
      self.model_version = model_version
      self.storage_id = storage_id
      self._metadata = metadata_dict or {}
      self.is_loaded = False
      self.inference_ready = False
      self.last_loaded_at: Optional[str] = None
      self.last_inference_at: Optional[str] = None
      self.last_successful_inference_at: Optional[str] = None
      self.inference_count = 0
      self.inference_failures = 0
      self.load_latency = None
      self._artifact = None
      self._metadata.setdefault("load_source", None)
      self._metadata.setdefault("load_error", None)
      self._metadata.setdefault("attempted_paths", None)
      self._metadata.setdefault("load_time", None)

  def _remote_artifact_bytes(self, storage_id: str) -> Optional[bytes]:
      try:
          from app.core.config import settings
          from app.services.storage_client import storage_client
      except ImportError:
          return None

      if not getattr(settings, "VIT_STORAGE_URL", None):
          return None

      try:
          return storage_client.download_sync(storage_id)
      except Exception as exc:
          logger.warning("Remote storage download failed for %s: %s", storage_id, exc)
          self._metadata["load_error"] = str(exc)
          return None

  def _supports_inference(self, artifact: Any) -> bool:
      if artifact is None:
          return False
      model, _ = self._resolve_artifact(artifact)
      return model is not None and (hasattr(model, "predict_proba") or hasattr(model, "predict"))

  def load(self) -> bool:
      model_dir = os.getenv("MODEL_DIR", "/app/models")
      candidates = []

      if self.storage_id:
          storage = str(self.storage_id)
          if storage.startswith("local://") or storage.startswith("file://"):
              path = storage.split("://", 1)[1]
              if not os.path.isabs(path):
                  path = os.path.join(model_dir, path)
              candidates.append(path)
          elif os.path.isabs(storage) or os.path.exists(storage):
              candidates.append(storage)
          else:
              # Treat relative storage_id paths as relative to MODEL_DIR
              candidates.append(os.path.join(model_dir, storage))

      candidates.extend([
          os.path.join(model_dir, f"{self.model_id}.pkl"),
          os.path.join(model_dir, f"{self.model_id}_v1.pkl"),
          os.path.join(model_dir, f"{self.model_id}_v2.pkl"),
      ])

      attempted_paths = []
      for path in candidates:
          attempted_paths.append(path)
          if os.path.exists(path):
              try:
                  import joblib
                  start = time.time()
                  self._artifact = joblib.load(path)
                  self.is_loaded = True
                  self._metadata["load_source"] = path
                  self._metadata["load_time"] = round(time.time() - start, 4)
                  self.load_latency = self._metadata["load_time"]
                  self.last_loaded_at = datetime.now(timezone.utc).isoformat()
                  self.inference_ready = self._supports_inference(self._artifact)
                  self._metadata["load_error"] = None
                  logger.info("Loaded artifact %s from %s", self.model_id, path)
                  return True
              except Exception as exc:
                  logger.warning("joblib load failed for %s: %s", self.model_id, exc)
                  self._metadata["load_error"] = str(exc)

      if self.storage_id:
          remote_data = self._remote_artifact_bytes(self.storage_id)
          if remote_data:
              try:
                  import joblib
                  start = time.time()
                  self._artifact = joblib.load(io.BytesIO(remote_data))
                  self.is_loaded = True
                  self._metadata["load_source"] = f"remote://{self.storage_id}"
                  self._metadata["load_time"] = round(time.time() - start, 4)
                  self.load_latency = self._metadata["load_time"]
                  self.last_loaded_at = datetime.now(timezone.utc).isoformat()
                  self.inference_ready = self._supports_inference(self._artifact)
                  self._metadata["load_error"] = None
                  logger.info("Loaded artifact %s from remote storage %s", self.model_id, self.storage_id)
                  return True
              except Exception as exc:
                  logger.warning("Remote joblib load failed for %s: %s", self.model_id, exc)
                  self._metadata["load_error"] = str(exc)

      if not self.is_loaded:
          self._metadata["attempted_paths"] = attempted_paths
          self._metadata["load_time"] = None
          self._metadata["load_source"] = None
          self._metadata["inference_ready"] = False
          self.load_latency = None
          logger.warning(
              "No .pkl artifact loaded for %s in %s — not loaded. "
              "Set MODEL_DIR, provide a local storage_id, or ensure the artifact "
              "is bundled in the Docker image.",
              self.model_id, model_dir,
          )
      return self.is_loaded

  def unload(self) -> bool:
      self._artifact = None
      self.is_loaded = False
      return True

  def _resolve_artifact(self, artifact=None):
      artifact = self._artifact if artifact is None else artifact
      if isinstance(artifact, dict) and "model" in artifact:
          return artifact["model"], artifact.get("scaler")
      return artifact, None

  def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
      if not self.is_loaded or self._artifact is None:
          raise RuntimeError(
              f"Model '{self.model_id}' not loaded. "
              "Check MODEL_DIR and ensure .pkl files are bundled in the Docker image."
          )
      try:
          import numpy as np
          model, scaler = self._resolve_artifact()
          features = payload.get("features", [])
          if not isinstance(features, list) or not features:
              raise ValueError("Payload must include a non-empty 'features' list")

          X = np.array(features).reshape(1, -1)
          if scaler is not None and hasattr(scaler, "transform"):
              X = scaler.transform(X)

          if hasattr(model, "predict_proba"):
              proba = model.predict_proba(X)[0]
              if len(proba) < 2:
                  raise ValueError("Model predict_proba returned invalid probability vector")
              self.inference_count += 1
              self.last_inference_at = datetime.now(timezone.utc).isoformat()
              self.last_successful_inference_at = datetime.now(timezone.utc).isoformat()
              self.inference_ready = True
              return {
                  "status": "success",
                  "model_id": self.model_id,
                  "version": self.model_version,
                  "prediction": float(proba.max()),
                  "probabilities": proba.tolist()
              }
          elif hasattr(model, "predict"):
              pred = model.predict(X)[0]
              self.inference_count += 1
              self.last_inference_at = datetime.now(timezone.utc).isoformat()
              self.inference_ready = True
              self.last_successful_inference_at = datetime.now(timezone.utc).isoformat()
              return {
                  "status": "success",
                  "model_id": self.model_id,
                  "version": self.model_version,
                  "prediction": float(pred)
              }
          raise RuntimeError(f"Artifact for '{self.model_id}' has no predict method")
      except Exception as exc:
          self.inference_failures += 1
          self.last_inference_at = datetime.now(timezone.utc).isoformat()
          self.inference_ready = False
          logger.error("Inference error for %s: %s", self.model_id, exc)
          raise

  def batch_predict(self, payloads):
      return [self.predict(p) for p in payloads]

  def explain(self, payload):
      importance = {}
      try:
          if isinstance(self._artifact, dict):
              feature_columns = self._artifact.get("feature_columns") or []
              importance = {feat: 0.0 for feat in feature_columns}
      except Exception:
          importance = {}
      return {
          "model_id": self.model_id,
          "confidence": 0.5,
          "feature_importance": importance,
          "explanation": "SHAP/LIME not yet wired"
      }

  def validate(self, payload):
      return "features" in payload

  def version(self):
      return self.model_version

  def metadata(self):
      return {
          "model_id": self.model_id,
          "version": self.model_version,
          "storage_id": self.storage_id,
          "is_loaded": self.is_loaded,
          "artifact_available": self._artifact is not None,
          "inference_ready": self.inference_ready,
          "health": "healthy" if self.is_loaded and self._artifact is not None else "degraded",
          "load_source": self._metadata.get("load_source"),
          "load_error": self._metadata.get("load_error"),
          "attempted_paths": self._metadata.get("attempted_paths"),
          "load_time": self._metadata.get("load_time"),
          "last_loaded_at": self.last_loaded_at,
          "last_inference_at": self.last_inference_at,
          "last_successful_inference_at": self.last_successful_inference_at,
          "inference_count": self.inference_count,
          "inference_failures": self.inference_failures,
          "has_artifact": self._artifact is not None,
          **self._metadata
      }

  def health_check(self):
      return self.is_loaded and self._artifact is not None
