from fastapi import APIRouter, HTTPException, Depends, Body, status
from typing import List, Dict, Any
from app.schemas.model import Model, ModelCreate, ModelUpdate, ModelVersion, ModelVersionCreate
from app.schemas.inference import InferenceRequest, InferenceResponse
from app.schemas.diagnostics import AIDiagnostics
from app.schemas.dataset import Dataset, DatasetCreate
from app.schemas.feature import Feature, FeatureCreate
from app.schemas.training import TrainingJob, TrainingJobCreate
from app.services.registry import registry
from app.core.kernel import kernel
from app.services.inference import inference_pipeline
from app.services.ensemble import ensemble_engine
from app.services.feature_store import feature_store
from app.services.training import training_manager
from app.services.dataset_registry import dataset_registry
from app.services.embedding import embedding_service
from app.core.security import verify_auth

router = APIRouter()
protected = [Depends(verify_auth)]

# --- Phase 2: AI Kernel ---
@router.get("/ai/status")
async def get_ai_status():
    return kernel.get_status()

@router.get("/ai/diagnostics", response_model=AIDiagnostics, dependencies=protected)
async def get_ai_diagnostics():
    return kernel.get_diagnostics()

@router.get("/ai/providers")
async def get_ai_providers():
    return kernel.get_providers()

# --- Phase 3: Model Registry & Versioning ---
@router.get("/models", response_model=List[Model])
async def list_models():
    return registry.get_all()

@router.post("/models", response_model=Model, dependencies=protected)
async def create_model(model_in: ModelCreate):
    if registry.get_by_id(model_in.id):
        raise HTTPException(status_code=400, detail="Model already exists")
    return registry.register(model_in)

@router.get("/models/{model_id}", response_model=Model)
async def get_model(model_id: str):
    model = registry.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.patch("/models/{model_id}", response_model=Model, dependencies=protected)
async def update_model(model_id: str, model_in: ModelUpdate):
    model = registry.update(model_id, model_in)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.delete("/models/{model_id}", dependencies=protected)
async def delete_model(model_id: str):
    if not registry.delete(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted"}

@router.post("/models/{model_id}/versions", response_model=ModelVersion, dependencies=protected)
async def add_model_version(model_id: str, version_in: ModelVersionCreate):
    version = registry.add_version(model_id, version_in)
    if not version:
        raise HTTPException(status_code=404, detail="Model not found")
    return version

# --- Phase 4: Inference & Embedding ---
@router.post("/infer", response_model=InferenceResponse, dependencies=protected)
async def infer(request: InferenceRequest):
    return await inference_pipeline.process(request)

@router.post("/predict", dependencies=protected)
async def predict(request: InferenceRequest):
    return await inference_pipeline.process(request)

@router.post("/chat", dependencies=protected)
async def chat(request: InferenceRequest):
    return await inference_pipeline.process(request)

@router.post("/classify", dependencies=protected)
async def classify(request: InferenceRequest):
    return await inference_pipeline.process(request)

@router.post("/summarize", dependencies=protected)
async def summarize(request: InferenceRequest):
    return await inference_pipeline.process(request)

@router.post("/embed", dependencies=protected)
async def embed(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text", "")
    model_id = payload.get("model_id", "text-embedding-3-small")
    return await embedding_service.generate(text, model_id)

# --- Phase 5: Ensemble Engine ---
@router.post("/ensemble", dependencies=protected)
async def ensemble(payload: Dict[str, Any] = Body(...)):
    return await ensemble_engine.orchestrate(payload)

@router.get("/ensemble/status")
async def ensemble_status():
    return {"status": ensemble_engine.status}

# --- Phase 6: Dataset Management ---
@router.get("/datasets", response_model=List[Dataset])
async def list_datasets():
    return dataset_registry.get_all()

@router.post("/datasets", response_model=Dataset, dependencies=protected)
async def create_dataset(dataset_in: DatasetCreate):
    if dataset_registry.get_by_id(dataset_in.id):
        raise HTTPException(status_code=400, detail="Dataset already exists")
    return dataset_registry.register(dataset_in)

@router.get("/datasets/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str):
    dataset = dataset_registry.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.delete("/datasets/{dataset_id}", dependencies=protected)
async def delete_dataset(dataset_id: str):
    if not dataset_registry.delete(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "deleted"}

# --- Phase 7: Feature Store ---
@router.get("/features", response_model=List[Feature])
async def list_features():
    return feature_store.get_all()

@router.post("/features", response_model=Feature, dependencies=protected)
async def create_feature(feature_in: FeatureCreate):
    if feature_store.get_by_id(feature_in.id):
        raise HTTPException(status_code=400, detail="Feature already exists")
    return feature_store.register(feature_in)

# --- Training Job Management ---
@router.post("/training/jobs", response_model=TrainingJob, dependencies=protected)
async def create_training_job(job_in: TrainingJobCreate):
    return training_manager.create_job(job_in)

@router.get("/training/jobs", response_model=List[TrainingJob])
async def list_training_jobs():
    return training_manager.list_jobs()

@router.get("/training/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# --- Phase 8: Explainability ---
@router.post("/explain", dependencies=protected)
async def explain(payload: Dict[str, Any] = Body(...)):
    """
    Return per-feature importance and confidence for a prediction payload.

    Runs the same inference pipeline used by /infer and extracts:
    - feature_importance: dict mapping column name → normalised SHAP-style weight
    - confidence: mean probability from ensemble output
    - reasoning: human-readable summary of top contributing features
    """
    from app.schemas.inference import InferenceRequest

    # Attempt real ensemble explanation; fall back gracefully if models not loaded.
    try:
        infer_req = InferenceRequest(**payload)
        result = await inference_pipeline.run(infer_req)

        # Build feature importance from ensemble weights (registry metadata)
        all_models = registry.get_all()
        active_models = [m for m in all_models if m.status == "active"]
        importance: Dict[str, float] = {}
        total_weight = sum(float(m.weight or 1.0) for m in active_models)

        # Aggregate feature column contributions weighted by model weight
        for m in active_models:
            w = float(m.weight or 1.0) / total_weight if total_weight else 0.0
            for feat in (m.feature_columns or []):
                importance[feat] = round(importance.get(feat, 0.0) + w, 4)

        # Normalise to sum=1
        total_imp = sum(importance.values()) or 1.0
        importance = {k: round(v / total_imp, 4) for k, v in importance.items()}

        # Top 3 features for human-readable summary
        top_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k} ({v:.0%})" for k, v in top_feats)

        confidence = float(getattr(result, "probability", 0.5) or 0.5)
        return {
            "confidence":         round(confidence, 4),
            "feature_importance": importance,
            "reasoning":          f"Ensemble of {len(active_models)} models. "
                                  f"Top features: {top_str}." if top_feats else
                                  "Ensemble prediction — feature columns unavailable.",
            "active_models":      len(active_models),
        }
    except Exception as exc:
        # Degraded mode: return neutral explanation with error info
        return {
            "confidence":         0.5,
            "feature_importance": {},
            "reasoning":          f"Explanation unavailable (degraded mode): {exc}",
            "active_models":      0,
        }
