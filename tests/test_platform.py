import os
os.environ.setdefault("VIT_AI_API_KEY", "vit-internal-key")
os.environ.setdefault("MODEL_DIR", "/workspaces/vit-ai/models")

import pytest
from fastapi.testclient import TestClient
from app.main import app

headers = {"X-API-KEY": "vit-internal-key"}

def test_full_inference_lifecycle():
    # 1. Register a model with storage_id
    model_data = {
        "id": "production-xgb",
        "name": "Production XGB",
        "description": "Production model",
        "capabilities": ["prediction"],
        "provider": "internal",
        "input_schema": {},
        "output_schema": {},
        "initial_version": "1.0.0",
        "storage_id": "xgb_v1.pkl"
    }
    with TestClient(app) as client:
        client.post("/api/v1/models", json=model_data, headers=headers)

        # 2. Run inference on the model
        infer_data = {
            "model_id": "production-xgb",
            "payload": {"features": [0.5, 0.4, 0.3, 0.2, 0.25, 0.3, 0.45, 0.6, 0.5, 0.1]}
        }
        response = client.post("/api/v1/infer", json=infer_data, headers=headers)
        assert response.status_code == 200
        assert response.json()["metadata"]["provider"] == "internal"
        assert response.json()["result"]["provider"] == "internal"

def test_ensemble_with_registered_models():
    # Register another model to be part of ensemble
    model_data = {
        "id": "production-xgb",
        "name": "Production XGB",
        "description": "desc",
        "capabilities": ["prediction"],
        "provider": "internal",
        "input_schema": {},
        "output_schema": {},
        "initial_version": "1.0.0"
    }
    with TestClient(app) as client:
        client.post("/api/v1/models", json=model_data, headers=headers)

        # Run ensemble
        ensemble_data = {"market_odds": {"home": 2.0, "draw": 3.0, "away": 3.5}}
        response = client.post("/api/v1/ensemble", json=ensemble_data, headers=headers)
        assert response.status_code == 200
        assert response.json()["prediction"]["home_prob"] > 0
        assert isinstance(response.json()["prediction"].get("attribution"), list)
        assert len(response.json()["prediction"].get("attribution", [])) <= 3

def test_training_jobs():
    job_data = {
        "model_id": "production-lstm",
        "dataset_id": "ds-1",
        "params": {"epochs": 10}
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/training/jobs", json=job_data, headers=headers)
        assert response.status_code == 200
        job_id = response.json()["id"]

        response = client.get(f"/api/v1/training/jobs/{job_id}")
        assert response.json()["status"] == "queued"

def test_standardized_model_and_providers_interfaces():
    import asyncio
    from app.services.base_model import StandardizedModel
    from app.services.providers import InternalProvider, AdHocProvider, EnsembleProvider
    from app.services.ensemble import ensemble_engine

    async def run_checks():
        # 1. Test Model Interface
        model = StandardizedModel(model_id="xgb_v1", model_version="1.0", storage_id="xgb_v1.pkl")
        assert model.health_check() is False
        assert model.validate({"features": [0.1, 0.2, 0.3, 0.4, 0.5]}) is True
        assert model.version() == "1.0"
        assert model.metadata()["model_id"] == "xgb_v1"

        assert model.load() is True
        assert model.health_check() is True

        pred = model.predict({"features": [0.5] * 10})
        assert pred["status"] == "success"
        assert "prediction" in pred

        batch_preds = model.batch_predict([
            {"features": [0.2] * 10},
            {"features": [0.8] * 10}
        ])
        assert len(batch_preds) == 2

        explanation = model.explain({"features": [0.5] * 10})
        assert "confidence" in explanation
        assert "feature_importance" in explanation

        assert model.unload() is True
        assert model.health_check() is False

        # 2. Test Provider Interface
        internal_prov = InternalProvider()
        assert await internal_prov.initialize() is True
        assert await internal_prov.authenticate() is True
        assert await internal_prov.shutdown() is True
        met = await internal_prov.metrics()
        assert met["provider_type"] == "internal"

        adhoc_prov = AdHocProvider()
        assert await adhoc_prov.initialize() is True
        assert await adhoc_prov.authenticate() is True
        assert await adhoc_prov.shutdown() is True
        met_adhoc = await adhoc_prov.metrics()
        assert met_adhoc["provider_type"] == "adhoc"

        ensemble_prov = EnsembleProvider(ensemble_engine)
        assert await ensemble_prov.initialize() is True
        assert await ensemble_prov.authenticate() is True
        assert await ensemble_prov.shutdown() is True
        met_ens = await ensemble_prov.metrics()
        assert met_ens["provider_type"] == "ensemble"

    asyncio.run(run_checks())
