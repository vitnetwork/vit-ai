import os
os.environ.setdefault("VIT_AI_API_KEY", "vit-internal-key")
os.environ.setdefault("MODEL_DIR", os.getenv("MODEL_DIR") or os.path.abspath("models"))

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

def test_ai_status():
    with TestClient(app) as client:
        response = client.get("/api/v1/ai/status")
        assert response.status_code == 200


def test_ai_diagnostics_requires_auth():
    with TestClient(app) as client:
        response = client.get("/api/v1/ai/diagnostics")
        assert response.status_code == 401


def test_ai_diagnostics_with_auth():
    headers = {"X-API-KEY": "vit-internal-key"}
    with TestClient(app) as client:
        response = client.get("/api/v1/ai/diagnostics", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert "summary" in body
        assert "models" in body
        assert body["summary"]["models_registered"] >= 0
        assert "components" in body
        assert isinstance(body["models"], list)


def test_ai_providers():
    with TestClient(app) as client:
        response = client.get("/api/v1/ai/providers")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_protected_endpoint_no_auth():
    with TestClient(app) as client:
        response = client.post("/api/v1/models", json={})
        assert response.status_code == 401


def test_protected_endpoint_with_api_key():
    headers = {"X-API-KEY": "vit-internal-key"}
    model_data = {
        "id": "test-model-auth",
        "name": "Auth Model",
        "version": "1.0.0",
        "description": "desc",
        "capabilities": [],
        "provider": "internal",
        "input_schema": {},
        "output_schema": {}
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/models", json=model_data, headers=headers)
        assert response.status_code == 200


def test_infer_smoke_local_model():
    # Smoke test: call /infer for a seeded local model with a valid feature vector
    import random
    features = [random.random() for _ in range(10)]
    payload = {"model_id": "xgb_v1", "payload": {"features": features}}
    headers = {"X-API-KEY": "vit-internal-key"}
    with TestClient(app) as client:
        response = client.post("/api/v1/infer", json=payload, headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body.get("model_id") == "xgb_v1"
        assert "result" in body


def test_infer_batch_smoke_local_model():
    headers = {"X-API-KEY": "vit-internal-key"}
    payload = {
        "model_id": "xgb_v1",
        "payloads": [
            {"features": [0.1] * 10},
            {"features": [0.9] * 10},
        ],
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/infer/batch", json=payload, headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == "xgb_v1"
        assert len(body["results"]) == 2
        assert all("status" in item for item in body["results"])


def test_feature_and_dataset_patch_endpoints():
    headers = {"X-API-KEY": "vit-internal-key"}
    feature_payload = {
        "id": "test-feature-patch",
        "name": "Patch Feature",
        "type": "numeric",
        "description": "used for patch coverage",
        "metadata": {"source": "api-test"},
    }
    dataset_payload = {
        "id": "test-dataset-patch",
        "name": "Patch Dataset",
        "version": "1.0",
        "description": "used for patch coverage",
        "checksum": "abc123",
        "metadata": {"source": "api-test"},
    }

    with TestClient(app) as client:
        feature_response = client.post("/api/v1/features", json=feature_payload, headers=headers)
        assert feature_response.status_code == 200

        updated_feature = client.patch(
            "/api/v1/features/test-feature-patch",
            json={"name": "Updated Feature", "description": "updated via patch"},
            headers=headers,
        )
        assert updated_feature.status_code == 200
        assert updated_feature.json()["name"] == "Updated Feature"

        dataset_response = client.post("/api/v1/datasets", json=dataset_payload, headers=headers)
        assert dataset_response.status_code == 200

        updated_dataset = client.patch(
            "/api/v1/datasets/test-dataset-patch",
            json={"name": "Updated Dataset", "description": "updated via patch"},
            headers=headers,
        )
        assert updated_dataset.status_code == 200
        assert updated_dataset.json()["name"] == "Updated Dataset"
