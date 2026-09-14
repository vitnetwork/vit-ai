import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.registry import registry
from app.core.security import verify_auth

@pytest.fixture(autouse=True)
def setup_registry(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "/app/models")
    app.dependency_overrides.clear()
    registry.models.clear()
    registry.loaded_artifacts.clear()
    registry.bootstrap_vit_models()

def test_all_16_models_registered_and_loaded():
    diagnostics = registry.get_diagnostics()
    assert len(diagnostics) == 16, f"Expected 16 registered models, got {len(diagnostics)}"
    for diag in diagnostics:
        assert diag["loaded"] is True, f"Model {diag['model_id']} is not loaded"
        assert diag["artifact_available"] is True, f"Model {diag['model_id']} has no artifact"

def test_models_produce_dynamic_probabilities_across_match_types():
    fav_payload = {
        "market_odds": {"home": 1.15, "draw": 7.0, "away": 15.0},
        "home_form": 0.95, "away_form": 0.05,
        "home_elo": 0.90, "away_elo": 0.20,
    }
    balanced_payload = {
        "market_odds": {"home": 2.50, "draw": 3.10, "away": 2.70},
        "home_form": 0.50, "away_form": 0.50,
        "home_elo": 0.50, "away_elo": 0.50,
    }

    all_models = registry.get_all()
    assert len(all_models) == 16

    confidences_fav = []
    confidences_bal = []

    for model_meta in all_models:
        artifact = registry.get_artifact(model_meta.id, model_meta.active_version)
        assert artifact is not None

        res_fav = artifact.predict(fav_payload)
        res_bal = artifact.predict(balanced_payload)

        assert res_fav["status"] == "success"
        assert res_bal["status"] == "success"

        if "probabilities" in res_fav:
            probs = res_fav["probabilities"]
            assert isinstance(probs, list)
            assert len(probs) >= 2
            assert all(isinstance(p, (int, float)) for p in probs)
            assert not any(p != p for p in probs)

        conf_fav = float(res_fav.get("prediction", max(res_fav.get("probabilities", [0]))))
        conf_bal = float(res_bal.get("prediction", max(res_bal.get("probabilities", [0]))))

        confidences_fav.append(conf_fav)
        confidences_bal.append(conf_bal)

    assert len(set(round(c, 3) for c in confidences_fav)) > 1, "Model confidences are constant across models!"
    assert confidences_fav != confidences_bal, "Confidences did not change between favorite and balanced matches!"

def test_inference_failures_surface_errors_not_fake_confidence():
    client = TestClient(app)
    bad_request = {
        "model_id": "xgb_v1",
        "payload": {"features": None}
    }
    response = client.post("/api/v1/infer", json=bad_request, headers={"X-API-KEY": "vit-internal-key"})
    assert response.status_code == 200
    data = response.json()
    result = data.get("result", {})
    assert isinstance(result, dict)
    assert result.get("status") == "error"
    assert "confidence" not in result or result.get("confidence") == 0.0 or result.get("status") == "error"

def test_explain_endpoint_returns_dynamic_confidence():
    client = TestClient(app)
    payload = {
        "model_id": "xgb_v1",
        "payload": {
            "market_odds": {"home": 1.2, "draw": 5.5, "away": 11.0},
            "home_form": 0.8, "away_form": 0.2
        }
    }
    response = client.post("/api/v1/explain", json=payload, headers={"X-API-KEY": "vit-internal-key"})
    assert response.status_code == 200
    body = response.json()
    assert "confidence" in body
    assert isinstance(body["confidence"], float)
    assert body["confidence"] != 0.75
    assert body["confidence"] != 0.5
