import os
os.environ.setdefault("VIT_AI_API_KEY", "vit-internal-key")
os.environ.setdefault("MODEL_DIR", "/workspaces/vit-ai/models")

from app.services.feature_adapter import feature_adapter
from app.services.base_model import StandardizedModel


def test_feature_adapter_builds_classifier_vector():
    payload = {
        "market_odds": {"home": 2.10, "draw": 3.30, "away": 3.80},
        "home_form": 0.6,
        "away_form": 0.4,
        "home_elo": 0.72,
        "away_elo": 0.68,
        "h2h_home_rate": 0.55,
        "over_25_implied": 0.61,
    }
    vector = feature_adapter.build_feature_vector(payload, ["home_odds", "draw_odds", "away_odds", "home_implied", "draw_implied", "away_implied", "lam_h", "lam_a", "over_25_implied", "strength_ratio"])
    assert len(vector) == 10
    assert vector[0] == 2.10
    assert vector[8] == 0.61
    assert vector[9] > 0


def test_model_predict_accepts_match_payload():
    model = StandardizedModel(model_id="xgb_v1", model_version="1.0", storage_id="xgb_v1.pkl")
    assert model.load() is True
    payload = {
        "market_odds": {"home": 2.10, "draw": 3.30, "away": 3.80},
        "home_form": 0.6,
        "away_form": 0.4,
        "home_elo": 0.72,
        "away_elo": 0.68,
        "h2h_home_rate": 0.55,
        "over_25_implied": 0.61,
    }
    result = model.predict(payload)
    assert result["status"] == "success"
    assert "probabilities" in result
