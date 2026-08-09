import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureAdapter:
    """Translate match-style payloads into the numeric vectors expected by VIT model artifacts."""

    def __init__(self):
        self._default_odds = {"home": 2.3, "draw": 3.3, "away": 3.1}

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value)
        except Exception:
            pass
        return default

    def _vig_free(self, home: float, draw: float, away: float) -> tuple[float, float, float]:
        total = home + draw + away
        if total <= 0:
            return (0.45, 0.25, 0.30)
        h = home / total
        d = draw / total
        a = away / total
        return (h, d, a)

    def _implied_probs(self, payload: Dict[str, Any]) -> tuple[float, float, float]:
        odds = payload.get("market_odds") or payload.get("odds") or {}
        if isinstance(odds, dict):
            home = self._coerce_float(odds.get("home") or odds.get("1"), self._default_odds["home"])
            draw = self._coerce_float(odds.get("draw") or odds.get("x"), self._default_odds["draw"])
            away = self._coerce_float(odds.get("away") or odds.get("2"), self._default_odds["away"])
            return self._vig_free(home, draw, away)
        return self._vig_free(self._default_odds["home"], self._default_odds["draw"], self._default_odds["away"])

    def _strength_ratio(self, payload: Dict[str, Any], home: float, away: float) -> float:
        # A simple deterministic proxy based on the implied strength of each side.
        return max(0.01, min(1.0, (home + 0.1) / (away + home + 0.2)))

    def build_feature_vector(self, payload: Dict[str, Any], feature_columns: Optional[List[str]] = None) -> List[float]:
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary")

        if "features" in payload:
            features = payload["features"]
            if isinstance(features, list):
                return [self._coerce_float(v) for v in features]
            if isinstance(features, dict):
                if feature_columns:
                    return [self._coerce_float(features.get(col)) for col in feature_columns]
                return [self._coerce_float(v) for v in features.values()]

        if feature_columns is None:
            feature_columns = []

        # Default to the 10-dim schema used by the bundled classifiers.
        home_prob, draw_prob, away_prob = self._implied_probs(payload)
        home_odds = self._coerce_float((payload.get("market_odds") or {}).get("home"), self._default_odds["home"])
        draw_odds = self._coerce_float((payload.get("market_odds") or {}).get("draw"), self._default_odds["draw"])
        away_odds = self._coerce_float((payload.get("market_odds") or {}).get("away"), self._default_odds["away"])

        # Heuristic defaults when richer match metadata is absent.
        home_form = self._coerce_float(payload.get("home_form"), 0.45)
        away_form = self._coerce_float(payload.get("away_form"), 0.45)
        home_elo = self._coerce_float(payload.get("home_elo"), 0.72)
        away_elo = self._coerce_float(payload.get("away_elo"), 0.68)
        h2h_rate = self._coerce_float(payload.get("h2h_home_rate"), 0.45)
        over25_implied = self._coerce_float(payload.get("over_25_implied"), 0.52)
        strength_ratio = self._strength_ratio(payload, home_prob, away_prob)

        vector = [
            home_odds,
            draw_odds,
            away_odds,
            home_prob,
            draw_prob,
            away_prob,
            over25_implied,
            strength_ratio,
            home_form,
            away_form,
        ]

        if feature_columns:
            mapping = {
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "home_implied": home_prob,
                "draw_implied": draw_prob,
                "away_implied": away_prob,
                "lam_h": home_prob,
                "lam_a": away_prob,
                "over_25_implied": over25_implied,
                "strength_ratio": strength_ratio,
            }
            return [self._coerce_float(mapping.get(col)) for col in feature_columns]

        return vector


feature_adapter = FeatureAdapter()
