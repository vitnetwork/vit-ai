"""Compatibility package for legacy market model pickles."""

import numpy as np

from app.services.base_model import StandardizedModel


class BTTSModel(StandardizedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(model_id="btts_v2", model_version="2.0", storage_id=None)
        self.args = args
        self.kwargs = kwargs

    def predict(self, payload):
        return {"status": "success", "model_id": "btts_v2", "prediction": 0.5, "probabilities": [0.5, 0.5]}


class OverUnderModel(StandardizedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(model_id="over_under_v2", model_version="2.0", storage_id=None)
        self.args = args
        self.kwargs = kwargs

    def predict(self, payload):
        return {"status": "success", "model_id": "over_under_v2", "prediction": 0.5, "probabilities": [0.5, 0.5]}


class CorrectScoreModel(StandardizedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(model_id="correct_score_v2", model_version="2.0", storage_id=None)
        self.args = args
        self.kwargs = kwargs

    def predict(self, payload):
        return {"status": "success", "model_id": "correct_score_v2", "prediction": 0.5, "probabilities": [0.5, 0.5]}


__all__ = ["StandardizedModel", "BTTSModel", "OverUnderModel", "CorrectScoreModel"]
