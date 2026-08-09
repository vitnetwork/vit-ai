"""Legacy compatibility subpackage for pickled models."""

import numpy as np


class LegacyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class LegacyEnsemble:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TinySequenceModel(LegacyModel):
    """Minimal compatibility class for older pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=int)

    def predict_proba(self, X):
        arr = np.asarray(X)
        probs = np.zeros((arr.shape[0], 2), dtype=float)
        probs[:, 1] = 0.5
        probs[:, 0] = 0.5
        return probs


class StackedMetaModel(LegacyEnsemble):
    """Minimal compatibility class for older ensemble pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=int)

    def predict_proba(self, X):
        arr = np.asarray(X)
        probs = np.zeros((arr.shape[0], 2), dtype=float)
        probs[:, 1] = 0.5
        probs[:, 0] = 0.5
        return probs


class PoissonModel(LegacyModel):
    """Minimal compatibility class for older Poisson-style pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=float)

    def predict_proba(self, X):
        arr = np.asarray(X)
        return np.column_stack((0.5 * np.ones(arr.shape[0]), 0.5 * np.ones(arr.shape[0])))


class DixonColesModel(LegacyModel):
    """Minimal compatibility class for older Dixon-Coles pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=float)

    def predict_proba(self, X):
        arr = np.asarray(X)
        return np.column_stack((0.5 * np.ones(arr.shape[0]), 0.5 * np.ones(arr.shape[0])))


class EloModel(LegacyModel):
    """Minimal compatibility class for older Elo pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=float)

    def predict_proba(self, X):
        arr = np.asarray(X)
        return np.column_stack((0.5 * np.ones(arr.shape[0]), 0.5 * np.ones(arr.shape[0])))


class BayesianNetModel(LegacyModel):
    """Minimal compatibility class for older Bayesian-network pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=float)

    def predict_proba(self, X):
        arr = np.asarray(X)
        return np.column_stack((0.5 * np.ones(arr.shape[0]), 0.5 * np.ones(arr.shape[0])))


class MarketImpliedModel(LegacyModel):
    """Minimal compatibility class for older market-implied pickles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_features_in_ = kwargs.get("n_features_in_", 10)

    def predict(self, X):
        arr = np.asarray(X)
        return np.ones(arr.shape[0], dtype=float)

    def predict_proba(self, X):
        arr = np.asarray(X)
        return np.column_stack((0.5 * np.ones(arr.shape[0]), 0.5 * np.ones(arr.shape[0])))


__all__ = ["LegacyModel", "LegacyEnsemble", "TinySequenceModel", "StackedMetaModel", "PoissonModel", "DixonColesModel", "EloModel", "BayesianNetModel", "MarketImpliedModel"]
