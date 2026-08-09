import importlib
import os
from pathlib import Path


def test_model_dir_defaults_to_workspace_models(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    import app.core.config as config_module

    importlib.reload(config_module)

    expected = str(Path.cwd() / "models")
    assert config_module.settings.MODEL_DIR == expected
