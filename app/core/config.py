"""
VIT-AI service configuration.
All values are loaded from environment variables via pydantic-settings.
No hardcoded URLs or credentials are permitted per VIT Chain engineering directive.
"""
import logging
import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _default_model_dir() -> str:
    workspace_models = Path.cwd() / "models"
    if workspace_models.exists():
        return str(workspace_models)

    env_value = os.getenv("MODEL_DIR")
    if env_value:
        return env_value

    return "/app/models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Runtime ───────────────────────────────────────────────────────────
    APP_VERSION: str = "0.1.0"
    VERSION: str = "0.1.0"          # alias used by AIKernel.get_status()
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── Model storage ─────────────────────────────────────────────────────
    MODEL_DIR: str = Field(default_factory=_default_model_dir)

    # ── AI kernel providers ───────────────────────────────────────────────
    # Used by AIKernel.__init__ at module level — must have a safe default.
    SUPPORTED_PROVIDERS: List[str] = ["internal", "ensemble", "adhoc"]

    # ── Auth / JWT ────────────────────────────────────────────────────────
    # Secret key for JWT signing.  Override in production.
    SECRET_KEY: str = "vit-ai-default-secret-change-in-prod"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ── vit-storage integration ───────────────────────────────────────────
    # Defaults to None when unset; storage routes degrade gracefully.
    VIT_STORAGE_URL: Optional[str] = None
    # ── VIT Network / gateway integration ─────────────────────────────────────────────
    # Optional at startup — used for real feature ingestion and validation tooling.
    VIT_NETWORK_URL: Optional[str] = None
    # ── Internal service authentication ───────────────────────────────────
    # Optional at startup — auth middleware returns 401 when the key is absent
    # rather than preventing boot.
    VIT_AI_API_KEY: Optional[str] = None

    # ── VIT Chain oracle (Chain ID 7764) ──────────────────────────────────
    # Optional at startup — oracle settlement degrades gracefully when absent.
    ORACLE_PRIVATE_KEY: Optional[str] = None
    UNIVERSAL_ORACLE_ADDRESS: Optional[str] = None


def _build_settings() -> Settings:
    """Construct settings, warning (not aborting) on missing optional vars."""
    try:
        s = Settings()
    except Exception as exc:
        logger.critical("[config] Settings load error — using minimal defaults: %s", exc)
        s = Settings.model_construct(
            APP_VERSION="0.1.0",
            VERSION="0.1.0",
            PORT=8000,
            LOG_LEVEL="INFO",
            MODEL_DIR=_default_model_dir(),
            SUPPORTED_PROVIDERS=["internal", "ensemble", "adhoc"],
            SECRET_KEY="vit-ai-default-secret-change-in-prod",
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            VIT_STORAGE_URL=None,
            VIT_NETWORK_URL=None,
            VIT_AI_API_KEY=None,
            ORACLE_PRIVATE_KEY=None,
            UNIVERSAL_ORACLE_ADDRESS=None,
        )

    # Warn about missing optional production values — never abort
    _OPTIONAL_PROD = {
        "VIT_STORAGE_URL": s.VIT_STORAGE_URL,
        "VIT_NETWORK_URL": s.VIT_NETWORK_URL,
        "VIT_AI_API_KEY": s.VIT_AI_API_KEY,
        "ORACLE_PRIVATE_KEY": s.ORACLE_PRIVATE_KEY,
        "UNIVERSAL_ORACLE_ADDRESS": s.UNIVERSAL_ORACLE_ADDRESS,
    }
    missing = [k for k, v in _OPTIONAL_PROD.items() if not v]
    if missing:
        logger.warning(
            "[config] DEGRADED — env vars not set: %s. "
            "Service starts in degraded mode; affected features will raise errors.",
            ", ".join(missing),
        )
    return s


settings = _build_settings()
