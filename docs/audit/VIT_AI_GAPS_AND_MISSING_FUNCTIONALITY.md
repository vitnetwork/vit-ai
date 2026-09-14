# VIT-AI Gaps and Missing Functionality Audit

## Executive Summary
This document provides a comprehensive audit of gaps, missing functionality, degraded operational modes, and technical debt across the `vit-ai` service codebase.

---

## 1. Codebase Gaps & Resolved Issues

### GAP-01: `InferencePipeline` Missing `run()` Method
- **Severity**: Critical
- **Impact**: The `/explain` endpoint (`app/api/endpoints.py`) calls `await inference_pipeline.run(infer_req)`. Missing `run()` method caused runtime `AttributeError: 'InferencePipeline' object has no attribute 'run'`.
- **Status**: **RESOLVED**. Added `async def run(self, request: InferenceRequest)` alias method in `app/services/inference.py`.

### GAP-02: Hardcoded Test Environment Paths & Model Artifact Resolution
- **Severity**: Major
- **Impact**: Test modules hardcoded `MODEL_DIR="/workspaces/vit-ai/models"`. In environments where the path differs (such as `/app/models` or local working directory `models/`), artifact loading returned `False`, causing test failures.
- **Status**: **RESOLVED**.
  - Enhanced `StandardizedModel.load()` in `app/services/base_model.py` with multi-directory fallback resolution (`os.path.abspath("models")`, `/app/models`).
  - Updated test environment defaults in `tests/test_feature_adapter.py`, `tests/test_api.py`, and `tests/test_platform.py`.

### GAP-03: Pydantic v2 Deprecation Warnings and Protected Namespace Conflicts
- **Severity**: Minor / Code Hygiene
- **Impact**: Schema models emitted `PydanticDeprecatedSince20` and `UserWarning: Field "model_type"/"model_id" has conflict with protected namespace "model_"`.
- **Status**: **RESOLVED**. Updated schema classes in `app/schemas/model.py`, `app/schemas/dataset.py`, `app/schemas/feature.py`, `app/schemas/training.py`, and `app/schemas/inference.py` to use `model_config = ConfigDict(from_attributes=True, protected_namespaces=())`.

---

## 2. Platform Architecture & Service Degraded Modes

### GAP-04: External Integration Degraded Modes
- **Severity**: Informational / Expected Environment Behavior
- **Description**: The service operates in degraded mode when optional environment variables are absent:
  - `VIT_STORAGE_URL`: Model download from `vit-storage` degrades to local `.pkl` disk fallback.
  - `VIT_NETWORK_URL` & `ORACLE_PRIVATE_KEY`: Oracle on-chain settlement degrades gracefully.
  - `REDIS_URL`: Storage persistence falls back to process memory (state is not persisted across container restarts).

### GAP-05: Missing API Endpoints for Full CRUD Capabilities
- **Severity**: Moderate
- **Description**:
  - `StandardizedModel` supports `batch_predict()`, but no dedicated `/api/v1/infer/batch` endpoint is exposed.
  - Feature Store and Dataset Registry provide `/features` and `/datasets` GET/POST endpoints, but lack PATCH/UPDATE endpoints.

---

## 3. Verification & Test Suite Status

- **Unit & Integration Tests**: 15/15 tests passing cleanly across `test_api.py`, `test_feature_adapter.py`, `test_model_dir_resolution.py`, and `test_platform.py`.
- **Real Validation Suite**: 16/16 core models (`xgb_v1`, `lstm_v1`, `transformer_v1`, `rf_v1`, `gbm_v1`, `bayes_v1`, `logistic_v1`, `elo_v1`, `poisson_v1`, `dixon_coles_v1`, `hybrid_v1`, `market_v1`, `ensemble_v1`, `btts_v2`, `over_under_v2`, `correct_score_v2`) successfully loaded and verified.
