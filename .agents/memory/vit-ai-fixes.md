---
name: vit-ai route prefix and key bugs
description: Critical fixes applied to vit-ai; what was broken and what the correct state is
---

## Fixes applied (2026-08-04)

### 1. Route prefix (CRITICAL)
- **Bug:** `app.include_router(router)` — no prefix, so all routes were at `/models`, `/infer`, etc.
- **Fix:** `app.include_router(router, prefix="/api/v1")`
- **Why:** vitnetwork's vit_ai_client called `/api/v1/chat`, `/api/v1/models` — all 404'd before fix

### 2. training.py datetime crash
- **Bug:** `datetime.now(datetime.UTC)` — AttributeError, `UTC` is a module constant not a class attr
- **Fix:** `datetime.now(UTC)` — UTC already imported at top of file

### 3. Auth key (security.py)
- **Bug:** Hardcoded fallback `"vit-internal-key"` — anyone reading the source could authenticate
- **Fix:** `settings.VIT_AI_API_KEY or os.getenv("INTERNAL_API_KEY", "")` — no auth when both absent
- **Render:** VIT_AI_API_KEY now set on srv-d97aur4s728c73d8jpk0

## Known remaining gaps
- Embedding service returns random 128-dim floats (stub, not a real model)
- /explain endpoint returns hardcoded stub data
- Training jobs + feature store + dataset registry are in-memory only (wiped on restart)
- /metrics Prometheus endpoint referenced in docs but not implemented
- 16 models trained on synthetic data — need retraining on real sports data for production accuracy
- CI uses Python 3.11, Dockerfile uses 3.12 (minor mismatch)

**How to apply:** When working on vit-ai, treat /api/v1 as the canonical prefix for all routes.
