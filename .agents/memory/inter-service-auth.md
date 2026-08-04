---
name: VIT inter-service auth contract
description: How vitnetwork authenticates to vit-ai; shared key setup; what's wired and what's not
---

## Current auth mechanism (as of 2026-08-04)

**Shared API key approach:**
- `VIT_AI_API_KEY` set on vit-ai's Render service (srv-d97aur4s728c73d8jpk0)
- `VIT_AI_API_KEY` set on vitnetwork's Render service (srv-d8sipgjeo5us73eis7hg)
- vitnetwork sends `X-API-KEY: {VIT_AI_API_KEY}` on all outgoing requests via `vit_ai_client._auth_headers()`
- vit-ai validates in `security.py`: `settings.VIT_AI_API_KEY or os.getenv("INTERNAL_API_KEY", "")`

## HMAC service token system (built, not activated)
- `app/core/service_auth.py` in vitnetwork has a full HMAC-SHA256 signed token system
- Requires `SERVICE_TOKEN_SECRET` env var set on both services
- Token format: `{service_name}.{minute_bucket}.{hmac_sig}` — TTL ~2 min
- Not yet wired into vit-ai for validation — future upgrade path

## Chat payload contract
- vitnetwork sends: `{"model_id": "ensemble_v1", "payload": {"prompt": "...", ...}}`
- vit-ai expects: `InferenceRequest(model_id, payload)` → `InferenceResponse(request_id, model_id, result, latency, metadata)`
- vitnetwork reads: `data.get("result")` from InferenceResponse

**Why:** Getting the auth wrong caused all protected vit-ai endpoints to 401; the X-API-KEY approach is the active contract to maintain.
