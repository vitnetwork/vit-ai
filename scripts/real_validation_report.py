#!/usr/bin/env python3
"""Produce a REAL validation report for local model artifacts.

Strict rules: no synthetic inputs; only use artifact-provided samples or external services
configured via environment (e.g., VIT_NETWORK_URL). If no real input available,
report NOT VALIDATED.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

from app.services.registry import registry
from app.services.base_model import StandardizedModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
VIT_NETWORK_URL = os.getenv("VIT_NETWORK_URL")

report = {
    "real_model_inventory": [],
    "real_inference": [],
    "sports_ai": [],
    "providers": {},
    "storage": {},
    "metrics": {},
    "registry_transitions": [],
}

# Bootstrap registry to ensure registered models exist
try:
    registry.bootstrap_vit_models()
except Exception:
    pass

# Discover actual .pkl files
files = [p for p in MODEL_DIR.glob("*.pkl") if p.is_file()]
if not files:
    print("NO LOCAL MODEL ARTIFACTS FOUND")
    sys.exit(2)

# Build inventory per artifact file
for p in sorted(files):
    model_id = p.stem
    size = p.stat().st_size
    # Try to find registry entry
    reg = registry.get_by_id(model_id)
    registered = reg is not None
    # Use StandardizedModel to load artifact in isolation to inspect
    sm = StandardizedModel(model_id=model_id, model_version="unknown", storage_id=None)
    loaded = False
    load_err = None
    try:
        sm.load()
        loaded = sm.is_loaded
    except Exception as exc:
        load_err = str(exc)
        loaded = sm.is_loaded

    artifact_type = type(sm._artifact).__name__ if sm._artifact is not None else None
    inference_ready = sm.inference_ready
    metadata = sm.metadata() if sm._artifact is not None else {}

    # compute state similar to registry logic
    failed_flag = bool(metadata.get("load_error") or (loaded and not inference_ready))
    if failed_flag:
        state = "FAILED"
    elif inference_ready:
        state = "INFERENCE_READY"
    elif loaded:
        state = "LOADED"
    else:
        state = "ARTIFACT_AVAILABLE" if sm._artifact is not None else "UNAVAILABLE"

    report["real_model_inventory"].append({
        "model_id": model_id,
        "version": getattr(reg, "active_version", sm.model_version if sm.model_version else "unknown") if reg else sm.model_version,
        "artifact_path": str(p),
        "artifact_type": artifact_type,
        "artifact_size": size,
        "registered": registered,
        "artifact_available": sm._artifact is not None,
        "loaded": loaded,
        "inference_ready": inference_ready,
        "state": state,
        "load_source": metadata.get("load_source"),
        "load_error": metadata.get("load_error") or load_err,
    })

# For each inference-ready model, attempt real inference only when a real input is available
for entry in report["real_model_inventory"]:
    if not entry["inference_ready"]:
        entry_report = {"model_id": entry["model_id"], "validated": False, "reason": "Not inference-ready"}
        report["real_inference"].append(entry_report)
        continue

    # load standardized model from registry if available else isolated
    model_id = entry["model_id"]
    model_version = entry["version"]
    artifact = registry.get_artifact(model_id, model_version) if registry.get_artifact(model_id, model_version) else StandardizedModel(model_id, model_version)
    # If artifact contains sample input, use it
    sample = None
    try:
        if isinstance(artifact._artifact, dict):
            sample = artifact._artifact.get("sample_input") or artifact._artifact.get("example")
    except Exception:
        sample = None

    # If no sample, check whether this is a sports model and VIT_NETWORK_URL is configured
    is_sports = any(t in model_id for t in ("xgb", "rf", "gbm", "ensemble", "market", "btts", "over_under", "correct_score"))
    inference_entry = {"model_id": model_id, "model_version": model_version}

    if sample:
        # Use sample input as-is (must be real per artifact authorship)
        try:
            start = time.time()
            result = artifact.predict(sample)
            latency = time.time() - start
            inference_entry.update({"input_schema": "artifact_sample", "inference_success": True, "latency": latency, "output": result})
        except Exception as exc:
            inference_entry.update({"input_schema": "artifact_sample", "inference_success": False, "error": str(exc)})
        report["real_inference"].append(inference_entry)
        continue

    if is_sports and VIT_NETWORK_URL:
        try:
            import requests

            def _get_gateway_json(path: str, params=None):
                url = f"{VIT_NETWORK_URL.rstrip('/')}{path}"
                resp = requests.get(url, params=params or {}, timeout=10)
                resp.raise_for_status()
                return resp.json()

            # Prefer real feature records when the gateway exposes them.
            features_result = None
            try:
                features_payload = _get_gateway_json("/api/pipeline/features", {"active_only": "true", "limit": 1})
                if isinstance(features_payload, dict):
                    records = features_payload.get("records") or []
                    if records:
                        first = records[0]
                        if isinstance(first, dict) and first.get("features"):
                            features_result = first["features"]
            except Exception:
                features_result = None

            # Fallback to a match record if feature records are unavailable.
            if features_result is None:
                try:
                    match_payload = _get_gateway_json("/api/matches", {"status": "upcoming", "limit": 1})
                    if isinstance(match_payload, dict):
                        matches = match_payload.get("matches") or []
                        if matches:
                            match = matches[0]
                            features_result = match.get("features") or match.get("match_features")
                except Exception:
                    features_result = None

            if features_result:
                start = time.time()
                result = artifact.predict({"features": features_result})
                latency = time.time() - start
                inference_entry.update({"input_schema": "gateway_real_features", "inference_success": True, "latency": latency, "output": result})
            else:
                inference_entry.update({"validated": False, "reason": "Gateway did not provide a real engineered feature vector"})
        except Exception as exc:
            inference_entry.update({"validated": False, "reason": f"Gateway validation error: {exc}"})
        report["real_inference"].append(inference_entry)
        continue

    # If no artifact sample and no gateway, we cannot fabricate inputs
    inference_entry.update({"validated": False, "reason": "NO REAL INPUT AVAILABLE — NOT VALIDATED"})
    report["real_inference"].append(inference_entry)

# Providers: snapshot counts
try:
    from app.metrics import updater
    report["providers"]["provider_requests_total"] = {str(k): int(v._value.get()) for k, v in getattr(updater, 'c_provider_requests')._value.items()} if hasattr(updater, 'c_provider_requests') else {}
except Exception:
    report["providers"]["error"] = "could not read provider counters"

# Storage: check VIT_STORAGE_URL
report["storage"]["VIT_STORAGE_URL"] = bool(os.getenv("VIT_STORAGE_URL"))

# Metrics summary
try:
    from app.metrics import updater
    report["metrics"]["inference_total"] = int(updater.inference_total._value.get())
    report["metrics"]["inference_success_total"] = int(updater.inference_success_total._value.get())
    report["metrics"]["inference_failure_total"] = int(updater.inference_failure_total._value.get())
except Exception:
    pass

# Registry transitions: dump diagnostics
for d in registry.get_diagnostics():
    report["registry_transitions"].append({"model_id": d["model_id"], "state": d.get("state"), "reason": d.get("load_error")})

# Output report as pretty JSON
print(json.dumps(report, indent=2, sort_keys=True))

# Enforce failure if no inference-ready models validated
validated = [r for r in report["real_inference"] if r.get("inference_success")]
if not validated:
    print("NO REAL INFERENCE-READY MODEL AVAILABLE OR NO REAL INPUTS — exiting non-zero")
    sys.exit(3)

print("REAL VALIDATION REPORT: at least one model validated.")
sys.exit(0)
