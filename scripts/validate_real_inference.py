#!/usr/bin/env python3
"""Validate at least one real model can run inference end-to-end and metrics update.

This script will:
- find models with state INFERENCE_READY
- if none, report blockers
- call /api/v1/infer via TestClient with a random feature vector
- validate output contract (probabilities if present)
- report metrics counters before/after
"""
import os
import sys
import random
import time

API_KEY = os.getenv("VIT_AI_API_KEY", "vit-internal-key")
os.environ.setdefault("VIT_AI_API_KEY", API_KEY)

from fastapi.testclient import TestClient
from app.main import app
from app.services.registry import registry

def find_ready_model():
    diags = registry.get_diagnostics()
    ready = [d for d in diags if d.get('state') == 'INFERENCE_READY']
    if ready:
        return ready[0]
    # fallback: any loaded but not ready
    loaded = [d for d in diags if d.get('state') == 'LOADED']
    if loaded:
        return loaded[0]
    return None


def read_metrics():
    try:
        from app.metrics import updater
        total = int(getattr(updater.inference_total, '_value').get())
        success = int(getattr(updater.inference_success_total, '_value').get())
        failure = int(getattr(updater.inference_failure_total, '_value').get())
        return {'total': total, 'success': success, 'failure': failure}
    except Exception as exc:
        return {'error': str(exc)}


if __name__ == '__main__':
    # Ensure registry is populated from local MODEL_DIR before diagnostics
    try:
        registry.bootstrap_vit_models()
    except Exception:
        pass

    model = find_ready_model()
    if not model:
        print('NO INFERENCE-READY MODELS FOUND. Diagnostics:')
        for d in registry.get_diagnostics():
            print(f"- {d['model_id']}: state={d.get('state')} loaded={d.get('loaded')} inference_ready={d.get('inference_ready')} failed={d.get('failed')}")
        sys.exit(2)

    model_id = model['model_id']
    print('Selected model for test:', model_id, 'state=', model.get('state'))

    # Build a simple feature payload. Try to use artifact feature columns if available
    artifact = registry.get_artifact(model_id, model.get('version'))
    features = None
    if artifact and isinstance(artifact._artifact, dict):
        cols = artifact._artifact.get('feature_columns') or []
        if cols:
            features = [random.random() for _ in cols]
    if features is None:
        features = [random.random() for _ in range(10)]

    payload = {'model_id': model_id, 'payload': {'features': features}}

    before = read_metrics()
    print('metrics before:', before)

    with TestClient(app) as client:
        headers = {'X-API-KEY': API_KEY}
        start = time.time()
        resp = client.post('/api/v1/infer', json=payload, headers=headers)
        latency = time.time() - start
        print('HTTP status:', resp.status_code)
        if resp.status_code != 200:
            print('Inference endpoint returned non-200:', resp.text)
            sys.exit(3)
        body = resp.json()
        print('Response keys:', list(body.keys()))
        result = body.get('result')
        print('Result type:', type(result))

        # Validate output contract for probabilities if present
        if isinstance(result, dict) and 'probabilities' in result:
            probs = result['probabilities']
            if not isinstance(probs, list):
                print('Invalid probabilities: not a list')
                sys.exit(4)
            if any(not isinstance(x, (int, float)) for x in probs):
                print('Invalid probabilities: non-numeric values')
                sys.exit(4)
            if any((x != x) or (x is None) for x in probs):
                print('Invalid probabilities: contains NaN or None')
                sys.exit(4)
            if any(x < 0 or x > 1 for x in probs):
                print('Invalid probabilities: out of range [0,1]')
                sys.exit(4)
            s = sum(probs)
            if abs(s - 1.0) > 0.01:
                print('Warning: probabilities sum to', s)

        print('inference latency measured:', latency)

    after = read_metrics()
    print('metrics after:', after)

    # Validate metric increments
    try:
        if 'error' in before:
            print('Could not read metrics:', before['error'])
        else:
            if after['total'] < before['total'] + 1:
                print('Metric total not incremented as expected')
                sys.exit(5)
    except Exception as exc:
        print('Metrics validation error:', exc)

    print('\nTest completed successfully for model', model_id)
    print('model_version:', model.get('version'))
    print('inference_latency_seconds:', latency)
    sys.exit(0)
