from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram

# Shared CollectorRegistry for on-demand metric exposition and runtime updates
registry = CollectorRegistry()

# Gauges for registry state
g_registered = Gauge("vit_models_registered", "Number of registered models", registry=registry)
g_loaded = Gauge("vit_models_loaded", "Number of loaded model artifacts", registry=registry)
g_inference_ready = Gauge("vit_models_inference_ready", "Models inference-ready", registry=registry)
g_failed = Gauge("vit_models_failed", "Models failed to load or inference", registry=registry)

# Counters for inference totals
inference_total = Counter("inference_total", "Total inference attempts", registry=registry)
inference_failure_total = Counter("inference_failure_total", "Failed inference attempts", registry=registry)
inference_success_total = Counter("inference_success_total", "Successful inference attempts", registry=registry)

# Provider metrics
c_provider_requests = Counter("provider_requests_total", "Requests processed by provider", ["provider"], registry=registry)
g_provider_initialized = Gauge("vit_provider_initialized", "Provider initialized (1/0)", ["provider"], registry=registry)

# Per-model counters and latency histogram
c_model_inference = Counter("model_inference_count", "Per-model inference count", ["model_id"], registry=registry)
h_inference_latency = Histogram("inference_latency_seconds", "Inference latency seconds", ["model_id"], registry=registry, buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))


def record_inference(model_id: str, latency: float, success: bool = True) -> None:
    """Record a single inference event: latency and counters."""
    try:
        h_inference_latency.labels(model_id=model_id).observe(float(latency))
    except Exception:
        pass
    try:
        c_model_inference.labels(model_id=model_id).inc()
    except Exception:
        pass
    try:
        inference_total.inc()
    except Exception:
        pass
    if success:
        try:
            inference_success_total.inc()
        except Exception:
            pass
    else:
        try:
            inference_failure_total.inc()
        except Exception:
            pass


def record_provider_request(provider_name: str) -> None:
    """Record a single provider request event.

    Providers should call this when they actually process a request. This
    avoids snapshot-based increments which can double-count.
    """
    try:
        c_provider_requests.labels(provider=provider_name).inc()
    except Exception:
        pass


def snapshot_registry_state():
    """Pull runtime registry metrics (counts) and set gauges accordingly.

    This function is safe to call repeatedly before scraping/exporting metrics.
    """
    try:
        from app.services.registry import registry as model_registry
        g_registered.set(len(model_registry.get_all()))
        g_loaded.set(model_registry.loaded_model_count())
        g_inference_ready.set(model_registry.inference_ready_count())
        g_failed.set(model_registry.failed_model_count())
    except Exception:
        # On error, set zeros conservatively
        try:
            g_registered.set(0)
            g_loaded.set(0)
            g_inference_ready.set(0)
            g_failed.set(0)
        except Exception:
            pass


def snapshot_providers():
    """Sync provider counters from inference pipeline into counters/gauges."""
    try:
        from app.services.inference import inference_pipeline
        for name, prov in getattr(inference_pipeline, "providers", {}).items():
            try:
                c_provider_requests.labels(provider=name).inc(0)  # ensure label exists
                # set current requests_processed as a delta by reading attribute
                # use gauge for initialized state
                g_provider_initialized.labels(provider=name).set(1 if getattr(prov, "_is_initialized", False) else 0)
                # Set initialized gauge only; providers should call record_provider_request
                # to increment their request counters to avoid double-counting.
            except Exception:
                continue
    except Exception:
        return
