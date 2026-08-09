from prometheus_client import generate_latest
from app.metrics.updater import registry as shared_registry, snapshot_registry_state, snapshot_providers


def collect_metrics() -> bytes:
    """Update shared metrics from runtime and return the latest snapshot."""
    # Refresh snapshot values into shared_registry's metrics
    try:
        snapshot_registry_state()
        snapshot_providers()
    except Exception:
        pass
    return generate_latest(shared_registry)
