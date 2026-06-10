"""End-to-end test for the ray_metrics path: SchedulerActor invokes
sample_once, writes land in the head's diskcache, the Flask /api/ray/status
endpoint reads them back.

This is the test that would have caught both regressions from
2026-06-10:
  * SchedulerActor unpinned, scheduled on a worker → writes landed on
    the worker's disk, Flask on head read its own (stale) disk.
  * Module-level deque vs. cross-process diskcache mismatch from an
    earlier iteration.

Uses the real test Ray cluster set up by `init_ray` (in the project
conftest) and the existing Flask test_client pattern from
`test_endpoints.py`.
"""
from __future__ import annotations

import pytest
import ray


def _namespace() -> str:
    import os
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


@pytest.fixture
def client(test_context):
    """Flask test client backed by the same test_context (and therefore
    the same diskcache) the SchedulerActor writes into."""
    from web_server.backend.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_scheduler_actor_sample_lands_in_flask_endpoint(client):
    """Drive one sample through SchedulerActor.sample_ray_metrics_once
    and verify the Flask /api/ray/status response carries the freshly
    written history."""
    scheduler = ray.get_actor("scheduler", namespace=_namespace())

    # Endpoint should be reachable even before the first sample — but
    # cpuHistory may be empty.
    before = client.get("/api/ray/status").get_json()
    assert isinstance(before, dict)
    cpu_before = list(before.get("cpuHistory") or [])
    mem_before = list(before.get("memHistory") or [])

    # Trigger one sample synchronously inside the actor process and
    # block on the result so we know the diskcache write happened
    # before we hit the endpoint.
    ray.get(scheduler.sample_ray_metrics_once.remote())

    after = client.get("/api/ray/status").get_json()
    cpu_after = list(after.get("cpuHistory") or [])
    mem_after = list(after.get("memHistory") or [])

    # At least one new sample landed in each series after the tick.
    assert len(cpu_after) == len(cpu_before) + 1, (
        f"expected one new CPU sample after SchedulerActor tick "
        f"(before={len(cpu_before)}, after={len(cpu_after)})"
    )
    assert len(mem_after) == len(mem_before) + 1, (
        f"expected one new memory sample after SchedulerActor tick "
        f"(before={len(mem_before)}, after={len(mem_after)})"
    )

    # Values must be plausible psutil readings, not the previous
    # "Ray scheduled-resource" 12.5%/87% bookkeeping artifacts.
    last_cpu = cpu_after[-1]
    last_mem = mem_after[-1]
    assert 0.0 <= last_cpu <= 100.0
    assert 0.0 <= last_mem <= 100.0


def test_repeated_samples_grow_history(client):
    """Three consecutive ticks should produce three new samples — the
    rolling window only trims after 60. This guards the "constant
    sparkline" bug: stale value at the key, nothing overwriting it."""
    scheduler = ray.get_actor("scheduler", namespace=_namespace())

    before = client.get("/api/ray/status").get_json()
    cpu_before_len = len(before.get("cpuHistory") or [])

    for _ in range(3):
        ray.get(scheduler.sample_ray_metrics_once.remote())

    after = client.get("/api/ray/status").get_json()
    cpu_after_len = len(after.get("cpuHistory") or [])

    assert cpu_after_len == cpu_before_len + 3, (
        f"3 ticks should yield 3 new samples "
        f"(before={cpu_before_len}, after={cpu_after_len})"
    )


def test_scheduler_actor_is_pinned_to_head_node():
    """The SchedulerActor must run on the same Ray node as the driver
    (the head). If it lands on a worker, its diskcache writes go to
    the wrong machine's disk and the Flask endpoint reads stale data
    forever — that's exactly the "constant sparkline" bug we hit.

    On a single-node test cluster head == worker, so this is mostly
    sanity-checking the pinning code path doesn't error out. On a
    multi-node cluster (live cluster with a laptop joined) the same
    assertion catches a worker-placement regression.
    """
    scheduler = ray.get_actor("scheduler", namespace=_namespace())
    actor_node_id = ray.get(scheduler.get_node_id_for_introspection.remote())
    driver_node_id = ray.get_runtime_context().get_node_id()
    assert actor_node_id == driver_node_id, (
        f"SchedulerActor scheduled on node {actor_node_id!r}, "
        f"driver/head is {driver_node_id!r} — head-pin regression"
    )


def test_ray_status_endpoint_returns_expected_shape(client):
    """The endpoint contract the frontend RayStatus.jsx component
    depends on: a dict with cpuHistory / memHistory / objectStoreHistory
    list keys, plus the node/CPU/RAM summary fields."""
    scheduler = ray.get_actor("scheduler", namespace=_namespace())
    ray.get(scheduler.sample_ray_metrics_once.remote())

    data = client.get("/api/ray/status").get_json()
    assert isinstance(data, dict)

    # Shape contract — RayStatus.jsx reads these.
    for key in (
        "cpuHistory", "memHistory", "objectStoreHistory",
        "healthyNodes", "unhealthyNodes", "totalNodes",
        "totalCPUs", "totalRAM", "nodes",
    ):
        assert key in data, f"missing key {key!r}"

    assert isinstance(data["cpuHistory"], list)
    assert isinstance(data["memHistory"], list)
    assert isinstance(data["nodes"], list)
    # Single-node test cluster.
    assert data["totalNodes"] >= 1
    assert data["healthyNodes"] >= 1
