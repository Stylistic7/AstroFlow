"""End-to-end tests: pipeline → Flask API → JSON responses.

These tests run the full compute graph on test data, then verify that
the Flask API endpoints return correct, complete JSON responses. This
is the closest thing to testing the UI without rendering React.
"""
import pytest


@pytest.fixture
def client(test_context):
    """Flask test client backed by a fresh test context."""
    from web_server.backend.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.exhaustive
def test_orion_mosaic_endpoints(client, with_orion_mosaic):
    """Verify all API endpoints return correct data for a 4-pane Orion mosaic."""
    # /stacks
    stacks = client.get("/api/stacks").get_json()
    assert isinstance(stacks, dict)
    assert len(stacks) > 0
    stack = next(iter(stacks.values()))
    for field in ["name", "center", "image_scale", "rotation_angle",
                  "camera", "filter", "exposure", "integration",
                  "width_pixels", "height_pixels", "stable_hash"]:
        assert field in stack, f"Missing field: {field}"

    # /targets
    targets = client.get("/api/targets").get_json()
    assert isinstance(targets, dict)
    assert len(targets) == 4
    target = next(iter(targets.values()))
    assert "name" in target
    assert "center" in target
    assert "stack_ids" in target
    assert len(target["center"]) == 2

    # /mosaics
    mosaics = client.get("/api/mosaics").get_json()
    assert isinstance(mosaics, dict)
    assert len(mosaics) == 1
    mosaic_id = next(iter(mosaics.keys()))
    mosaic = mosaics[mosaic_id]
    assert "name" in mosaic
    assert "target_ids" in mosaic
    assert "strips" in mosaic
    assert len(mosaic["target_ids"]) == 4

    # /calibration_stacks
    cal_stacks = client.get("/api/calibration_stacks").get_json()
    assert isinstance(cal_stacks, dict)
    assert len(cal_stacks) > 0

    # /metadata — no master images (no shopping cart), returns 404
    stack_id = next(iter(stacks.keys()))
    resp = client.get(f"/api/metadata/stack/{stack_id}")
    assert resp.status_code == 404

    # /thumbnail
    resp = client.get(f"/thumbnail/stack/{stack_id}")
    assert resp.status_code in [200, 404]

    # /get_shopping_cart
    cart = client.get("/api/shopping_cart").get_json()
    assert isinstance(cart, list)

    # /compute_graph_status
    assert client.get("/api/compute_graph/status").status_code == 200

    # /ray_status
    data = client.get("/api/ray/status").get_json()
    assert isinstance(data, dict)

    # /api/performance
    data = client.get("/api/performance").get_json()
    assert "functions" in data

    # Verify targets reference valid stacks
    for target in targets.values():
        for sid in target["stack_ids"]:
            assert sid in stacks, f"Target references nonexistent stack {sid}"


@pytest.mark.exhaustive
def test_orion_mosaic_versions_endpoint(client, with_orion_mosaic):
    """Exercise the per-pipeline tap-points endpoint that the
    dashboard reads."""
    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))

    # /api/pipeline_outputs/mosaic/<id>: returns {filters, pipelines}
    # where each pipeline carries its discovered taps (display-labelled
    # @makes_image nodes the pipeline trace surfaced) and each tap's
    # per-filter on-disk presence.
    resp = client.get(f"/api/pipeline_outputs/mosaic/{mosaic_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "filters" in data and "pipelines" in data
    assert isinstance(data["filters"], list) and len(data["filters"]) > 0

    pipelines = data["pipelines"]
    assert len(pipelines) > 0
    saw_existing = False
    for p in pipelines:
        assert set(p.keys()) >= {"name", "taps"}
        assert isinstance(p["taps"], list)
        for t in p["taps"]:
            assert set(t.keys()) >= {"display", "outputs"}
            assert isinstance(t["outputs"], list)
            assert len(t["outputs"]) == len(data["filters"])
            for o in t["outputs"]:
                assert set(o.keys()) >= {"filter", "exists", "mtime"}
                if o["exists"]:
                    saw_existing = True
                    # /tap/tile_metadata and /tap/thumbnail should resolve.
                    tap = f"{p['name']}/{t['display']}"
                    thumb = client.get(
                        f"/tap/thumbnail/mosaic/{mosaic_id}/{tap}/{o['filter']}"
                    )
                    assert thumb.status_code in (200, 404), (
                        f"thumbnail {tap}/{o['filter']}: {thumb.status_code}"
                    )
                    meta = client.get(
                        f"/tap/tile_metadata/mosaic/{mosaic_id}/{tap}/{o['filter']}"
                    )
                    assert meta.status_code in (200, 404), (
                        f"tile_metadata {tap}/{o['filter']}: {meta.status_code}"
                    )
    assert saw_existing, "no pipeline produced any output for the Orion mosaic"

    # Unknown mosaic id → 404.
    assert client.get("/api/pipeline_outputs/mosaic/does_not_exist").status_code == 404


@pytest.mark.exhaustive
def test_pipeline_outputs_exists_is_backed_by_trace_and_disk(client, with_orion_mosaic):
    """The core contract of the output_index → trace+disk migration.

    Every `exists` flag the dashboard endpoint reports MUST agree with an
    INDEPENDENT re-derivation: predict the tap's primary path straight from
    the cached pipeline trace (`tap_path_for`) and ask the filesystem. There
    is no stateful tap→path index in between, so:

      * `exists=True`  ⟺ a real file sits at the trace-predicted path, and
      * `exists=False` ⟺ no file is there (real absence, never a stale
        "pending" placeholder — the live-app bug this refactor removed).

    Because the Orion snapshot actually runs the graph, the merge/master
    deliverables are on disk; at least one tap must therefore surface as
    existing, proving the previously-"pending" taps now show.
    """
    import os
    from dependency_graph.trace_paths import tap_path_for

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))

    data = client.get(f"/api/pipeline_outputs/mosaic/{mosaic_id}").get_json()

    checked = 0
    existing = 0
    for p in data["pipelines"]:
        name = p["name"]
        for t in p["taps"]:
            display = t["display"]
            for o in t["outputs"]:
                # Independent path prediction from the cached trace — the
                # same source the endpoint uses, recomputed here so a stale
                # or divergent result would surface as a mismatch.
                predicted = tap_path_for(
                    ("mosaic", mosaic_id, name), display, o["filter"],
                )
                on_disk = predicted is not None and os.path.exists(predicted)
                assert o["exists"] == on_disk, (
                    f"{name}/{display}/{o['filter']}: endpoint exists="
                    f"{o['exists']} but trace+disk says {on_disk} "
                    f"(predicted path={predicted})"
                )
                checked += 1
                existing += int(o["exists"])

    assert checked > 0, "fixture produced no taps to check"
    assert existing > 0, (
        "no tap reported exists=True — deliverables that are on disk for "
        "the Orion mosaic are not surfacing (the bug this refactor fixes)"
    )


@pytest.mark.exhaustive
def test_pipeline_outputs_second_call_hits_cache(client, with_orion_mosaic):
    """Regression for the live-app failure mode: every call to
    /api/pipeline_outputs/mosaic/<id> taking ~50s instead of the
    second-call cache hit @node was supposed to give us.

    Hits the endpoint twice for the same mosaic. The second call MUST
    be at least 10x faster than the first — if it's not, the @node
    cache on trace_pipeline isn't taking, and we have a real bug.
    Prints both timings so we can see where wall-clock goes when it
    breaks.
    """
    import time
    from context import local_context as context

    # Other tests hit /api/pipeline_outputs first and warm the cached
    # `_traced_paths` @node entries — so a fresh "first call" here would
    # be a cache hit and defeat the test. Drop those entries (cache key is
    # `("_traced_paths", call_hash)`) so the first hit actually re-traces
    # and the second can show the cache win. There is no tap index anymore:
    # the endpoint derives paths purely from this cached trace + disk.
    for k in list((context.get_matching(("_traced_paths",)) or {}).keys()):
        if k in context:
            context.delete(k)

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    url = f"/api/pipeline_outputs/mosaic/{mosaic_id}"

    t0 = time.perf_counter()
    r1 = client.get(url)
    dt1 = time.perf_counter() - t0
    assert r1.status_code == 200
    print(f"[cache-test] first call:  {dt1*1000:8.1f} ms")

    t0 = time.perf_counter()
    r2 = client.get(url)
    dt2 = time.perf_counter() - t0
    assert r2.status_code == 200
    print(f"[cache-test] second call: {dt2*1000:8.1f} ms (speedup: {dt1/max(dt2, 1e-9):.1f}x)")

    # Sanity: responses should be structurally identical (same
    # pipelines listed). Catches "cache returns stale shape" bugs.
    assert r1.get_json() == r2.get_json()

    # The real assertion. Order of magnitude faster on the second hit.
    assert dt2 < dt1 / 10, (
        f"Second /api/pipeline_outputs call ({dt2*1000:.0f} ms) was not "
        f"≥10× faster than first ({dt1*1000:.0f} ms) — @node cache miss "
        f"on repeat, the live-app symptom."
    )


@pytest.mark.exhaustive
def test_metadata_route_trace_cache_actually_hits_on_second_call(client, with_orion_mosaic):
    """Diagnostic: prove (or disprove) that the trace `@node` cache is
    actually being hit on repeat requests, and identify which sub-step
    of the request path dominates wall-clock.

    The user-reported failure mode: first request 78 s, then raylet OOM.
    Earlier benchmark showed the trace itself was ~17 s; the endpoint
    is doing more than just the trace. This test times each piece
    separately so we can see where the 60+ extra seconds go.
    """
    import time
    from dependency_graph.trace_paths import (
        _resolve_pipeline_call, _traced_paths, traced_paths_for,
    )
    from hashing import hash_source_code
    from metadata.image_metadata import load_metadata
    import ray

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    stacks = client.get("/api/stacks").get_json()
    filter_ = next(
        (s["filter"] for s in stacks.values() if s.get("filter")),
        "L",
    )
    url = f"/api/metadata/mosaic/{mosaic_id}/PixInsight Downscaled/Merged/{filter_}"
    item = ("mosaic", mosaic_id, "PixInsight Downscaled")

    def time_it(label, fn):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        print(f"[diag] {label:40s} {dt*1000:8.1f} ms")
        return result, dt

    # Sub-step 1: resolve pipeline call (cheap pure Python).
    (fn, entity, config), _ = time_it("_resolve_pipeline_call",
        lambda: _resolve_pipeline_call(item))
    assert fn is not None

    # Sub-step 2: hash_source_code (cheap, memoized after first).
    _, _ = time_it("hash_source_code", lambda: hash_source_code(fn))
    pipeline_source_hash = hash_source_code(fn)

    # Sub-step 3: load_metadata.remote (the suspect — reads ALL
    # compute_image_metadata entries from diskcache, builds the dict,
    # stores via store_large, returns ContextRef).
    md_ref_1, t_lm1 = time_it("load_metadata.remote #1",
        lambda: ray.get(load_metadata.remote()))
    md_ref_2, t_lm2 = time_it("load_metadata.remote #2",
        lambda: ray.get(load_metadata.remote()))
    print(f"[diag] metadata_ref #1 content_hash = {md_ref_1.content_hash}")
    print(f"[diag] metadata_ref #2 content_hash = {md_ref_2.content_hash}")
    assert md_ref_1.content_hash == md_ref_2.content_hash, (
        f"load_metadata produced different content_hashes on back-to-back "
        f"calls — the trace cache key will change every request"
    )

    args = ("PixInsight Downscaled", entity, config, md_ref_1, pipeline_source_hash)

    # Sub-step 4: is_cached check (should be cheap diskcache lookup).
    cached_before, _ = time_it("is_cached BEFORE first trace",
        lambda: _traced_paths.is_cached(*args))
    print(f"[diag]   = {cached_before}")

    # Sub-step 5: traced_paths_for (the bulk — Ray submit + trace + cache write).
    paths, t_traced1 = time_it("traced_paths_for #1 (cold)",
        lambda: traced_paths_for(item))
    print(f"[diag]   returned {len(paths)} paths")

    cached_after, _ = time_it("is_cached AFTER first trace",
        lambda: _traced_paths.is_cached(*args))
    print(f"[diag]   = {cached_after}")
    assert cached_after, "trace cache not populated by first call"

    paths2, t_traced2 = time_it("traced_paths_for #2 (warm)",
        lambda: traced_paths_for(item))

    paths3, t_traced3 = time_it("traced_paths_for #3 (warm)",
        lambda: traced_paths_for(item))

    # Sub-step 6: end-to-end through Flask.
    _, t_get1 = time_it("client.GET (cold)", lambda: client.get(url))
    _, t_get2 = time_it("client.GET (warm)", lambda: client.get(url))
    _, t_get3 = time_it("client.GET (warm)", lambda: client.get(url))

    print(f"\n[diag] === Summary ===")
    print(f"[diag] load_metadata.remote:        {t_lm1*1000:.1f} ms (first), {t_lm2*1000:.1f} ms (second)")
    print(f"[diag] traced_paths_for cold:       {t_traced1*1000:.1f} ms")
    print(f"[diag] traced_paths_for warm:       {t_traced2*1000:.1f} ms, {t_traced3*1000:.1f} ms")
    print(f"[diag] /api/metadata cold:          {t_get1*1000:.1f} ms")
    print(f"[diag] /api/metadata warm:          {t_get2*1000:.1f} ms, {t_get3*1000:.1f} ms")
    print(f"[diag] load_metadata as % of warm request:"
          f" {100*t_lm1/t_get2:.1f}%")
    print(f"[diag] # compute_image_metadata cache entries: "
          f"{_count_compute_image_metadata_entries()}")

    # The actual claim under test. If caching is working, warm calls
    # should be at least 10× faster than cold.
    assert t_get2 < t_get1 / 10 or t_get2 < 0.5, (
        f"warm GET ({t_get2*1000:.0f}ms) not >10x faster than cold "
        f"({t_get1*1000:.0f}ms); cache isn't doing its job"
    )


def _count_compute_image_metadata_entries():
    """Count compute_image_metadata entries in the diskcache without
    fetching values."""
    from context import local_context
    n = 0
    for c in (local_context._cache, local_context._durable):
        for key in c.iterkeys():
            if isinstance(key, tuple) and len(key) > 0 and key[0] == "compute_image_metadata":
                n += 1
    return n


@pytest.mark.exhaustive
def test_metadata_route_master_path_is_passed_as_path_not_str(client, with_orion_mosaic):
    """Regression for the 2026-05-30 beartype error on a cache HIT:

        BeartypeCallHintParamViolation: ... parameter
        filename='/Volumes/.../ef4cde65...xisf' violates type hint
        <class 'pathlib._local.Path'>, as str ... not instance of <Path>

    `traced_paths_for` returns paths as `str` (cache-friendly), but
    `read_image_metadata` is beartype-annotated `Path`. The route must
    wrap the str before passing it down. This test forces the route to
    take the `os.path.exists -> read_image_metadata` branch by mocking
    os.path.exists to True for any master path the trace returns; if
    the str/Path conversion is missing, beartype raises and the route
    500s.
    """
    import os
    from unittest.mock import patch

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    stacks = client.get("/api/stacks").get_json()
    filter_ = next(
        (s["filter"] for s in stacks.values() if s.get("filter")),
        "L",
    )

    real_exists = os.path.exists
    def _exists_true_for_traced(path):
        # Pretend any traced stack_files path exists, so the route
        # actually invokes read_image_metadata.
        if isinstance(path, str) and "stack_files" in path:
            return True
        return real_exists(path)

    # We can't stub read_image_metadata to None because the route
    # accesses meta.plate_solution after; instead let it actually run
    # (it'll fail on the missing file with a different exception that
    # the route silently ignores). The point of this test is purely
    # that beartype doesn't reject the call signature.
    with patch("web_server.backend.metadata.os.path.exists",
               side_effect=_exists_true_for_traced):
        resp = client.get(
            f"/api/metadata/mosaic/{mosaic_id}/PixInsight Downscaled/Merged/{filter_}"
        )

    assert resp.status_code == 200, (
        f"got {resp.status_code}: {resp.data[:400]!r}"
    )


@pytest.mark.exhaustive
def test_stacks_endpoint_returns_raw_integration_s(client, with_orion_mosaic):
    """The blue (raw total) polygon in the dual exposure widget reads
    from `stack.raw_integration_s`. Verify the field is present and
    equals the sum of EXPOSURE across every subframe in the stack."""
    from context import local_context as ctx

    stacks = client.get("/api/stacks").get_json()
    assert stacks, "no stacks in /api/stacks response"

    ctx_stacks = ctx.stacks or {}
    checked = 0
    for stack_id, served in stacks.items():
        assert "raw_integration_s" in served, (
            f"stack {stack_id} missing raw_integration_s: {served!r}"
        )
        stack_obj = ctx_stacks[stack_id].get()
        expected = sum(
            m.header.get("EXPOSURE", 0)
            for s in stack_obj.contents.values()
            for m in s.metadata.values()
        )
        assert abs(served["raw_integration_s"] - expected) < 1e-3, (
            f"stack {stack_id} raw_integration_s={served['raw_integration_s']} "
            f"!= sum(EXPOSURE)={expected}"
        )
        checked += 1
    assert checked > 0, "no stacks were actually checked"


@pytest.mark.exhaustive
def test_metadata_route_with_pipeline_returns_200(client, with_orion_mosaic):
    """Regression for the 2026-05-30 production crash on
    /api/metadata/mosaic/.../<pipeline>/<display>/<filter>.

    The bug: `_get_polygons` called `stack_files.filename_fn(active_stack)`
    with only the stack, but post-@node-refactor that goes through
    `cache_key_narrow` → `_stack_files_narrow(args, kwargs)` which does
    `stack, config, *rest = args` — unpacks with 1 arg, ValueError, 500.

    Fix: look up the master path via `traced_path_for(cart_item,
    "stack_files", stack=...)` instead, which goes through the cached
    pipeline trace. This test exercises the same URL shape from the
    production traceback and asserts the route doesn't 500.
    """
    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))

    # Find a real filter so the URL is well-formed (the route's pipeline
    # / display / filter components otherwise hit divergent code paths).
    stacks = client.get("/api/stacks").get_json()
    filter_ = next(
        (s["filter"] for s in stacks.values() if s.get("filter")),
        "L",
    )

    # Same URL shape that crashed in production. The display name
    # ("Merged") doesn't need to correspond to a real tap on disk —
    # the route returns polygons either way; we just care that it
    # doesn't 500 inside _get_polygons.
    resp = client.get(
        f"/api/metadata/mosaic/{mosaic_id}/PixInsight Downscaled/Merged/{filter_}"
    )
    assert resp.status_code == 200, (
        f"got {resp.status_code}: {resp.data[:300]!r}"
    )
    data = resp.get_json()
    assert "objects" in data, f"missing 'objects' in response: {data}"


@pytest.mark.exhaustive
def test_lazy_thumbnails_regenerate_on_miss(client, with_orion_mosaic):
    """Thumbnails are no longer eagerly generated; the web server must
    rebuild them on first request. Proves all three regen shapes:
    file-backed (stack master/subframe) and compute-backed (the
    flat-correction Run thumbnails, which have no on-disk source)."""
    from context import local_context as ctx

    stacks = ctx.stacks or {}
    assert len(stacks) > 0

    saw_stack = saw_run = False
    for sid, stack_ref in stacks.items():
        stack = stack_ref.get()

        # File-backed: stack subframe/master, rebuilt from disk on miss.
        r = client.get(f"/thumbnail/stack/{sid}")
        assert r.status_code == 200, f"/thumbnail/stack/{sid}: {r.status_code}"
        assert r.mimetype == "image/png" and len(r.data) > 0
        saw_stack = True

        # Compute-backed: the uncorrected Run thumbnail has NO on-disk
        # source — it can only come from invoking the producer on miss.
        for sub_id in stack.contents:
            r = client.get(f"/thumbnail/run/{sid}/{sub_id}")
            assert r.status_code == 200, (
                f"/thumbnail/run/{sid}/{sub_id}: {r.status_code} "
                f"(compute-on-miss regen failed)"
            )
            assert r.mimetype == "image/png" and len(r.data) > 0
            saw_run = True
            break  # one substack is enough to exercise the path
        if saw_stack and saw_run:
            break

    assert saw_stack and saw_run, "did not exercise both regen shapes"
