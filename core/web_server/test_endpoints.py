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
            # RAGGED: a tap carries ONLY the filters that route through it in the
            # trace, not the global column union — so `outputs` is a NON-EMPTY,
            # duplicate-free SUBSET of `data["filters"]` in the same canonical
            # order, never padded with phantom blanks. (A composite tap like
            # Hubble is thus a one-image row, not a blank-filled grid.)
            tap_filters = [o["filter"] for o in t["outputs"]]
            assert tap_filters, f"{p['name']}/{t['display']} has no outputs"
            assert len(tap_filters) == len(set(tap_filters)), (
                f"{p['name']}/{t['display']} duplicate filters {tap_filters}")
            assert set(tap_filters) <= set(data["filters"]), (
                f"{p['name']}/{t['display']} filters {tap_filters} ⊄ "
                f"{data['filters']}")
            assert tap_filters == [f for f in data["filters"] if f in set(tap_filters)], (
                f"{p['name']}/{t['display']} filters {tap_filters} not in "
                f"canonical order")
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
    the build-time filing (`catalog.predict_path`) and ask the filesystem.
    There is no other tap→path index in between, so:

      * `exists=True`  ⟺ a real file sits at the trace-predicted path, and
      * `exists=False` ⟺ no file is there (real absence, never a stale
        "pending" placeholder — the live-app bug this refactor removed).

    Because the Orion snapshot actually runs the graph, the merge/master
    deliverables are on disk; at least one tap must therefore surface as
    existing, proving the previously-"pending" taps now show.
    """
    import os
    from catalog import Product, predict_path

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
                predicted = predict_path(
                    Product("mosaic", mosaic_id, name, display, o["filter"]))
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
def test_pipeline_outputs_reads_snapshot_not_trace(client, with_orion_mosaic):
    """Regression for the live-app failure mode: /api/pipeline_outputs taking
    tens of seconds (or 500ing on the Ray-free web worker) because it traced on
    demand. It now reads the build-time tap snapshot straight from the diskcache
    — no trace, no Ray — so it's fast on EVERY call. There is no on-demand trace
    cache to clear: a snapshot miss on a read-only caller returns `[]`, it never
    traces."""
    import time

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    url = f"/api/pipeline_outputs/mosaic/{mosaic_id}"

    t0 = time.perf_counter()
    r = client.get(url)
    dt = time.perf_counter() - t0
    assert r.status_code == 200
    print(f"[snapshot] pipeline_outputs: {dt*1000:.1f} ms (trace cache cleared)")
    # A snapshot read is sub-second even cold; an on-demand trace would be many
    # seconds. Generous bound to stay robust on a loaded box.
    assert dt < 5.0, (
        f"/api/pipeline_outputs took {dt*1000:.0f} ms with the on-demand trace "
        f"cache cleared — it must read the build-time snapshot, not re-trace."
    )


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
    rebuild them on first request. Proves both regen shapes:
    file-backed (stack master/subframe) and compute-backed (the Run
    calibration previews, which have no on-disk source)."""
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

        # Compute-backed: a Run calibration preview has NO on-disk source
        # — it can only come from invoking the producer on miss. Both the
        # resolved view and the explicit `flat=none` view take that path.
        for sub_id in stack.contents:
            for query in ("", "?flat=none"):
                url = f"/thumbnail/run/{sid}/{sub_id}{query}"
                r = client.get(url)
                assert r.status_code == 200, (
                    f"{url}: {r.status_code} (compute-on-miss regen failed)"
                )
                assert r.mimetype == "image/png" and len(r.data) > 0
            saw_run = True
            break  # one substack is enough to exercise the path
        if saw_stack and saw_run:
            break

    assert saw_stack and saw_run, "did not exercise both regen shapes"


def test_flats_route_offers_star_flats_first(client):
    """The manual flat picker: the group's star flat(s) leftmost, then the
    explicit masters, all restricted to this sensor, filter and optical
    train — the same candidate set `resolve_calibration` considers."""
    import datetime
    from frozendict import frozendict
    from context import context
    from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
    from stacks.run import RunsDict
    from stacks.star_flat import StarFlatStack
    from stacks.types import StackHash
    from units import Celsius, Seconds

    def _flat(cls, sid, **kw):
        fields = dict(content_hash=sid, image_type="flat", camera="cam", filter="H", mean_temperature=Celsius(-10.0),
                      exposure=Seconds(1.0), integration=Seconds(0.0), width_pixels=1080, height_pixels=720,
                      stable_hash=sid, oldest_date=datetime.datetime(2026, 1, 1), newest_date=datetime.datetime(2026, 1, 2),
                      state=frozendict(), contents=RunsDict(), camera_id_hash=None)
        fields.update(kw)
        return cls(**fields)

    context.calibration_stacks = CalibrationStacksDict({
        StackHash("explicit"): _flat(CalibrationStack, "explicit"),                      # no FOCALLEN: wildcard
        StackHash("star350"): _flat(StarFlatStack, "star350", focal_length=350.0, member_files=("/a", "/b")),
        StackHash("star1230"): _flat(StarFlatStack, "star1230", focal_length=1230.0, member_files=("/c", "/d")),
        StackHash("starS"): _flat(StarFlatStack, "starS", filter="S", focal_length=350.0, member_files=("/e", "/f")),
        StackHash("other"): _flat(CalibrationStack, "other", width_pixels=6248, height_pixels=4176),
    })
    assert client.get("/api/flats/1080/720/H?focal_length=351").get_json() == ["star350", "explicit"]
    assert client.get("/api/flats/1080/720/H").get_json() == ["star350", "star1230", "explicit"]
    assert client.get("/api/flats/1080/720/S?focal_length=350").get_json() == ["starS"]
    assert client.get("/api/flats/1080/720/O").get_json() == []


@pytest.mark.exhaustive
def test_run_preview_url_names_every_calibration_input(
    client, with_orion_mosaic, monkeypatch,
):
    """`/thumbnail/run/<stack>/<run>` takes its calibration by NAME.

    The contract, which the positional URL it replaced could not express:

    * an omitted kind falls back to what `resolve_calibration` picks for
      this run — NOT to nothing. The old route read the UNRESOLVED stack,
      whose run dependencies are always empty, so every flat-correction
      preview divided a frame still carrying its bias+dark pedestal and
      came out washed out (2026-09-08);
    * `<kind>=none` is an explicit "do not apply this kind", which is a
      different statement from omitting it;
    * an id that is not a calibration stack is a 404, never a silent
      fallback to some other frame.
    """
    import flask
    import web_server.backend.thumbnails as thumbnails
    from context import local_context as ctx
    from decisions.resolve import resolve_calibration

    sid, stack_ref = next(iter((ctx.stacks or {}).items()))
    stack = stack_ref.get()
    resolved = resolve_calibration(
        stack, ctx.calibration_stacks, ctx.settings.decision_overrides,
    )
    run_id, run = next(iter(resolved.contents.items()))
    resolved_ids = {
        kind: getattr(run.dependencies.get(kind), "stack_id", None)
        for kind in ("flat", "dark", "bias")
    }

    captured = {}

    def fake_get_thumbnail(entity, *extra_args):
        captured["args"] = extra_args
        return flask.Response(b"", mimetype="image/png")

    monkeypatch.setattr(thumbnails, "get_thumbnail", fake_get_thumbnail)

    # Nothing named: every kind comes from the resolved run.
    assert client.get(f"/thumbnail/run/{sid}/{run_id}").status_code == 200
    _, flat, dark, bias = captured["args"]
    assert (flat, dark, bias) == (
        resolved_ids["flat"], resolved_ids["dark"], resolved_ids["bias"],
    )

    # `flat=none` drops ONLY the flat — the dark still comes from the run,
    # which is exactly what the washed-out preview was missing.
    assert client.get(
        f"/thumbnail/run/{sid}/{run_id}?flat=none"
    ).status_code == 200
    _, flat, dark, bias = captured["args"]
    assert flat is None
    assert (dark, bias) == (resolved_ids["dark"], resolved_ids["bias"])

    # A named calibration stack wins over the resolved one.
    other = next(
        (c for c in ctx.calibration_stacks
         if c != resolved_ids["flat"]),
        None,
    )
    if other is not None:
        assert client.get(
            f"/thumbnail/run/{sid}/{run_id}?flat={other}"
        ).status_code == 200
        assert captured["args"][1] == other

    # An unknown id is a 404, not a silent substitution.
    assert client.get(
        f"/thumbnail/run/{sid}/{run_id}?flat=nosuchstack"
    ).status_code == 404
