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
