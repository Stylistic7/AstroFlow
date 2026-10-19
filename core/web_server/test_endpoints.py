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
    assert "panel_target_ids" in mosaic
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


def test_orion_mosaic_finishes_endpoint(client, with_orion_mosaic):
    """The per-pipeline finish list the pills and the dashboard read."""
    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))

    resp = client.get(f"/api/finishes/mosaic/{mosaic_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) == {"pipelines", "default"}
    pipelines = data["pipelines"]
    assert len(pipelines) > 0
    saw_servable = False
    for p in pipelines:
        assert set(p.keys()) == {"name", "finishes"}
        names = [f["name"] for f in p["finishes"]]
        assert len(names) == len(set(names)), f"{p['name']} lists a finish twice"
        for f in p["finishes"]:
            assert set(f.keys()) == {"name", "role", "status", "mtime", "inputs"}
            assert f["status"] in ("live", "out_of_date", "absent")
            assert f["inputs"] and all(set(i) == {"tap", "filter", "status"} for i in f["inputs"])
            if f["status"] == "absent":
                continue
            saw_servable = True
            view = f"/mosaic/{mosaic_id}/{p['name']}/{f['name']}"
            assert client.get(f"/tap/thumbnail{view}").status_code in (200, 404)
            meta = client.get(f"/tap/tile_metadata{view}", follow_redirects=True)
            assert meta.status_code == 200, f"tile_metadata {view}: {meta.status_code}"
            m = meta.get_json()
            assert set(m) == {"name", "role", "status", "finish", "inputs", "tilePorts"}
            assert m["finish"]["name"] == f["name"]
            assert m["inputs"][0]["bound"] and m["inputs"][0]["bandRoles"]
            assert client.get(f"/api/metadata{view}").status_code == 200
    assert saw_servable, "no pipeline produced a servable finish for the Orion mosaic"
    assert data["default"] is None or set(data["default"]) == {"pipeline", "finish"}

    # Unknown mosaic id → 404.
    assert client.get("/api/finishes/mosaic/does_not_exist").status_code == 404

@pytest.mark.exhaustive
def test_finish_status_is_backed_by_the_tap_resolver(client, with_orion_mosaic):
    """Every input status the finishes endpoint reports MUST agree with an
    INDEPENDENT re-derivation through the catalog's product resolver, and
    at least one finish must be servable for the built Orion mosaic."""
    from catalog import Product, resolve_product

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    data = client.get(f"/api/finishes/mosaic/{mosaic_id}").get_json()

    checked = servable = 0
    for p in data["pipelines"]:
        for f in p["finishes"]:
            for i in f["inputs"]:
                if i["filter"] is None:
                    assert i["status"] == "absent"
                    continue
                path, status = resolve_product(
                    Product("mosaic", mosaic_id, p["name"], i["tap"], i["filter"]))
                assert i["status"] == status, (
                    f"{p['name']}/{f['name']}/{i['tap']}: endpoint {i['status']} "
                    f"but the resolver says {status} ({path})")
                checked += 1
            servable += int(f["status"] != "absent")
    assert checked > 0, "fixture produced no finish inputs to check"
    assert servable > 0

@pytest.mark.exhaustive
def test_finishes_read_snapshot_not_trace(client, with_orion_mosaic):
    """Regression for the live-app failure mode: the view listing taking tens
    of seconds (or 500ing on the Ray-free web worker) because it traced on
    demand. It reads the build-time tap snapshot straight from the diskcache
    — no trace, no Ray — so it's fast on EVERY call."""
    import time

    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    t0 = time.perf_counter()
    r = client.get(f"/api/finishes/mosaic/{mosaic_id}")
    dt = time.perf_counter() - t0
    assert r.status_code == 200
    print(f"[snapshot] finishes: {dt*1000:.1f} ms")
    assert dt < 5.0, (
        f"/api/finishes took {dt*1000:.0f} ms — it must read the build-time "
        f"snapshot, not re-trace.")

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


def test_metadata_route_selects_stacks_by_finish_role(client, monkeypatch):
    """A channel finish's primary input is a band-join tap bound as "RGB"; the
    polygons must be selected by the finish's ROLE (the stack filter), else no
    target has a matching stack and every master footprint -- and the depth
    overlay drawn on them -- disappears (2026-09-30). Composites pass None."""
    import catalog
    from types import SimpleNamespace
    from catalog.finishes import BoundFinish, BoundInput
    from web_server.backend import metadata as md

    def bound(role):
        finish = SimpleNamespace(name=role or "Hubble", role=role)
        inp = BoundInput(index=0, tap="Linear SHO", filter="RGB", path=None,
                         status="live", optional=False)
        return BoundFinish("mosaic", "m0", "Pipe", finish, (inp,), "live")

    seen = []
    monkeypatch.setattr(md, "_resolve_object", lambda ot, oid, flt: seen.append(flt) or ("obj", flt))
    monkeypatch.setattr(md, "_get_polygons", lambda pipeline, viewbox, *args: [])
    for role in ("S", None):
        monkeypatch.setattr(catalog, "resolve_finish", lambda *a, _r=role: bound(_r))
        resp = client.get(f"/api/metadata/mosaic/m0/Pipe/{role or 'Hubble'}")
        assert resp.status_code == 200, resp.data[:200]
    assert seen == ["S", None], seen


@pytest.mark.exhaustive
def test_metadata_route_with_finish_returns_polygons(client, with_orion_mosaic):
    """Regression for the 2026-05-30 production crash on the metadata route
    (`_get_polygons` unpacked `stack_files`' narrowed args wrongly and 500ed).
    The route is finish-keyed now: the polygons of the finish's primary input,
    at the filter that input binds."""
    mosaics = client.get("/api/mosaics").get_json()
    mosaic_id = next(iter(mosaics.keys()))
    data = client.get(f"/api/finishes/mosaic/{mosaic_id}").get_json()
    views = [(p["name"], f["name"]) for p in data["pipelines"] for f in p["finishes"]]
    assert views, "the Orion mosaic offers no finish"
    for pipeline, finish in views[:3]:
        resp = client.get(f"/api/metadata/mosaic/{mosaic_id}/{pipeline}/{finish}")
        assert resp.status_code == 200, f"got {resp.status_code}: {resp.data[:300]!r}"
        assert set(resp.get_json()) == {"objects"}
    assert client.get(f"/api/metadata/mosaic/{mosaic_id}/{views[0][0]}/NoSuchFinish").status_code == 404


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
    """The manual flat picker: the masters the default is chosen among
    first, then the group's star flat(s), all restricted to this sensor,
    filter and optical train — the same candidate set `resolve_calibration`
    considers."""
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
    assert client.get("/api/flats/1080/720/H?focal_length=351").get_json() == ["explicit", "star350"]
    assert client.get("/api/flats/1080/720/H").get_json() == ["explicit", "star350", "star1230"]
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
