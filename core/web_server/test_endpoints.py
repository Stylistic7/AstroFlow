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


@pytest.mark.heavy
def test_orion_mosaic_endpoints(client, with_orion_mosaic):
    """Verify all API endpoints return correct data for a 4-pane Orion mosaic."""
    # /stacks
    stacks = client.get("/stacks").get_json()
    assert isinstance(stacks, dict)
    assert len(stacks) > 0
    stack = next(iter(stacks.values()))
    for field in ["name", "center", "image_scale", "rotation_angle",
                  "camera", "filter", "exposure", "integration",
                  "width_pixels", "height_pixels", "stable_hash"]:
        assert field in stack, f"Missing field: {field}"

    # /targets
    targets = client.get("/targets").get_json()
    assert isinstance(targets, dict)
    assert len(targets) == 4
    target = next(iter(targets.values()))
    assert "name" in target
    assert "center" in target
    assert "stack_ids" in target
    assert len(target["center"]) == 2

    # /mosaics
    mosaics = client.get("/mosaics").get_json()
    assert isinstance(mosaics, dict)
    assert len(mosaics) == 1
    mosaic_id = next(iter(mosaics.keys()))
    mosaic = mosaics[mosaic_id]
    assert "name" in mosaic
    assert "target_ids" in mosaic
    assert "strips" in mosaic
    assert len(mosaic["target_ids"]) == 4

    # /calibration_stacks
    cal_stacks = client.get("/calibration_stacks").get_json()
    assert isinstance(cal_stacks, dict)
    assert len(cal_stacks) > 0

    # /metadata — no master images (no shopping cart), returns 404
    stack_id = next(iter(stacks.keys()))
    resp = client.get(f"/metadata/stack/{stack_id}")
    assert resp.status_code == 404

    # /thumbnail
    resp = client.get(f"/thumbnail/stack/{stack_id}")
    assert resp.status_code in [200, 404]

    # /get_shopping_cart
    cart = client.get("/get_shopping_cart").get_json()
    assert isinstance(cart, list)

    # /compute_graph_status
    assert client.get("/compute_graph_status").status_code == 200

    # /ray_status
    data = client.get("/ray_status").get_json()
    assert isinstance(data, dict)

    # /api/performance
    data = client.get("/api/performance").get_json()
    assert "functions" in data

    # Verify targets reference valid stacks
    for target in targets.values():
        for sid in target["stack_ids"]:
            assert sid in stacks, f"Target references nonexistent stack {sid}"
