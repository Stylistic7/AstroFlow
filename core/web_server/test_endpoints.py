"""End-to-end tests: pipeline → Flask API → JSON responses.

These tests run the full compute graph on test data, then verify that
the Flask API endpoints return correct, complete JSON responses. This
is the closest thing to testing the UI without rendering React.
"""
import datetime
import json
from pathlib import Path

import pytest

from testing import copy_testdata_frames
from context import context
from dependency_graph import run_graph
from compute_graph.compute_graph import build_compute_graph
from mosaics.orion_mosaic import add_orion_mosaic_images


@pytest.fixture
def client(test_context):
    """Flask test client backed by the test context's Ray cluster."""
    from web_server.backend.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestStacksEndpoint:

    @pytest.mark.heavy
    def test_stacks_returns_json(self, client, test_context):
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        resp = client.get("/stacks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert len(data) > 0

    @pytest.mark.heavy
    def test_stack_has_required_fields(self, client, test_context):
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        data = client.get("/stacks").get_json()
        stack = next(iter(data.values()))

        for field in ["name", "center", "image_scale", "rotation_angle",
                      "camera", "filter", "exposure", "integration",
                      "width_pixels", "height_pixels", "stable_hash"]:
            assert field in stack, f"Missing field: {field}"


class TestMosaicEndpoints:

    @pytest.mark.heavy
    def test_mosaic_pipeline_to_api(self, client, test_context):
        """Full pipeline: ingest → identify → API response."""
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        # /mosaics endpoint
        mosaics = client.get("/mosaics").get_json()
        assert isinstance(mosaics, dict)
        assert len(mosaics) == 1

        mosaic_id = next(iter(mosaics.keys()))
        mosaic = mosaics[mosaic_id]

        assert "name" in mosaic
        assert "target_ids" in mosaic
        assert "strips" in mosaic
        assert len(mosaic["target_ids"]) == 4

    @pytest.mark.heavy
    def test_targets_endpoint(self, client, test_context):
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        targets = client.get("/targets").get_json()
        assert isinstance(targets, dict)
        assert len(targets) == 4

        target = next(iter(targets.values()))
        assert "name" in target
        assert "center" in target
        assert "stack_ids" in target
        assert len(target["center"]) == 2

    @pytest.mark.heavy
    def test_calibration_stacks_endpoint(self, client, test_context):
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        cal_stacks = client.get("/calibration_stacks").get_json()
        assert isinstance(cal_stacks, dict)
        # Orion test data includes darks and flats
        assert len(cal_stacks) > 0

    @pytest.mark.heavy
    def test_metadata_endpoint_before_processing(self, client, test_context):
        """Metadata returns 404 for stacks not yet processed."""
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        stacks = client.get("/stacks").get_json()
        stack_id = next(iter(stacks.keys()))

        # No master images exist yet (no shopping cart), so metadata returns 404
        resp = client.get(f"/metadata/stack/{stack_id}")
        assert resp.status_code == 404

    @pytest.mark.heavy
    def test_thumbnail_endpoint(self, client, test_context):
        add_orion_mosaic_images()
        run_graph(build_compute_graph())

        stacks = client.get("/stacks").get_json()
        stack_id = next(iter(stacks.keys()))

        resp = client.get(f"/thumbnail/stack/{stack_id}")
        # Should return an image or 404 if thumbnail not yet generated
        assert resp.status_code in [200, 404]

    @pytest.mark.heavy
    def test_shopping_cart(self, client, test_context):
        resp = client.get("/get_shopping_cart")
        assert resp.status_code == 200
        cart = resp.get_json()
        assert isinstance(cart, list)
        assert len(cart) == 0

    @pytest.mark.heavy
    def test_constellation_polylines(self, client, test_context):
        resp = client.get("/constellation_polylines")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.heavy
    def test_compute_graph_status(self, client, test_context):
        resp = client.get("/compute_graph_status")
        assert resp.status_code == 200

    @pytest.mark.heavy
    def test_ray_status(self, client, test_context):
        resp = client.get("/ray_status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data or isinstance(data, dict)

    @pytest.mark.heavy
    def test_performance_endpoint(self, client, test_context):
        resp = client.get("/api/performance")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "functions" in data


class TestMosaicWithShoppingCart:
    """Test the full mosaic pipeline including stacking and post-processing."""

    @pytest.mark.heavy
    def test_full_mosaic_produces_correct_api_response(self, client, test_context):
        add_orion_mosaic_images()

        # First pass: subframe previews
        run_graph(build_compute_graph())

        mosaics = client.get("/mosaics").get_json()
        assert len(mosaics) == 1
        mosaic_id = next(iter(mosaics.keys()))

        # Add mosaic to shopping cart
        resp = client.get(f"/add_to_shopping_cart/mosaic/{mosaic_id}")
        assert resp.status_code == 204

        cart = client.get("/get_shopping_cart").get_json()
        assert len(cart) == 1

        # Second pass: stacking and master processing
        run_graph(build_compute_graph())

        # Verify the mosaic still appears correctly
        mosaics = client.get("/mosaics").get_json()
        assert len(mosaics) == 1
        mosaic = mosaics[mosaic_id]
        assert len(mosaic["target_ids"]) == 4
        assert len(mosaic["strips"]) > 0

        # Verify stacks have all expected fields
        stacks = client.get("/stacks").get_json()
        for stack in stacks.values():
            assert "exposure" in stack
            assert "integration" in stack
            assert "filenames" in stack

        # Verify targets reference valid stacks
        targets = client.get("/targets").get_json()
        for target in targets.values():
            for sid in target["stack_ids"]:
                assert sid in stacks, f"Target references nonexistent stack {sid}"
