"""The shard redirect must carry the query string.

`stateful_route` 307s a request to a sharded backend chosen by
`sharding_key`; a redirect built from `request.path` alone silently drops
the caller's query parameters, so the shard answers a DIFFERENT request.
Measured live 2026-09-11, when tap reads carried a resolution parameter
there: the SAME shard returned 200 with it and 404 without, and every
stack- and target-level view was blank while mosaic views (which needed no
parameter) kept working — exactly what hid the bug.
"""
from flask import Flask

from web_server.backend.router import (
    BACKEND_PORTS, ROUTER_PORT, pick_backend_port, stateful_route,
)


def _app():
    app = Flask(__name__)

    @stateful_route(app.route("/tap/<object_id>"), sharding_key="object_id")
    def tap(object_id: str):
        return "served"

    return app


def test_shard_redirect_preserves_the_query_string():
    app = _app()
    qs = "ancestors=mosaic:75a1da6463e118e6,target:bb7f13f017a367d1"
    with app.test_client() as c:
        r = c.get(f"/tap/6d5b69676e23fdd9?{qs}", headers={"Host": f"h:{ROUTER_PORT}"})
    assert r.status_code == 307
    assert qs in r.headers["Location"], (
        f"shard redirect dropped the query string: {r.headers['Location']} — "
        "the shard cannot resolve a child stack's tap without `ancestors`"
    )


def test_shard_redirect_targets_the_sharded_port():
    app = _app()
    with app.test_client() as c:
        r = c.get("/tap/6d5b69676e23fdd9", headers={"Host": f"h:{ROUTER_PORT}"})
    port = pick_backend_port("6d5b69676e23fdd9")
    assert port in BACKEND_PORTS
    assert f":{port}/" in r.headers["Location"]


def test_no_query_string_leaves_no_trailing_question_mark():
    """`request.full_path` would append a bare '?' — keep clean URLs clean."""
    app = _app()
    with app.test_client() as c:
        r = c.get("/tap/6d5b69676e23fdd9", headers={"Host": f"h:{ROUTER_PORT}"})
    assert not r.headers["Location"].endswith("?")
