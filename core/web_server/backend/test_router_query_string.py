"""The shard redirect must carry the query string.

A tap read addressed at a bare leaf (a child stack) has no tap snapshot of
its own — only the cart item ancestor has one — so the frontend passes
`?ancestors=type:id,...` and the backend filters the ancestor's snapshot down
to the leaf (`_route.parse_ancestors`). `stateful_route` 307s the request to
a sharded backend chosen by `sharding_key`, and if that redirect is built
from `request.path` the param is silently dropped: the shard sees a bare leaf,
cannot resolve it, and 404s.

Measured live 2026-09-11 on stack 6d5b69676e23fdd9 under mosaic
75a1da6463e118e6 — the SAME shard returned 200 with `ancestors` and 404
without. Every stack- and target-level tap view was blank; mosaic views kept
working because a mosaic IS the cart item and needs no ancestors, which is
exactly what hid the bug.
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
