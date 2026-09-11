from flask import request, redirect
from flask_cors import cross_origin
import functools
import inspect
import os
import xxhash

BACKEND_PORTS = [8002, 8003, 8004, 8005]
ROUTER_PORT = "8000"


def pick_backend_port(key: str) -> int:
    h = xxhash.xxh32(key).intdigest()
    return BACKEND_PORTS[h % len(BACKEND_PORTS)]


def stateful_route(
    route_decorator,
    sharding_key: str = None,
    fixed_port: int = None,
    inject_name: str = "backend_port",
    cors_origins: str = "*",
):
    def outer(func):
        sig = inspect.signature(func)
        accepts_port = inject_name in sig.parameters

        @functools.wraps(func)
        # `expose_headers`: this per-route @cross_origin takes precedence over
        # the app-global CORS for stateful routes, so the global expose_headers
        # doesn't reach them. The tile route (a stateful route, fetched
        # cross-origin from a sharded port) returns tile width/height/BAND COUNT
        # in response headers; without exposing them the browser hides them from
        # JS → `X-Tile-Bands` reads null → colour tiles decode as mono (r32float)
        # → the image renders all red. Exposing non-present headers on other
        # stateful routes is a harmless no-op.
        @cross_origin(origins=cors_origins,
                      expose_headers=["X-Tile-Width", "X-Tile-Height", "X-Tile-Bands"])
        def wrapper(*args, **kwargs):
            if fixed_port is not None:
                port = fixed_port
            else:
                key = kwargs.get(sharding_key)
                if not key:
                    return f"Missing sharding key: {sharding_key}", 400
                port = pick_backend_port(key)

            host_port = request.host.split(":")[-1]

            if host_port == ROUTER_PORT:
                # The redirect host must be one the BROWSER can reach. Behind
                # the CRA dev proxy (changeOrigin) `request.host` is
                # "localhost:8000", so a remote browser would be redirected to
                # its own loopback and tiles break. Prefer an explicitly
                # advertised host, then the browser's forwarded host, then
                # the raw Host header (correct in the local/no-proxy case).
                host = (
                    os.environ.get("ASTROFLOW_ADVERTISED_HOST")
                    or request.headers.get("X-Forwarded-Host", request.host)
                ).split(":")[0]
                # CARRY THE QUERY STRING. `request.path` drops it, and the
                # shard then sees a bare leaf with no `ancestors=` — which is
                # the ONE thing that lets it resolve a child stack's tap out
                # of its cart-item ancestor's snapshot (see
                # `_route.parse_ancestors`). Measured 2026-09-11: the same
                # shard returns 200 with the param and 404 without it, so
                # EVERY stack- and target-level tap view 404'd while mosaic
                # views — the cart items, which need no ancestors — worked,
                # hiding it.
                query = request.query_string.decode()
                return redirect(
                    f"http://{host}:{port}{request.path}"
                    + (f"?{query}" if query else ""),
                    code=307,
                )

            if accepts_port:
                kwargs[inject_name] = port

            return func(*args, **kwargs)

        return route_decorator(wrapper)

    return outer
