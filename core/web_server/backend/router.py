from flask import request, redirect
from flask_cors import cross_origin
import functools
import inspect
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
        @cross_origin(origins=cors_origins)
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
                return redirect(
                    f"http://{request.host.split(':')[0]}:{port}{request.path}",
                    code=307,
                )

            if accepts_port:
                kwargs[inject_name] = port

            print(
                f"### Calling stateful_route function {func.__name__} on port {host_port}"
            )
            return func(*args, **kwargs)

        return route_decorator(wrapper)

    return outer
