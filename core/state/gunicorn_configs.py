from state.gunicorn_control import GunicornConfig


def guincorn_configs() -> list[GunicornConfig]:
    return {
        "static_server": GunicornConfig(
            name="static_server",
            config_file="web_server/backend/gunicorn_app_config.py",
            module="web_server.backend.app:app",
            worker_type="sync",
            num_servers=1,
            start_port=8000,
            # FOUR worker PROCESSES, not the gunicorn default of one.
            # `sync` + the shared `--threads 4` silently promotes to a
            # single gthread process, and threads are GIL-bound: one
            # CPU-heavy request (a trace rebuild after a cart change, a
            # first-view pyramid build) starves every other request and the
            # whole UI reads as hung — three times on 2026-08-21/22, each
            # cleared only by killing the worker. gthread also heartbeats
            # from its own thread, so `timeout` never trips on a long
            # request. Processes are the only thing that actually isolates.
            num_workers=4,
            timeout=300,
        ),
        "stream_server": GunicornConfig(
            name="stream_server",
            config_file="web_server/backend/gunicorn_config.py",
            module="web_server.backend.stream:app",
            worker_type="gevent",
            num_servers=1,
            start_port=8001,
            num_workers=1,
            timeout=0,
        ),
        "tile_server": GunicornConfig(
            name="tile_server",
            config_file="web_server/backend/gunicorn_app_config.py",
            module="web_server.backend.app:app",
            worker_type="sync",
            num_servers=4,
            start_port=8002,
            # Four processes per tile server (16 in all): one process is one
            # GIL, and four servers of one process each plateaued at ~250
            # tiles/s no matter how many requests were in flight (2026-10-01).
            num_workers=4,
            timeout=300,
        ),
    }
