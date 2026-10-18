"""One web server process on one port, for Windows.

gunicorn is POSIX-only (it forks), so on Windows `state.gunicorn_control`
runs each server as this module under waitress: one process of `--threads`
threads, without the isolation gunicorn's worker processes give against a
CPU-bound request. The app is named `module:attribute`, as gunicorn names it:

    python -m web_server.backend.serve web_server.backend.app:app --port 8000 --threads 16

`--on-start module:function` runs once in this process before it serves —
the stream server's announcer loop, which gunicorn's config starts in
`post_fork`.
"""
import argparse
import importlib

from waitress import serve


def _load(spec: str):
    module, attribute = spec.split(":")
    return getattr(importlib.import_module(module), attribute)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("app")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--on-start", default="")
    args = parser.parse_args()
    app = _load(args.app)
    if args.on_start:
        _load(args.on_start)()
    # channel_timeout drops a connection that makes no progress for that long:
    # the stall that wedged gunicorn's threads (gunicorn_app_config.py).
    serve(app, host="0.0.0.0", port=args.port, threads=args.threads, channel_timeout=120)


if __name__ == "__main__":
    main()
