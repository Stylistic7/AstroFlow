# The stream server's own settings (state/gunicorn_configs.py).

# Its requests are event streams that never end, so a graceful stop, which
# waits for requests to finish, always waited out gunicorn's default 30 s
# before killing the worker: Quit lingered, and a stop that gave up sooner left
# the server running (2026-10-17). Nothing is lost by cutting them: browsers
# reconnect an event stream by themselves.
graceful_timeout = 2


def post_fork(server, worker):
    # Delay import until after fork + gevent monkey-patching, so that
    # queue.Queue is the gevent-patched cooperative version.
    from web_server.backend.announcer import start_event_loop
    start_event_loop()
