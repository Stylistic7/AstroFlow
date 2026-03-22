def post_fork(server, worker):
    # Delay import until after fork + gevent monkey-patching, so that
    # queue.Queue is the gevent-patched cooperative version.
    from web_server.backend.announcer import start_event_loop
    start_event_loop()
