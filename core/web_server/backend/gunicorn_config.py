def post_fork(server, worker):
    # Delay import until after fork + gevent monkey-patching, so that
    # queue.Queue is the gevent-patched cooperative version.
    from web_server.backend.announcer import start_event_loop
    start_event_loop()
    # Cluster CPU% sampler for the Ray-status chip's sparkline.
    # Runs in this same gunicorn worker as a gevent greenlet — see
    # `ray_metrics.py` for why it lives here rather than on the
    # SchedulerActor.
    from web_server.backend.ray_metrics import start_polling
    start_polling()
