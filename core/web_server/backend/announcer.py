import time
import os

from context import queue
from web_server.backend.message_announcer import MessageAnnouncer

QUEUE_NAME = "sse_announcer_queue"

announcer = MessageAnnouncer()


def announce(event, data=None):
    queue.put(QUEUE_NAME, (event, data))


def post_fork(server, worker):
    start_event_loop()


def start_event_loop():
    import gevent

    gevent.spawn(event_loop)


def event_loop():
    queue.clear(QUEUE_NAME)

    try:
        while True:
            announcement = queue.get(QUEUE_NAME)
            if announcement is None:
                time.sleep(0.1)
                continue

            announcer.announce(*announcement)

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
