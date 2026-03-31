import os
import time
import flask
from flask import Response
from queue import Empty
from web_server.backend.announcer import announcer

print(f"In importer, id(announcer) = {id(announcer)}")

app = flask.Flask("stream_server")


def format_sse(event, data=None) -> str:
    msg = f"event: {event}\n"
    if data:
        msg += f"data: {data}\n"
    return msg + "\n"


@app.route("/stream")
def stream():
    def eventStream():
        messages = announcer.listen()
        while True:
            try:
                event, data = messages.get(timeout=15)
            except Empty:
                # Send a heartbeat if no new messages arrive
                event, data = "heartbeat", f"keep-alive"
            if data == None:
                data = time.ctime(time.time())
            sse = format_sse(event, data)
            yield sse

    response = Response(eventStream(), mimetype="text/event-stream")
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Connection"] = "keep-alive"
    response.headers["Cache-Control"] = "no-cache"

    return response
