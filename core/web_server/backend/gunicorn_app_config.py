"""Gunicorn config for the APP servers (static_server + the tile shards).

Bounds how long one request thread may sit in a single socket operation.

The wedge it fixes (seen repeatedly, 2026-09-08/09; `sample` on a hung
worker showed every thread in `sock_sendall` -> `__sendto`): these run as
gthread workers — `-k sync` plus the shared `--threads 4` silently promotes
to gthread — and gthread's accept loop heartbeats from the MAIN thread, so
`--timeout 300` reaps nothing while the pool threads are stuck. A browser
tab closed mid-download leaves its socket with a full send buffer and no
reader; a blocking `sendall` into it never returns. Four such requests
consume all four threads and the server answers nothing, forever, with a
perfectly healthy-looking master. Only a restart cleared it.

A socket timeout turns that permanent block into a `TimeoutError`, which is
an `OSError` — gunicorn's `ThreadWorker.handle` already catches it, logs
"Socket error processing request", and `finish_request` closes the
connection and frees the thread. Recovery is the path gunicorn already has;
all this does is make it reachable.

The timeout is per socket OPERATION, not per request: a big response is
written block by block (`Response.write_file` -> `socket.sendfile`, whose
selector waits per block), so a slow-but-progressing client is never cut.
It has to make NO progress for the whole window to be dropped.
"""
# Generous: nothing on a LAN legitimately stalls a single send or recv for
# two minutes, and it sits well under the arbiter's `--timeout 300`, so the
# socket gives up first and the worker survives.
SOCKET_TIMEOUT_S = 120.0

_patched = False


def post_fork(server, worker):
    """Bound every accepted connection's socket operations.

    Applied by patching `TConn.init`, which is the one point that runs
    AFTER `ThreadWorker.handle`'s unconditional `setblocking(True)` — a
    timeout set anywhere earlier (on the listener, at accept, via
    `socket.setdefaulttimeout`) is cleared by that call. Runs post-fork, so
    the patch is per worker process.
    """
    global _patched
    if _patched:
        return
    _patched = True

    from gunicorn.workers.gthread import TConn

    original_init = TConn.init

    def init_with_socket_timeout(self):
        original_init(self)
        self.sock.settimeout(SOCKET_TIMEOUT_S)

    TConn.init = init_with_socket_timeout
