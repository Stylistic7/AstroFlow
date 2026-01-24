import threading
import time
from werkzeug.serving import run_simple

from web_server.backend.app import app


class FlaskThread(threading.Thread):
    """Runs a Flask web server in a separate thread with clean start/stop handling."""

    def __init__(self, host="0.0.0.0", port=80):
        super().__init__(daemon=True)  # Ensures the thread exits with the main program
        self.host = host
        self.port = port
        self.server = None

    def run(self):
        """Starts the Flask server inside the thread."""
        print("Starting Flask server...")
        self.server = run_simple(self.host, self.port, app, threaded=True)
        self.server.serve_forever()

    def stop(self):
        """Stops the Flask server gracefully."""
        if self.server:
            self.server.shutdown()
        print("Flask server stopped.")
