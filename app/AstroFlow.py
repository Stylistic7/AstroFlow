import atexit
import socket
import subprocess
import threading
import webbrowser

from AppKit import NSApplication, NSApp, NSApplicationActivationPolicyRegular
from Foundation import NSObject
from PyObjCTools import AppHelper

from base_path import get_bin_dir
from dependency_graph.serialization import enable_function_fqname_pickling
from state.state_machine import Action, enqueue_action, main_event_loop

_ASTROFLOW_URL = "http://localhost:8000"


def _force_cleanup():
    ray_path = get_bin_dir() / "ray"
    subprocess.run(["pkill", "-f", "gunicorn"], capture_output=True)
    subprocess.run([ray_path, "stop"], capture_output=True)


def _wait_and_open_browser():
    """Poll until the web server is accepting connections, then open browser."""
    for _ in range(60):
        try:
            s = socket.create_connection(("localhost", 8000), timeout=1)
            s.close()
            webbrowser.open(_ASTROFLOW_URL)
            return
        except OSError:
            import time

            time.sleep(1)


class AppDelegate(NSObject):
    _shutting_down = False

    def applicationDidFinishLaunching_(self, notification):
        self._backend = threading.Thread(target=self._run_backend, daemon=True)
        self._backend.start()

        threading.Thread(target=_wait_and_open_browser, daemon=True).start()

    def _run_backend(self):
        enable_function_fqname_pickling()
        enqueue_action(Action(type="init"))
        main_event_loop()
        # Graceful shutdown finished — terminate the app on the main thread
        self._shutting_down = True
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "doTerminate:", None, False
        )

    def doTerminate_(self, sender):
        NSApp.terminate_(None)

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, flag):
        webbrowser.open(_ASTROFLOW_URL)
        return False

    def applicationShouldTerminate_(self, app):
        if self._shutting_down:
            return 1  # NSTerminateNow
        self._shutting_down = True
        enqueue_action(dict(type="quit"))
        return 0  # NSTerminateCancel — backend thread will terminate when done


def main():
    atexit.register(_force_cleanup)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
