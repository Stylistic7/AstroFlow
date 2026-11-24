"""AstroFlow's Windows launcher: a tray icon that runs AstroFlow.

AstroFlow itself (`core/main.py`) runs as a child process, as `app/AstroFlow.py`
runs it on the Mac, with its output in %LOCALAPPDATA%\\AstroFlow\\Logs. This
process owns only the Windows side: the tray icon and its menu, opening the
dashboard once it answers, and a graceful quit.

The installer runs this with the bundled Python (pythonw.exe, no console) and
puts the bundled tools on PATH through the environment below; from a
checkout, `uv run pythonw app/windows/AstroFlow.pyw`.
"""
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import pystray
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]      # the install's AstroFlow folder, or a checkout
CORE = ROOT / "core"
TOOLS = ROOT.parent / "tools"                   # the installer's, beside AstroFlow; absent from a checkout
DASHBOARD_PORT = 8000
DASHBOARD = f"http://localhost:{DASHBOARD_PORT}"
LOGS = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AstroFlow" / "Logs"
LOG = LOGS / "AstroFlow.log"
ICON = next((p for p in (ROOT.parent / "icon.ico", ROOT / "app" / "windows" / "icon.ico")
             if p.exists()), None)
# How long Quit waits for AstroFlow's own graceful shutdown (in-flight work
# finishing) before it interrupts it, then kills it.
GRACEFUL_QUIT_S = 600


def _dashboard_answers() -> bool:
    try:
        socket.create_connection(("localhost", DASHBOARD_PORT), timeout=1).close()
        return True
    except OSError:
        return False


def _environment() -> dict[str, str]:
    """The bundled tools first on PATH, and where the bundled data is; a
    checkout has none of it and runs with the machine's own."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    paths = [str(TOOLS / "bin"), str(TOOLS / "solve-field" / "bin"), str(TOOLS / "vips" / "bin")]
    env["PATH"] = os.pathsep.join([*(p for p in paths if Path(p).is_dir()), env.get("PATH", "")])
    if (TOOLS / "vips").is_dir():
        env["VIPSHOME"] = str(TOOLS / "vips")
    return env




class Launcher:
    def __init__(self):
        self.child: subprocess.Popen | None = None
        self.quitting = False
        self.icon = pystray.Icon(
            "AstroFlow", self._image(), "AstroFlow",
            menu=pystray.Menu(
                pystray.MenuItem("Open AstroFlow", lambda: webbrowser.open(DASHBOARD), default=True),
                pystray.MenuItem("Folders…", lambda: webbrowser.open(f"{DASHBOARD}/folders")),
                pystray.MenuItem("Show Log", lambda: os.startfile(LOG)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit AstroFlow", self._quit),
            ))

    @staticmethod
    def _image() -> Image.Image:
        return Image.open(ICON) if ICON else Image.new("RGB", (64, 64), (30, 60, 120))

    def run(self):
        if _dashboard_answers():
            webbrowser.open(DASHBOARD)      # already running (from a terminal): just open it
            return
        self._start_child()
        self.icon.run()

    def _start_child(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        if LOG.exists():
            LOG.replace(LOG.with_suffix(".previous.log"))
        # Its own process group: Quit sends it Ctrl-Break (main.py quits on
        # SIGBREAK as on Ctrl-C), and a console Ctrl-C never reaches it.
        with open(LOG, "ab") as log:
            self.child = subprocess.Popen(
                [sys.executable.replace("pythonw.exe", "python.exe"), "main.py"], cwd=CORE,
                env=_environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
        threading.Thread(target=self._watch_child, daemon=True).start()
        threading.Thread(target=self._open_dashboard_when_up, daemon=True).start()

    def _watch_child(self):
        code = self.child.wait()
        if not self.quitting:
            self.icon.notify(f"AstroFlow stopped unexpectedly (exit {code}). See Show Log.", "AstroFlow")

    def _open_dashboard_when_up(self):
        # A cold start (Ray, the web servers, the library scan) can take minutes.
        while self.child.poll() is None:
            if _dashboard_answers():
                webbrowser.open(DASHBOARD)
                return
            time.sleep(1)

    def _stop_child(self):
        """AstroFlow's graceful shutdown (one Ctrl-Break), then its forced
        one (a second), then the kill."""
        child = self.child
        if child is None or child.poll() is not None:
            return
        for wait in (GRACEFUL_QUIT_S, 60):
            child.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                child.wait(wait)
                return
            except subprocess.TimeoutExpired:
                pass
        child.kill()

    def _quit(self):
        self.quitting = True
        threading.Thread(target=lambda: (self._stop_child(), self.icon.stop()), daemon=True).start()


if __name__ == "__main__":
    Launcher().run()
