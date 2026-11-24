"""AstroFlow.app's main program: a Dock icon that runs AstroFlow.

AstroFlow itself (`core/main.py`) runs as a child process, exactly as
`make start` runs it, with its output in ~/Library/Logs/AstroFlow. This
process only owns the Mac side: the Dock icon and menus, opening the
dashboard once it answers, and a graceful shutdown on Quit.

What the bundle ships beside the code (`app/build_app.sh`): Python with every
package, GraXpert, solve-field and libvips (`Resources/tools`,
`Resources/lib`); the index files and the Gaia catalogue the app fetches at
first start (core/refdata). Nothing comes from Homebrew.

The app bundle (`app/build_app.sh`) runs this with its own Python; from a
checkout, `uv run --with pyobjc-framework-Cocoa python app/AstroFlow.py`.
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

from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSAlertSecondButtonReturn,
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSImage,
    NSMenu,
    NSMenuItem,
    NSWorkspace,
)
from Foundation import NSBundle, NSObject, NSURL
from PyObjCTools import AppHelper
import objc

ROOT = Path(__file__).resolve().parent.parent  # the bundle's Resources/AstroFlow, or a checkout
CORE = ROOT / "core"
RESOURCES = ROOT.parent  # the bundle's Resources (from a checkout, none of the below exist)
TOOLS_BIN = RESOURCES / "tools" / "bin"
VIPS_HOME = RESOURCES / "tools" / "vips"
ICON = next((p for p in (ROOT.parent / "icon.icns", ROOT / "app" / "icon.icns") if p.exists()), None)
DASHBOARD_PORT = 8000
DASHBOARD = f"http://localhost:{DASHBOARD_PORT}"
LOG = Path.home() / "Library" / "Logs" / "AstroFlow" / "AstroFlow.log"
# The bundle's own tools, then the system's: never Homebrew's, so every Mac in
# a cluster runs the same versions whatever else it has installed.
PATH = ":".join([str(TOOLS_BIN), "/usr/bin", "/bin", "/usr/sbin", "/sbin"])
SUPPORT = Path.home() / "Library" / "Application Support" / "AstroFlow"
# "worker" makes this Mac a worker; anything else (or no file) a head. A file,
# not NSUserDefaults: the process is the bundled Python, with no bundle id.
ROLE_FILE = SUPPORT / "role"
# How long Quit waits for AstroFlow's own graceful shutdown (in-flight work
# finishing) before it interrupts it.
GRACEFUL_QUIT_S = 600

NSTerminateNow, NSTerminateCancel, NSTerminateLater = 1, 0, 2


def _dashboard_answers() -> bool:
    try:
        socket.create_connection(("localhost", DASHBOARD_PORT), timeout=1).close()
        return True
    except OSError:
        return False


def _menu_item(title, action, key="", target=None):
    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
    if target is not None:
        item.setTarget_(target)
    return item


class AppDelegate(NSObject):
    child = None
    quitting = False
    restarting = False

    # --- lifecycle -------------------------------------------------------

    def applicationDidFinishLaunching_(self, _note):
        self._build_main_menu()
        NSApp.activateIgnoringOtherApps_(True)
        if _dashboard_answers():
            self._alert("AstroFlow is already running",
                        "It was started outside this app (from Terminal). Opening its dashboard.")
            webbrowser.open(DASHBOARD)
            NSApp.terminate_(None)
            return
        self._start_child()

    def applicationShouldTerminate_(self, _app):
        if self.child is None or self.child.poll() is not None:
            return NSTerminateNow
        self.quitting = True
        threading.Thread(target=self._stop_child_then_reply, daemon=True).start()
        return NSTerminateLater

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _app, _flag):
        self.openDashboard_(None)
        return False

    # --- the AstroFlow process ------------------------------------------

    @objc.python_method
    def _is_head(self) -> bool:
        return not (ROLE_FILE.exists() and ROLE_FILE.read_text().strip() == "worker")

    @objc.python_method
    def _start_child(self):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if LOG.exists():
            LOG.replace(LOG.with_suffix(".previous.log"))
        env = dict(os.environ, PATH=PATH,
                   PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                   ASTROFLOW_HEAD_NODE="1" if self._is_head() else "0")
        # libvips loads its plugins from here: an empty folder, never Homebrew's
        if VIPS_HOME.is_dir():
            env["VIPSHOME"] = str(VIPS_HOME)
        with open(LOG, "ab") as log:
            self.child = subprocess.Popen(
                [sys.executable, "main.py"], cwd=CORE, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        threading.Thread(target=self._watch_child, args=(self.child,), daemon=True).start()
        if self._is_head():
            threading.Thread(target=self._open_dashboard_when_up, args=(self.child,), daemon=True).start()

    @objc.python_method
    def _watch_child(self, child):
        child.wait()
        if not (self.quitting or self.restarting):
            self.performSelectorOnMainThread_withObject_waitUntilDone_("childDied:", None, False)

    def childDied_(self, _):
        choice = self._alert(
            "AstroFlow stopped unexpectedly",
            f"It exited with status {self.child.returncode}. The log has the details.",
            buttons=("Restart", "Show Log", "Quit"))
        if choice == NSAlertFirstButtonReturn:
            self._start_child()
        elif choice == NSAlertSecondButtonReturn:
            self.showLog_(None)
            self.childDied_(None)
        else:
            NSApp.terminate_(None)

    @objc.python_method
    def _open_dashboard_when_up(self, child):
        # A cold start (Ray, the web servers, the library scan) can take minutes.
        while child.poll() is None:
            if _dashboard_answers():
                webbrowser.open(DASHBOARD)
                return
            time.sleep(1)

    @objc.python_method
    def _stop_child(self):
        """AstroFlow's graceful shutdown, then its forced one: its Ctrl-C
        handling, where the first SIGINT enqueues the quit and a second runs
        its cleanup and exits. Never SIGTERM: Ray's crash handler owns SIGTERM
        in that process and kills it on the spot, leaving the web servers and
        Ray's agents running (2026-10-16)."""
        child = self.child
        if child is None or child.poll() is not None:
            return
        child.send_signal(signal.SIGINT)
        try:
            child.wait(GRACEFUL_QUIT_S)
            return
        except subprocess.TimeoutExpired:
            pass
        child.send_signal(signal.SIGINT)
        try:
            child.wait(60)
        except subprocess.TimeoutExpired:
            child.kill()

    @objc.python_method
    def _stop_child_then_reply(self):
        self._stop_child()
        self.performSelectorOnMainThread_withObject_waitUntilDone_("replyTerminate:", None, False)

    def replyTerminate_(self, _):
        NSApp.replyToApplicationShouldTerminate_(True)

    @objc.python_method
    def _restart_child(self):
        self.restarting = True
        self._stop_child()
        self.restarting = False
        self.performSelectorOnMainThread_withObject_waitUntilDone_("startChild:", None, False)

    def startChild_(self, _):
        self._start_child()

    # --- menus -----------------------------------------------------------

    @objc.python_method
    def _build_main_menu(self):
        app_menu = NSMenu.alloc().initWithTitle_("AstroFlow")
        app_menu.addItem_(_menu_item("About AstroFlow", "orderFrontStandardAboutPanel:"))
        app_menu.addItem_(NSMenuItem.separatorItem())
        app_menu.addItem_(_menu_item("Open AstroFlow", "openDashboard:", "o", self))
        app_menu.addItem_(_menu_item("Folders…", "openFolders:", ",", self))
        app_menu.addItem_(_menu_item("Show Log", "showLog:", "l", self))
        app_menu.addItem_(NSMenuItem.separatorItem())
        app_menu.addItem_(_menu_item("Hide AstroFlow", "hide:", "h"))
        app_menu.addItem_(_menu_item("Quit AstroFlow", "terminate:", "q"))
        root = _menu_item("AstroFlow", None)
        root.setSubmenu_(app_menu)
        bar = NSMenu.alloc().init()
        bar.addItem_(root)
        NSApp.setMainMenu_(bar)

    def applicationDockMenu_(self, _sender):
        menu = NSMenu.alloc().init()
        head = self._is_head()
        running = self.child is not None and self.child.poll() is None
        status = _menu_item(f"{'Head' if head else 'Worker'}: {'running' if running else 'stopped'}", None)
        status.setEnabled_(False)
        menu.addItem_(status)
        if head:
            menu.addItem_(_menu_item("Open AstroFlow", "openDashboard:", "", self))
            menu.addItem_(_menu_item("Folders…", "openFolders:", "", self))
        menu.addItem_(_menu_item("Show Log", "showLog:", "", self))
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(_menu_item("Run as a Worker" if head else "Run as the Head", "toggleRole:", "", self))
        return menu

    def openDashboard_(self, _sender):
        if self._is_head():
            webbrowser.open(DASHBOARD)

    def openFolders_(self, _sender):
        if self._is_head():
            webbrowser.open(f"{DASHBOARD}/folders")

    def showLog_(self, _sender):
        NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(LOG)))

    def toggleRole_(self, _sender):
        head = self._is_head()
        if self._alert(
                "Run this Mac as a worker?" if head else "Run this Mac as the head?",
                "A worker has no dashboard: it joins the head Mac on your network and takes a share "
                "of its work." if head else "The head runs the dashboard and hands work to any workers.",
                buttons=("Switch", "Cancel")) != NSAlertFirstButtonReturn:
            return
        ROLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ROLE_FILE.write_text("worker" if head else "head")
        threading.Thread(target=self._restart_child, daemon=True).start()

    @objc.python_method
    def _alert(self, title, text, buttons=("OK",)):
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(text)
        for b in buttons:
            alert.addButtonWithTitle_(b)
        NSApp.activateIgnoringOtherApps_(True)
        return alert.runModal()


def main():
    # Run from the bundle, this process is the bundled Python: give the menu
    # bar and the Dock the app's own name and icon.
    NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "AstroFlow"
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    if ICON:
        app.setApplicationIconImage_(NSImage.alloc().initWithContentsOfFile_(str(ICON)))
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
