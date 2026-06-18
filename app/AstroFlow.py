import atexit
import os
import socket
import subprocess
import threading
import webbrowser

from AppKit import (
    NSApplication,
    NSApp,
    NSApplicationActivationPolicyRegular,
    NSAlert,
    NSAlertFirstButtonReturn,
    NSMenu,
    NSMenuItem,
)
from Foundation import NSObject
from PyObjCTools import AppHelper

from base_path import get_bin_dir
from dependency_graph.serialization import enable_function_fqname_pickling
from state.state_machine import (
    Action,
    enqueue_action,
    main_event_loop,
    is_head_node,
    is_uncommitted_mode,
    get_state,
)
from state.network_settings import (
    get_network_settings,
    update_network_settings,
    role_has_been_chosen,
    mark_role_chosen,
)

_ASTROFLOW_URL = "http://localhost:8000"


def _prompt_for_role_if_unset():
    """First launch only: ask whether this Mac is the head or a worker, and
    persist it. The env var ASTROFLOW_HEAD_NODE (if set) wins and skips the
    prompt. Without this, a fresh second Mac would default to head and wrongly
    become a second head. Runs on the main thread before the backend starts."""
    if os.environ.get("ASTROFLOW_HEAD_NODE"):
        return
    if role_has_been_chosen():
        return
    alert = NSAlert.alloc().init()
    alert.setMessageText_("AstroFlow — choose this Mac's role")
    alert.setInformativeText_(
        "Head runs the dashboard and orchestrates the cluster.\n"
        "Worker runs headless and joins a head over the network.\n\n"
        "You can change this later from the Dock icon's menu."
    )
    alert.addButtonWithTitle_("Head")     # NSAlertFirstButtonReturn
    alert.addButtonWithTitle_("Worker")
    is_head = alert.runModal() == NSAlertFirstButtonReturn
    update_network_settings(ray_head_node_enabled=is_head)
    mark_role_chosen()


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
        # Ask head/worker on first launch BEFORE the backend reads the role.
        _prompt_for_role_if_unset()

        self._backend = threading.Thread(target=self._run_backend, daemon=True)
        self._backend.start()

        # Only the head serves a UI; workers run headless. Don't open a browser
        # on a worker (it has no web server).
        if is_head_node():
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

    def applicationDockMenu_(self, sender):
        """Right-click Dock menu — the control surface for a headless worker
        (and a handy one for the head). Rebuilt each time so it reflects current
        status. macOS appends its own Quit, so we don't add one."""
        menu = NSMenu.alloc().init()
        head = is_head_node()

        status = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Role: {'Head' if head else 'Worker'}", None, ""
        )
        status.setEnabled_(False)
        menu.addItem_(status)

        if not head:
            # Show what code this worker is on + which head it's following.
            from state.worker_sync import current_synced_sha
            sha = current_synced_sha()
            following = get_state("connected_head_ip")
            detail = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                f"Code: {sha[:7] if sha else '—'}   "
                f"Head: {following or 'searching…'}",
                None, "",
            )
            detail.setEnabled_(False)
            menu.addItem_(detail)

        menu.addItem_(NSMenuItem.separatorItem())

        toggle = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Switch to Worker" if head else "Switch to Head",
            "toggleRole:", "",
        )
        toggle.setTarget_(self)
        menu.addItem_(toggle)

        if head:
            deploy = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Deploy committed code" if is_uncommitted_mode()
                else "Deploy uncommitted working tree",
                "toggleUncommitted:", "",
            )
            deploy.setTarget_(self)
            menu.addItem_(deploy)

            dash = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Open Dashboard", "openDashboard:", ""
            )
            dash.setTarget_(self)
            menu.addItem_(dash)

        return menu

    def toggleRole_(self, sender):
        """Flip head/worker, persist, and restart into the new role. The fresh
        process re-reads the role at init and re-forms the cluster accordingly."""
        new_is_head = not is_head_node()
        update_network_settings(ray_head_node_enabled=new_is_head)
        mark_role_chosen()
        from state.restart import spawn_replacement
        spawn_replacement()  # head runs its dev repo; a worker re-discovers
        enqueue_action(dict(type="quit"))

    def toggleUncommitted_(self, sender):
        """Flip committed ↔ uncommitted deploy and restart to re-freeze and
        redeploy the whole cluster onto the new snapshot."""
        update_network_settings(uncommitted=not is_uncommitted_mode())
        from state.restart import spawn_replacement
        spawn_replacement()  # relaunch from the repo; it re-freezes
        enqueue_action(dict(type="quit"))

    def openDashboard_(self, sender):
        webbrowser.open(_ASTROFLOW_URL)

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, flag):
        # Reopening the dock icon reopens the UI — head only (workers have none).
        if is_head_node():
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
