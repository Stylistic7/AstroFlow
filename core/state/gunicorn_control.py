from dataclasses import dataclass
import os
import subprocess
import sys
import time

from state.procs import CHILD_FLAGS


@dataclass(frozen=True)
class GunicornConfig:
    name: str
    config_file: str
    module: str
    worker_type: str
    num_servers: int
    start_port: int
    num_workers: int
    timeout: int
    # Windows only (one waitress process per port): its threads. gunicorn's
    # worker processes keep their 4 each.
    threads: int = 4
    # Windows only: `module:function` run in the server's process before it
    # serves — what the gunicorn config file's `post_fork` does on POSIX.
    on_start: str = ""


# A worker that dies at boot dies FAST — every observed failure exited in
# the same second it spawned — so a short settle window sees it reliably.
_BOOT_SETTLE_S = 6.0
_MAX_RESPAWN_ROUNDS = 3


class GunicornManager:
    """Starts one router and N backend gunicorn processes.

    Then WATCHES them settle and respawns any that die at boot. Concurrent
    numpy imports across simultaneously-spawned servers intermittently fail
    with `RuntimeError: Could not parse python long as longdouble` (macOS,
    numpy 2.x); gunicorn treats a worker boot failure as fatal and shuts the
    whole master down, so before this heal pass 1-2 of the 5 app servers
    silently died on most starts — and whichever slice of the tile-shard
    space hashed to the dead port served no images until someone noticed
    (2026-08-18/19, three times). A LONE respawn reliably succeeds (3/3
    measured) because the import race needs company, so one staggered retry
    round is normally the end of it.

    On Windows, where gunicorn cannot run (it forks), each port is one
    waitress process (`web_server.backend.serve`), supervised the same way.
    """

    def __init__(self, config: GunicornConfig):
        self.config = config
        self.processes = []   # aligned with self.ports
        self.ports = []

    def _command(self, port: int) -> list[str]:
        if sys.platform == "win32":
            command = [sys.executable, "-m", "web_server.backend.serve", self.config.module,
                       "--port", str(port), "--threads", str(self.config.threads)]
            if self.config.on_start:
                command += ["--on-start", self.config.on_start]
            return command
        command = [
            sys.executable,
            "-m",
            "gunicorn",
            "-k",
            self.config.worker_type,
            self.config.module,
            "--bind",
            f"0.0.0.0:{port}",
            "--threads",
            "4",
            "--log-level",
            "warning",
            "--capture-output",
        ]

        if self.config.config_file:
            command += ["-c", self.config.config_file]

        if self.config.num_workers:
            command += ["--workers", str(self.config.num_workers)]

        if self.config.timeout is not None:
            command += ["--timeout", str(self.config.timeout)]
        return command

    def _spawn(self, port: int) -> subprocess.Popen:
        command = self._command(port)
        print(f"### Starting web server with command: {command}")
        # Web workers get a SHORT cache-op timeout (the compute graph's
        # default is 86400s = "block until done"): a gunicorn thread
        # stuck behind the cache.db writer lock must error out in
        # seconds — 4 wedged threads previously made every endpoint
        # unresponsive for minutes with no recovery (gthread heartbeats
        # from its accept loop, so --timeout never reaps it). With the
        # hot web keys moved to the lock-free durable store this should
        # rarely trigger; it is the backstop, not the mechanism.
        env = os.environ | {"ASTROFLOW_CACHE_TIMEOUT_S": "15"}
        return subprocess.Popen(command, env=env, creationflags=CHILD_FLAGS)

    def start(self):
        for i in range(self.config.num_servers):
            port = self.config.start_port + i
            self.ports.append(port)
            self.processes.append(self._spawn(port))
            time.sleep(0.5)

        self._heal_boot_failures()
        return self

    def _heal_boot_failures(self):
        """Respawn any server whose master exited during the settle window.

        Bounded: `_MAX_RESPAWN_ROUNDS` rounds of settle-then-respawn, ~6 s
        each — the healthy case costs one settle. Respawns are staggered a
        beat apart so a retry round cannot recreate the simultaneous-import
        pile-up it exists to recover from.
        """
        for round_ in range(_MAX_RESPAWN_ROUNDS):
            time.sleep(_BOOT_SETTLE_S)
            dead = [
                i for i, p in enumerate(self.processes)
                if p.poll() is not None
            ]
            if not dead:
                return
            for i in dead:
                print(
                    f"### gunicorn :{self.ports[i]} died at boot "
                    f"(exit {self.processes[i].returncode}); respawning "
                    f"(round {round_ + 1}/{_MAX_RESPAWN_ROUNDS})"
                )
                self.processes[i] = self._spawn(self.ports[i])
                time.sleep(2.0)
        still_dead = [
            self.ports[i] for i, p in enumerate(self.processes)
            if p.poll() is not None
        ]
        if still_dead:
            # Loud, greppable, and non-fatal: the rest of the fleet serves.
            print(f"### gunicorn ports STILL DEAD after respawns: {still_dead}")

    def stop(self):
        for proc in self.processes:
            print(f"Stopping gunicorn (pid={proc.pid})...")
            proc.terminate()
        for proc in self.processes:
            proc.wait()
        print("All gunicorn processes stopped.")
