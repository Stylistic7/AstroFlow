from dataclasses import dataclass
import os
import subprocess
import sys
import time


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


class GunicornManager:
    """Starts one router and N backend gunicorn processes."""

    def __init__(self, config: GunicornConfig):
        self.config = config
        self.processes = []

    def start(self):
        for i in range(self.config.num_servers):
            port = self.config.start_port + i
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

            print(f"### Starting gunicorn process with command: {command}")
            # Web workers get a SHORT cache-op timeout (the compute graph's
            # default is 86400s = "block until done"): a gunicorn thread
            # stuck behind the cache.db writer lock must error out in
            # seconds — 4 wedged threads previously made every endpoint
            # unresponsive for minutes with no recovery (gthread heartbeats
            # from its accept loop, so --timeout never reaps it). With the
            # hot web keys moved to the lock-free durable store this should
            # rarely trigger; it is the backstop, not the mechanism.
            env = os.environ | {"ASTROFLOW_CACHE_TIMEOUT_S": "15"}
            self.processes.append(subprocess.Popen(command, env=env))

            time.sleep(0.5)

        return self

    def stop(self):
        for proc in self.processes:
            print(f"Stopping gunicorn (pid={proc.pid})...")
            proc.terminate()
        for proc in self.processes:
            proc.wait()
        print("All gunicorn processes stopped.")
