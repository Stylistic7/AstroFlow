import subprocess
import sys
import time
import os


class GunicornManager:
  """Starts one router and N backend gunicorn processes."""

  def __init__(self,
               module: str,
               num_backends: int,
               router_port: int = 8000,
               start_port: int = 8001,
               timeout: int = 300):
    self.module = module
    self.num_backends = num_backends
    self.router_port = router_port
    self.start_port = start_port
    self.timeout = timeout
    self.processes = []

  def start(self):
    print(f"Starting router on port {self.router_port}...")
    self.processes.append(
        subprocess.Popen([
            sys.executable, "-m", "gunicorn", self.module, "--bind",
            f"0.0.0.0:{self.router_port}", "--workers", "4", "--threads", "4",
            "--log-level", "debug", "--capture-output", "--timeout",
            str(self.timeout)
        ]))

    for i in range(self.num_backends):
      port = self.start_port + i
      print(f"Starting backend on port {port}...")
      self.processes.append(
          subprocess.Popen([
              sys.executable, "-m", "gunicorn", self.module, "--bind",
              f"0.0.0.0:{port}", "--workers", "1", "--threads",
              str(os.cpu_count()), "--log-level", "debug", "--capture-output",
              "--timeout",
              str(self.timeout)
          ]))
      time.sleep(0.3)  # optional stagger

  def stop(self):
    for proc in self.processes:
      print(f"Stopping gunicorn (pid={proc.pid})...")
      proc.terminate()
    for proc in self.processes:
      proc.wait()
    print("All gunicorn processes stopped.")
