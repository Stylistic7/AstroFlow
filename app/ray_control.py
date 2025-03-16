import time
import os

from base_path import get_base_path
from flask_control import start_flask


def start_ray_head_node(manager):
  # TODO: Check for an existing Ray cluster.
  # TODO: Put controls in the UI.
  base_path = get_base_path()
  ray_path = os.path.join(base_path, 'bin', 'ray')
  resources = """'{"PixInsight": 1, "diskcache": 1, "fastest_storage": 1}'"""
  command = f"""ulimit -n 10000 && {ray_path} start --head --resources {resources}"""
  print(f'### Starting Ray head node with command "{command}"')
  manager.start_process(command)

  def maybe_start_flask_server(ray_status_output):
    if "Autoscaler status" in ray_status_output:
      print('Ray head node is up; starting web server...')
      start_flask()
    else:
      time.sleep(1)
      check_ray_status()

  def check_ray_status():
    command = f"{ray_path} status"
    manager.start_process(command, done_callback=maybe_start_flask_server)

  check_ray_status()
