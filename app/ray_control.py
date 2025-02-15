import subprocess
import threading
import time
import os

from base_path import get_base_path


def start_ray_head_node():
  base_path = get_base_path()
  print(f'Starting Ray head node at base path {base_path}...')
  ray_path = os.path.join(base_path, 'bin', 'ray')

  # TODO: This may not work on Windows.
  resources = """'{"PixInsight": 1, "diskcache": 1, "fastest_storage": 1}'"""
  command = f"""ulimit -n 10000 && {ray_path} start --head --resources {resources}"""
  ray_process = subprocess.Popen(command, shell=True, start_new_session=True)

  # Spin until `ray status` tells us the node is up.
  while True:
    print('Calling `ray status`')
    process = subprocess.Popen(f"{ray_path} status",
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True,
                               text=True)
    output = process.communicate()[0]
    print('ray status output:', output)
    if "Autoscaler status" in output:
      print('Ray head node is up')
      break

    time.sleep(1)

  return ray_process
