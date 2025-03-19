import os
import subprocess
import time

from base_path import get_bin_dir


def _ray_status():
  ray_path = get_bin_dir() / 'ray'
  result = subprocess.run([ray_path, "status"], capture_output=True, text=True)
  return "Autoscaler status" in result.stdout


def start_ray_head_node():
  # TODO: Check for an existing Ray cluster.
  ray_path = get_bin_dir() / 'ray'
  resources = """'{"PixInsight": 1, "diskcache": 1, "fastest_storage": 1}'"""
  command = f"""ulimit -n 10000 && {ray_path} start --head --resources {resources}"""
  print(f'### Starting Ray head node with command "{command}"')
  process = subprocess.Popen(
      command,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1  # Line-buffered output
  )

  while True:
    print('### Waiting for Ray node to start...')
    if _ray_status():
      break
    time.sleep(1)

  return process


def stop_ray_node():
  ray_path = get_bin_dir() / 'ray'
  subprocess.run([ray_path, "stop"], capture_output=True, text=True)

  while True:
    print('### Waiting for Ray node to stop...')
    if not _ray_status():
      break
    time.sleep(1)
