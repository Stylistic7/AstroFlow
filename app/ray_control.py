import subprocess
import threading
import time


def start_ray_head_node_thread_body():
  # TODO: This may not work on Windows.
  command = """source ~/.zshrc && conda activate AstroFlow && ulimit -n 10000 && ray start --head --resources '{"PixInsight": 1, "diskcache": 1, "fastest_storage": 1}'"""
  subprocess.run(command, shell=True, start_new_session=True)

  # Spin until `ray status` tells us the node is up.
  while True:
    print('Calling `ray status`')
    process = subprocess.Popen(
        "source ~/.zshrc && conda activate AstroFlow && ray status",
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


def start_ray_head_node():
  threading.Thread(target=start_ray_head_node_thread_body).start()
