import webbrowser
import sys
import socket
import os

from state import enqueue_action, main_event_loop, AstroFlowZeroconf, AstroFlowListener
from context import get_cache


def get_unique_instance_name():
  hostname = socket.gethostname()
  pid = os.getpid()
  return f"AstroFlow-{hostname}-{pid}"


def main():
  astroflow_zc = AstroFlowZeroconf(instance_name=get_unique_instance_name())
  listener = AstroFlowListener(enqueue_action)

  enqueue_action(dict(type='start_flask'))

  cache = get_cache()
  if cache.get('ray_head_node_enabled', False):
    enqueue_action(dict(type='start_ray_head_node'))

  main_event_loop()


if __name__ == '__main__':
  main()
