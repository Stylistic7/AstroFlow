import webbrowser
import sys
import socket
import os
import hashlib
from zeroconf import ServiceBrowser

from state import enqueue_action, main_event_loop, AstroFlowZeroconf, AstroFlowListener
from context import get_cache


def get_unique_instance_name():
  hostname = socket.gethostname()
  pid = os.getpid()
  unique_string = f"AstroFlow-{hostname}-{pid}"
  short_hash = hashlib.md5(
      unique_string.encode()).hexdigest()[:6]  # 6-char hash
  return f"AstroFlow-{short_hash}"


def main():
  astroflow_zc = AstroFlowZeroconf(instance_name=get_unique_instance_name())
  listener = AstroFlowListener(enqueue_action)
  browser = ServiceBrowser(astroflow_zc.zeroconf, "_astroflow._tcp.local.",
                           listener)

  enqueue_action(dict(type='start_flask'))

  cache = get_cache()
  if cache.get('ray_head_node_enabled', False):
    # TODO: Check for another head node and alert the user.
    astroflow_zc.update_state(is_ray_head=True)
    enqueue_action(dict(type='start_ray_head_node'))

  try:
    main_event_loop()
  except KeyboardInterrupt:
    print("Stopping service discovery...")
    astroflow_zc.close()


if __name__ == '__main__':
  main()
