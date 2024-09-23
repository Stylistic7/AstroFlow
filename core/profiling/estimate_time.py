import ray
import ipaddress
import numpy as np

from context import context


def estimate_time(tag, number=1):
  node_ips = []
  resources = ray.cluster_resources()
  for k, v in resources.items():
    try:
      _, ip = k.split('node:')
      ip = ipaddress.ip_address(ip)
      node_ips.append(str(ip))
    except:
      pass

  total_count = 0
  total_time = 0
  for node_ip in node_ips:
    key = ('performance_history', node_ip, tag)
    if history := context.get(key, None):
      total_time += np.sum(history)
      total_count += len(history)

  if total_count:
    return number * (total_time / total_count)
