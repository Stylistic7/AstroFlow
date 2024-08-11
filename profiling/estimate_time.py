import ray
import ipaddress

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
    history = context.get(key, None)
    if history is None:
      continue

    total_time += history.time * history.count
    total_count += history.count

  if total_count:
    return number * (total_time / total_count)
