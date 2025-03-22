from dataclasses import dataclass, field
from typing import Callable
from zeroconf import Zeroconf, ServiceInfo, ServiceListener, ServiceBrowser
import socket
import json
import os
import hashlib

from state.types import Action


def get_unique_instance_name():
  hostname = socket.gethostname()
  pid = os.getpid()
  unique_string = f"AstroFlow-{hostname}-{pid}"
  short_hash = hashlib.md5(
      unique_string.encode()).hexdigest()[:6]  # 6-char hash
  return f"AstroFlow-{short_hash}"


class AstroFlowListener(ServiceListener):

  def __init__(self, dispatch_fn):
    self.instances = {}  # Store discovered instances
    self.dispatch_fn = dispatch_fn

  def add_service(self, zc, type, name):
    info = zc.get_service_info(type, name)
    print(f'### add_service {info}')
    if info:
      ip = info.parsed_addresses()[0]
      status = {
          "name":
              name,
          "ip":
              ip,
          "ray_head":
              info.properties.get(b"ray_head", b"no").decode(),
          "mounted_volumes":
              json.loads(
                  info.properties.get(b"mounted_volumes", b"[]").decode()),
      }
      self.instances[name] = status
      print(f"Discovered AstroFlow instance: {status}")

      if self.dispatch_fn:
        self.dispatch_fn(
            dict(type='update_zeroconf_services', instances=self.instances))

  def remove_service(self, zc, type, name):
    if name in self.instances:
      del self.instances[name]
      print(f"AstroFlow instance {name} removed")

      if self.dispatch_fn and self.instances:
        self.dispatch_fn(
            dict(type='update_zeroconf_services', instances=self.instances))

  def get_instances(self):
    """Returns a list of all currently discovered AstroFlow instances."""
    return list(self.instances.values())


@dataclass
class AstroFlowZeroconf:
  enqueue_action_fn: Callable[[Action], None]
  # instance_name: str
  # listener: AstroFlowListener
  # browser: ServiceBrowser
  port: int = 6379
  is_ray_head: bool = False
  mounted_volumes: list[str] = field(default_factory=list)
  zeroconf: Zeroconf = field(init=False, repr=False)
  service_info: ServiceInfo | None = field(default=None, init=False, repr=False)

  def __post_init__(self):
    """Initialize Zeroconf and register the service."""
    self.instance_name = get_unique_instance_name()
    self.zeroconf = Zeroconf()
    self.service_name = f"{self.instance_name}._astroflow._tcp.local."
    self.listener = AstroFlowListener(self.enqueue_action_fn)
    self.browser = ServiceBrowser(self.zeroconf, "_astroflow._tcp.local.",
                                  self.listener)

  def get_ip(self) -> str:
    """Retrieve the local machine's LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      s.connect(("8.8.8.8", 80))
      ip = s.getsockname()[0]
    except Exception:
      ip = "127.0.0.1"
    finally:
      s.close()
    return ip

  def register_service(self):
    """Registers or updates the AstroFlow instance on the network."""
    properties = {
        "ray_head": "yes" if self.is_ray_head else "no",
        "mounted_volumes": json.dumps(self.mounted_volumes),
    }

    hostname = socket.gethostname()
    info = ServiceInfo("_astroflow._tcp.local.",
                       self.service_name,
                       addresses=[socket.inet_aton(self.get_ip())],
                       port=self.port,
                       properties=properties,
                       server=hostname)

    if self.service_info:
      self.zeroconf.unregister_service(self.service_info)  # Remove old service

    self.zeroconf.register_service(info)  # Register updated service
    self.service_info = info  # Store latest service info
    print(f'Registering AstroFlow service: '
          f'{self.instance_name} on {self.get_ip()}')

  def update_state(self,
                   is_ray_head: bool | None = None,
                   mounted_volumes: list[str] | None = None):
    """Call this function when the state changes."""
    if is_ray_head is not None:
      self.is_ray_head = is_ray_head
    if mounted_volumes is not None:
      self.mounted_volumes = mounted_volumes
    self.register_service()  # Re-register with updated state

  def close(self):
    """Unregister the service and clean up Zeroconf."""
    if self.service_info:
      self.zeroconf.unregister_service(self.service_info)
    self.zeroconf.close()
