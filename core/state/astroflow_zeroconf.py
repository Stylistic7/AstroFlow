from dataclasses import dataclass, field
from typing import Callable
from zeroconf import Zeroconf, ServiceInfo, ServiceListener, ServiceBrowser
import socket
import json
import os
import hashlib
import threading
import time

from state.types import Action


def get_unique_instance_name():
    hostname = socket.gethostname()
    pid = os.getpid()
    unique_string = f"AstroFlow-{hostname}-{pid}"
    short_hash = hashlib.md5(unique_string.encode()).hexdigest()[:6]  # 6-char hash
    return f"AstroFlow-{short_hash}"


def _pick_peer_ip(addresses: list[str]) -> str:
    """Choose which of a peer's resolved addresses to connect to.

    A peer's hostname resolves to EVERY interface it has — the OS publishes an
    A-record per interface (10 GbE + WiFi) — so `parsed_addresses()` returns
    them all, in arbitrary order. Taking `[0]` blindly can land on the slow
    WiFi and defeat the fast-link pinning. Prefer a direct-link (link-local,
    169.254/16) address for the 10 GbE, then any routable address, and only
    then loopback. This is the fallback; a peer's explicit `advertised_ip` TXT
    (the interface it *chose* to advertise) wins over it when present.
    """
    if not addresses:
        return ""
    link_local = [a for a in addresses if a.startswith("169.254.")]
    routable = [a for a in addresses
                if a != "127.0.0.1" and not a.startswith("169.254.")]
    return (link_local or routable or addresses)[0]


class AstroFlowListener(ServiceListener):
    def __init__(self, dispatch_fn):
        self.instances = {}  # Store discovered instances
        self.dispatch_fn = dispatch_fn

    def _emit(self) -> None:
        if self.dispatch_fn:
            self.dispatch_fn(
                dict(type="update_astroflow_services", instances=self.instances)
            )

    def add_service(self, zc, type, name):
        info = zc.get_service_info(type, name)
        print(f"### add_service {info}")
        if info:
            def prop(key: bytes, default: bytes) -> str:
                # `.get(key, default)` substitutes the default only when the
                # key is ABSENT. A TXT property advertised with an empty value
                # (e.g. cluster_version="" before the head freezes a snapshot)
                # reads back as None, so guard for that explicitly.
                value = info.properties.get(key)
                return (value if value is not None else default).decode()

            # Prefer the peer's EXPLICITLY-advertised interface IP (the one it
            # pinned via `advertised_ip`); fall back to choosing among resolved
            # A-records — preferring the fast direct link, never blind `[0]`,
            # which mixes every interface and can land on the slow WiFi.
            ip = prop(b"advertised_ip", b"") or _pick_peer_ip(info.parsed_addresses())

            status = {
                "name": name,
                "ip": ip,
                "ray_head": prop(b"ray_head", b"no"),
                "mounted_volumes": json.loads(prop(b"mounted_volumes", b"[]")),
                # The committed SHA (or wt:<hash>) the head is running. Workers
                # follow it; absent/"" means a peer not advertising a snapshot.
                "cluster_version": prop(b"cluster_version", b""),
            }
            self.instances[name] = status
            print(f"Discovered AstroFlow instance: {status}")

            self._emit()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        # Treat updates as refreshes (common pattern)
        self.add_service(zc, type_, name)

    def remove_service(self, zc, type, name):
        if name in self.instances:
            del self.instances[name]
            print(f"AstroFlow instance {name} removed")

            self._emit()

    def get_instances(self):
        """Returns a list of all currently discovered AstroFlow instances."""
        return list(self.instances.values())


@dataclass
class AstroFlowZeroconf:
    enqueue_action_fn: Callable[[Action], None]
    port: int = 6379
    is_ray_head: bool = False
    mounted_volumes: list[str] = field(default_factory=list)
    # The committed SHA this node is running (or "dev"). Advertised so workers
    # can follow the head's code. Empty until the head computes it.
    cluster_version: str = ""
    # When set, advertise this IP instead of the OS default-route
    # heuristic. Used to broadcast the user-chosen cluster interface
    # (e.g. 10 GbE) regardless of which interface holds the default
    # route (typically WiFi). Multicast still goes out on every
    # interface — only the address inside the announcement changes.
    advertised_ip: str | None = None
    zeroconf: Zeroconf = field(init=False, repr=False)
    listener: AstroFlowListener = field(init=False, repr=False)
    service_info: ServiceInfo | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize Zeroconf, register the service, and start periodic re-check."""
        self.instance_name = get_unique_instance_name()
        self.zeroconf = Zeroconf()
        self.service_name = f"{self.instance_name}._astroflow._tcp.local."
        self.listener = AstroFlowListener(self.enqueue_action_fn)
        self.browser = ServiceBrowser(
            self.zeroconf, "_astroflow._tcp.local.", self.listener
        )
        self.register_service()
        self._rescan_thread = threading.Thread(
            target=self._periodic_rescan, daemon=True
        )
        self._rescan_thread.start()

    def get_ip(self) -> str:
        """The IP this instance advertises in zeroconf announcements.

        Prefers the explicitly-pinned `advertised_ip` (set when the user
        chose a specific cluster interface). Falls back to the OS
        default-route heuristic, which on a WiFi+10GbE Mac will always
        pick WiFi.
        """
        if self.advertised_ip:
            return self.advertised_ip
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
            "cluster_version": self.cluster_version,
            # The interface IP this node CHOSE to advertise (the pinned cluster
            # interface, e.g. 10 GbE). Peers use this directly instead of
            # guessing from the hostname's A-records, which mix every interface.
            "advertised_ip": self.get_ip(),
        }

        hostname = socket.gethostname()
        info = ServiceInfo(
            "_astroflow._tcp.local.",
            self.service_name,
            addresses=[socket.inet_aton(self.get_ip())],
            port=self.port,
            properties=properties,
            server=hostname,
        )

        if self.service_info:
            self.zeroconf.unregister_service(self.service_info)  # Remove old service

        self.zeroconf.register_service(info)  # Register updated service
        self.service_info = info  # Store latest service info
        print(f"Registering AstroFlow service: {self.instance_name} on {self.get_ip()}")

    def update_state(
        self,
        is_ray_head: bool | None = None,
        mounted_volumes: list[str] | None = None,
        advertised_ip: str | None = None,
        cluster_version: str | None = None,
    ):
        """Call this function when the state changes."""
        if is_ray_head is not None:
            self.is_ray_head = is_ray_head
        if mounted_volumes is not None:
            self.mounted_volumes = mounted_volumes
        if advertised_ip is not None:
            self.advertised_ip = advertised_ip
        if cluster_version is not None:
            self.cluster_version = cluster_version
        self.register_service()  # Re-register with updated state

    def _periodic_rescan(self):
        """Re-emit discovered instances every 30s so the state machine can retry."""
        while True:
            time.sleep(30)
            if self.listener.instances:
                self.listener._emit()

    def close(self):
        """Unregister the service and clean up Zeroconf."""
        if self.service_info:
            self.zeroconf.unregister_service(self.service_info)
        self.zeroconf.close()
