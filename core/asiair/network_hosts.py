"""Hosts on the directly attached networks with a TCP port open: how the
ASIAirs are found (`asiair.live_poller`), since their mDNS names collide."""
from __future__ import annotations

import ipaddress
import itertools
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import netifaces

_MAX_ADDRESSES = 1024   # a /22; a bigger interface is not a home LAN, and each address costs a probe
_CONNECT_TIMEOUT_S = 0.5


def _local_networks() -> list[ipaddress.IPv4Network]:
    nets = []
    for iface in netifaces.interfaces():
        for addr in netifaces.ifaddresses(iface).get(netifaces.AF_INET, []):
            if "netmask" not in addr or "broadcast" not in addr:
                continue   # loopback or point-to-point (VPN)
            net = ipaddress.ip_network(f"{addr['addr']}/{addr['netmask']}", strict=False)
            if not net.is_link_local and net.num_addresses <= _MAX_ADDRESSES:
                nets.append(net)
    return nets


def check_socket(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(_CONNECT_TIMEOUT_S)
        return sock.connect_ex((host, port)) == 0


def find_hosts_with_open_port(port: int) -> list[str]:
    hosts = sorted({str(h) for net in _local_networks() for h in net.hosts()}, key=ipaddress.IPv4Address)
    with ThreadPoolExecutor(max_workers=32) as ex:
        is_open = ex.map(check_socket, hosts, itertools.repeat(port))
        return [h for h, ok in zip(hosts, is_open) if ok]


if __name__ == "__main__":
    print(find_hosts_with_open_port(4700))
