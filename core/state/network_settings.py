"""Per-node network/Ray infrastructure settings.

Deliberately separate from `decisions.settings.Settings`, which feeds the
compute-graph cache key. These are runtime knobs (which interface to bind
Ray to, whether this machine runs the head) — they don't affect any
science output, and putting them in the hashed Settings would invalidate
every cached stack on a network change.

Persisted as a single diskcache entry under `cache["network_settings"]`.
"""
import socket
from dataclasses import dataclass, replace
from typing import Optional

import psutil

from context import get_cache


_CACHE_KEY = "network_settings"
# Whether the user has explicitly picked this Mac's role (head/worker) at least
# once. Kept as its own key (not a NetworkSettings field) so the first-run
# dialog logic doesn't depend on the dataclass shape. Absent → never chosen.
_ROLE_CHOSEN_KEY = "role_chosen"


def role_has_been_chosen() -> bool:
    return bool(get_cache().get(_ROLE_CHOSEN_KEY, False))


def mark_role_chosen() -> None:
    get_cache()[_ROLE_CHOSEN_KEY] = True


@dataclass(frozen=True)
class NetworkSettings:
    ray_head_node_enabled: bool = True
    cluster_interface: Optional[str] = None  # interface name (e.g. "en6"); None = auto-detect
    # Opt-in: deploy the head's live working tree to the WHOLE cluster (a
    # `wt:<hash>` snapshot) instead of the committed HEAD. The cluster still
    # runs identical code; it's just not reproducible from git, so it's off by
    # default. See state.code_version.
    uncommitted: bool = False


def get_network_settings() -> NetworkSettings:
    cache = get_cache()
    stored = cache.get(_CACHE_KEY)
    if isinstance(stored, NetworkSettings):
        return stored
    return NetworkSettings()


def set_network_settings(ns: NetworkSettings) -> None:
    cache = get_cache()
    cache[_CACHE_KEY] = ns


def update_network_settings(**fields) -> NetworkSettings:
    ns = replace(get_network_settings(), **fields)
    set_network_settings(ns)
    return ns


def resolve_interface_ip(interface_name: Optional[str]) -> Optional[str]:
    """Resolve `interface_name` to its current IPv4 address.

    Returns None when `interface_name` is None — caller should fall
    back to OS auto-detection. Raises ValueError if the interface is
    named but missing or has no IPv4 (e.g., link is down). Failing
    loud beats silently binding to the wrong NIC.
    """
    if interface_name is None:
        return None
    addrs = psutil.net_if_addrs()
    if interface_name not in addrs:
        raise ValueError(
            f"cluster_interface={interface_name!r} not found. "
            f"Available: {sorted(addrs.keys())}"
        )
    for addr in addrs[interface_name]:
        if addr.family == socket.AF_INET:
            return addr.address
    raise ValueError(
        f"cluster_interface={interface_name!r} has no IPv4 address. "
        "Is the link up?"
    )
