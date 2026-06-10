from dataclasses import dataclass, field
from typing import Callable, Dict, Any
from zeroconf import Zeroconf, ServiceListener, ServiceBrowser
import socket, subprocess, re, time, json

from state.types import Action


def _decode_props(props) -> Dict[str, str]:
    if not props:
        return {}
    out = {}
    for k, v in props.items():
        kd = k.decode() if isinstance(k, bytes) else str(k)
        if v is None:
            continue
        out[kd] = v.decode(errors="ignore") if isinstance(v, bytes) else str(v)
    return out


def _mac_for_ipv4(ip: str) -> str:
    subprocess.run(
        ["ping", "-c", "1", "-W", "1", ip],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    p = subprocess.run(["arp", "-n", ip], capture_output=True, text=True)
    s = (
        p.stdout
        if p.returncode == 0 and p.stdout
        else subprocess.run(["arp", "-an"], capture_output=True, text=True).stdout
    )
    m = re.search(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})", s or "")
    return m.group(1).lower() if m else ""


def _looks_like_asiair(name: str, server: str, txt: Dict[str, str]) -> bool:
    print(f"### _looks_like_asiair {name=} {server=} {txt=}")
    n = name.lower()
    if any(k in n for k in ("asiair", "asi-air", "asi_air")):
        return True
    if server and "asiair" in server.lower():
        return True
    if any("macsamba" in (v or "").lower() for v in txt.values()):
        return True
    return False


class ASIAirListener(ServiceListener):
    def __init__(self, dispatch_fn: Callable[[Action], None]):
        print(f"### ASIAirListener.__init__")
        self.dispatch_fn = dispatch_fn
        self.by_ip: Dict[str, Dict[str, Any]] = {}
        self.name_index: Dict[str, set[str]] = {}

    def _emit(self) -> None:
        print(f"### dispatching update_asiair_services action")
        if self.dispatch_fn:
            self.dispatch_fn(
                dict(
                    type="update_asiair_services",
                    devices=list(self.by_ip.values()),
                )
            )
        # Tell connected web clients to refetch the ASIAir panel so the
        # dropdown reflects the new device list without waiting for the
        # next polling tick. Lazy-import to avoid pulling the web_server
        # blueprints into the discovery import path (the discovery code
        # lives in the state machine, no Flask dependency).
        try:
            from web_server.backend.announcer import announce
            announce("asiair_devices")
        except Exception as e:
            print(f"### asiair_zeroconf: announce failed (non-fatal): {e}")

    def add_service(self, zc, svc_type, name):
        info = zc.get_service_info(svc_type, name, timeout=1500)
        if not info:
            return
        txt = _decode_props(info.properties)
        if not _looks_like_asiair(name, info.server or "", txt):
            return
        for ip in info.parsed_addresses():
            if ":" in ip:  # IPv6 skipped for MAC
                continue
            rec = {
                "ip": ip,
                "mac": self.by_ip.get(ip, {}).get("mac") or _mac_for_ipv4(ip),
                "hostname": info.server or "",
                "instance": name,
                "service_type": svc_type,
                "txt": txt,
            }
            self.by_ip[ip] = rec
            self.name_index.setdefault(name, set()).add(ip)

        self._emit()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        # Treat updates as refreshes (common pattern)
        self.add_service(zc, type_, name)

    def remove_service(self, zc, svc_type, name):
        ips = self.name_index.pop(name, set())
        for ip in ips:
            self.by_ip.pop(ip, None)

        self._emit()

    def get_devices(self):
        return list(self.by_ip.values())


@dataclass
class ASIAIRZeroconf:
    enqueue_action_fn: Callable[[Action], None]
    zeroconf: Zeroconf = field(init=False, repr=False)
    listener: ASIAirListener = field(init=False, repr=False)

    def __post_init__(self):
        self.zeroconf = Zeroconf()
        self.listener = ASIAirListener(self.enqueue_action_fn)
        self.b1 = ServiceBrowser(self.zeroconf, "_smb._tcp.local.", self.listener)
        self.b2 = ServiceBrowser(
            self.zeroconf, "_device-info._tcp.local.", self.listener
        )

    def close(self):
        self.zeroconf.close()
