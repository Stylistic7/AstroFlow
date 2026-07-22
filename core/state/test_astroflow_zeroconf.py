"""AstroFlowListener.add_service property parsing. Ray-free: fake zeroconf
service info, no real mDNS.

Regression: a peer advertising a TXT property with an empty value (e.g.
cluster_version="" before the head freezes a snapshot) reads back as None, not
b"". `.get(key, default)` only substitutes the default when the key is ABSENT,
so None.decode() crashed the ServiceBrowser thread (broke `uv run python
main.py` even single-machine, since the process discovers itself).
"""
from state.astroflow_zeroconf import AstroFlowListener, _pick_peer_ip


class _Info:
    def __init__(self, properties, addresses=("10.0.0.53",)):
        self.properties = properties
        self._addresses = list(addresses)

    def parsed_addresses(self):
        return self._addresses


class _ZC:
    def __init__(self, info):
        self._info = info

    def get_service_info(self, type_, name):
        return self._info


def _discover(properties, addresses=("10.0.0.53",)):
    captured = {}
    listener = AstroFlowListener(lambda action: captured.update(action))
    listener.add_service(_ZC(_Info(properties, addresses)),
                         "_astroflow._tcp.local.", "peer")
    return listener.instances["peer"], captured


def test_present_but_none_values_use_defaults():
    # The crashing case: keys present with None values (empty TXT).
    status, captured = _discover(
        {b"ray_head": None, b"mounted_volumes": None, b"cluster_version": None}
    )
    assert status["ray_head"] == "no"
    assert status["mounted_volumes"] == []
    assert status["cluster_version"] == ""
    assert captured["type"] == "update_astroflow_services"


def test_absent_keys_use_defaults():
    status, _ = _discover({})
    assert status["ray_head"] == "no"
    assert status["mounted_volumes"] == []
    assert status["cluster_version"] == ""


def test_present_values_are_parsed():
    sha = "a" * 40
    status, _ = _discover({
        b"ray_head": b"yes",
        b"mounted_volumes": b'["Astro", "Astro2"]',
        b"cluster_version": sha.encode(),
    })
    assert status["ray_head"] == "yes"
    assert status["mounted_volumes"] == ["Astro", "Astro2"]
    assert status["cluster_version"] == sha
    assert status["ip"] == "10.0.0.53"


def test_pick_peer_ip_prefers_link_local_direct_link():
    # The 10 GbE (link-local) wins over WiFi and loopback — never blind [0].
    assert _pick_peer_ip(["10.0.0.49", "127.0.0.1", "169.254.90.33"]) == "169.254.90.33"


def test_pick_peer_ip_falls_back_to_routable_then_anything():
    assert _pick_peer_ip(["127.0.0.1", "10.0.0.49"]) == "10.0.0.49"  # skip loopback
    assert _pick_peer_ip(["127.0.0.1"]) == "127.0.0.1"               # last resort
    assert _pick_peer_ip([]) == ""


def test_add_service_picks_direct_link_not_wifi():
    # The bug: a head resolves to BOTH its 10 GbE and WiFi; discovery must land
    # on the 10 GbE regardless of A-record order (here WiFi is listed first).
    status, _ = _discover({b"ray_head": b"yes"},
                          addresses=["10.0.0.49", "169.254.90.33"])
    assert status["ip"] == "169.254.90.33"


def test_add_service_honors_explicit_advertised_ip():
    # An explicit advertised_ip TXT wins over the address heuristic entirely.
    status, _ = _discover(
        {b"ray_head": b"yes", b"advertised_ip": b"169.254.90.33"},
        addresses=["10.0.0.49", "127.0.0.1"],
    )
    assert status["ip"] == "169.254.90.33"
