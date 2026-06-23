"""AstroFlowListener.add_service property parsing. Ray-free: fake zeroconf
service info, no real mDNS.

Regression: a peer advertising a TXT property with an empty value (e.g.
cluster_version="" before the head freezes a snapshot) reads back as None, not
b"". `.get(key, default)` only substitutes the default when the key is ABSENT,
so None.decode() crashed the ServiceBrowser thread (broke `uv run python
main.py` even single-machine, since the process discovers itself).
"""
from state.astroflow_zeroconf import AstroFlowListener


class _Info:
    def __init__(self, properties):
        self.properties = properties

    def parsed_addresses(self):
        return ["10.0.0.53"]


class _ZC:
    def __init__(self, info):
        self._info = info

    def get_service_info(self, type_, name):
        return self._info


def _discover(properties):
    captured = {}
    listener = AstroFlowListener(lambda action: captured.update(action))
    listener.add_service(_ZC(_Info(properties)), "_astroflow._tcp.local.", "peer")
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
