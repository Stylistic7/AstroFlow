"""Pure — no Ray, no context."""
from light_pollution.sites import Site, cluster_sites, same_site, site_of


def test_decimal_and_sexagesimal_headers_read_alike():
    assert site_of({"SITELAT": 37.7433, "SITELONG": -122.4139}) == (37.7433, -122.4139)
    assert site_of({"SITELAT": "+37 47 00.00", "SITELONG": "-122 25 12.0"}) == (37.7833, -122.42)
    assert site_of({"SITELAT": "37:44:36", "SITELONG": "-122:25:12"}) == (37.7433, -122.42)


def test_missing_or_placeholder_positions_are_no_site():
    assert site_of({}) is None
    assert site_of({"SITELAT": None, "SITELONG": None}) is None
    assert site_of({"SITELAT": "xxxxxx", "SITELONG": "xxxxxx"}) is None
    assert site_of({"SITELAT": 0, "SITELONG": 0}) is None
    assert site_of({"SITELAT": 91.0, "SITELONG": 10.0}) is None
    assert site_of({"SITELAT": float("nan"), "SITELONG": 10.0}) is None


def test_the_same_roof_reported_at_two_precisions_is_one_site():
    sf_a, sf_b, oakland = (37.73, -122.42), (37.7433, -122.4139), (37.78, -122.20)
    sites = cluster_sites([sf_a, sf_b, oakland, sf_b])
    assert sites[sf_a] == sites[sf_b]
    assert sites[sf_a] != sites[oakland]
    assert sites[sf_a] == Site("37.74,-122.42", 37.74, -122.42)
    assert sites[oakland].key == "37.78,-122.20"
    assert same_site(sf_a, sf_b) and not same_site(sf_a, oakland)


def test_no_positions_no_sites():
    assert cluster_sites([]) == {}
