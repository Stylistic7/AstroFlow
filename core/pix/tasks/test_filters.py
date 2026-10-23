"""`normalize_filter`: a filter name to its band."""
from pix.tasks.filters import normalize_filter


def test_band_names_and_names_that_go_on_to_give_a_wavelength():
    assert normalize_filter("Ha") == normalize_filter("H-Alpha") == "H"
    assert normalize_filter("H-Alpha 656/3") == "H"           # MDW's Hα
    assert normalize_filter("Ha 3nm") == "H" and normalize_filter("OIII-6.5nm") == "O"
    assert normalize_filter("SII 3nm") == "S"
    assert normalize_filter("L-Pro") is None and normalize_filter("") is None
