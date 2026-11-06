"""The sky viewer's endpoints on a tiny synthetic map, through a bare Flask
app carrying only the blueprint. No app, no Ray."""
import numpy as np
import pytest
from astropy.io import fits
from flask import Flask

from allsky import maps
from web_server.backend.healpix import healpix_bp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "MAPS_DIR", tmp_path)
    maps._cache.clear()
    nside = 4
    values = np.linspace(1.0, 2.0, 12 * nside * nside).astype(np.float32)
    col = fits.Column(name="SIGNAL", format="E", unit="MJy/sr", array=values)
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["ORDERING"] = "NESTED"
    hdu.header["NSIDE"] = nside
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(tmp_path / "dust.fits")
    app = Flask(__name__)
    app.register_blueprint(healpix_bp)
    return app.test_client()


def test_index_lists_maps_and_single_input_finishes(client):
    d = client.get("/api/healpix").get_json()
    assert d["maps"] == ["dust"]
    assert "H" in d["finishes"] and "Master" in d["finishes"]
    assert "Hubble" not in d["finishes"]
    assert len(d["finishes"]) == len(set(d["finishes"]))     # one entry per finish name


def test_metadata_carries_the_finish_and_the_seed_stats(client):
    d = client.get("/api/healpix/dust/metadata").get_json()
    assert (d["nside"], d["frame"], d["unit"]) == (4, "galactic", "MJy/sr")
    assert d["finish"]["name"] == "H" and "displayFinish" in d["finish"]["wgsl"]
    inp = d["inputs"][0]
    assert inp["bands"] == 1 and inp["bandRoles"] == ["H"] and inp["colorspace"] == "linear-scene"
    assert inp["channelStats"][0]["skyMedian"] == pytest.approx(0.005)
    m = client.get("/api/healpix/dust/metadata?finish=Master").get_json()
    assert m["finish"]["name"] == "Master"
    assert client.get("/api/healpix/dust/metadata?finish=Hubble").status_code == 404
    assert client.get("/api/healpix/nope/metadata").status_code == 404


def test_tiles_are_float32_rasters_in_file_units(client):
    d = client.get("/api/healpix/dust/metadata").get_json()
    assert (d["tileWidth"], d["maxOrder"], d["tiles"]) == (4, 0, "")
    r = client.get("/api/healpix/dust/tile/0/5.f32?v=1")
    assert r.status_code == 200
    assert "immutable" in r.headers["Cache-Control"]
    tile = np.frombuffer(r.data, "<f4").reshape(4, 4)
    assert tile.min() >= 1.0 and tile.max() <= 2.0       # file units, not the gauge's
    assert client.get("/api/healpix/dust/tile/1/0.f32").status_code == 404
    assert client.get("/api/healpix/dust/tile/0/12.f32").status_code == 404
