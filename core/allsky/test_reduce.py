"""The combiner on small synthetic masters: the HEALPix inverse against the
astropy-healpix fixture, a whole reduce whose overlap, single coverage,
noise, count and pyramid come out right, the sky-offset solve, and proper
coaddition itself: one master comes back as itself, two resolutions give the
predicted PSF and conserve flux, and a master added later lands the same
tile as a fresh build. Pure numpy + astropy + scipy; no app, no Ray."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from allsky import reduce as rd
from allsky.healpix_faces import compact_bits, xyf_from_vec
from allsky.hips import read_properties

TESTDATA = Path(__file__).parent / "testdata"
PSF = 60.0                        # arcsec: the test masters' PSF unless a sidecar says otherwise


def test_vec_from_xyf_hits_astropy_cell_centres_and_round_trips():
    fx = json.loads((TESTDATA / "healpix_reference.json").read_text())
    nside = fx["nside"]
    order = int(np.log2(nside))
    for npix, lon, lat in fx["centres"]:
        face = npix >> (2 * order)
        within = np.array([npix & (nside * nside - 1)], np.uint64)
        ix, iy = int(compact_bits(within)[0]), int(compact_bits(within >> np.uint64(1))[0])
        got_lon, got_lat = rd.lonlat_of(rd.vec_from_xyf([face], [(ix + 0.5) / nside], [(iy + 0.5) / nside])[0])
        assert abs(((got_lon - lon + 180) % 360) - 180) < 1e-8 and abs(got_lat - lat) < 1e-8, npix
    rng = np.random.default_rng(3)
    v = rng.normal(size=(2000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    face, x, y = xyf_from_vec(v)
    back = rd.vec_from_xyf(face, x, y)
    assert np.abs(back - v).max() < 1e-9


def test_vec_from_xyf_continues_past_the_face_edge():
    """A tile's margin reaches past its face: the mapping must carry on
    smoothly there (the step across the edge equals the step inside it)."""
    eps = 1e-3
    for face in range(12):
        for t in (0.1, 0.5, 0.9):
            for inside, across, deeper in (
                ((1 - eps, t), (1 + eps, t), (1 - 3 * eps, t)), ((eps, t), (-eps, t), (3 * eps, t)),
                ((t, 1 - eps), (t, 1 + eps), (t, 1 - 3 * eps)), ((t, eps), (t, -eps), (t, 3 * eps)),
            ):
                a, b, c = (rd.vec_from_xyf([face], [p[0]], [p[1]])[0] for p in (inside, across, deeper))
                step_across = np.arccos(np.clip(a @ b, -1, 1))
                step_inside = np.arccos(np.clip(a @ c, -1, 1))
                assert abs(step_across / step_inside - 1) < 0.05, (face, inside)


def _write_master(path, ra0, dec0, value, nx=160, ny=120, scale_arcsec=30.0, sigma=2.0, seed=0, sky=0.0,
                  fwhm_arcsec=None, stars=(), blob=None):
    """A flat field `value + sky` with noise `sigma`; optionally stars
    (x, y, flux) rendered with the PSF, a smooth blob (x, y, amplitude,
    sigma_px), and a sidecar `analysis` carrying the PSF and the noise."""
    rng = np.random.default_rng(seed)
    data = value + sky + rng.normal(0, sigma, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    if blob is not None:
        bx, by, amp, bs = blob
        data += amp * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * bs * bs))
    if stars:
        s = fwhm_arcsec / scale_arcsec * rd.FWHM_TO_SIGMA
        for sx, sy, flux in stars:
            data += flux / (2 * np.pi * s * s) * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * s * s))
    data = data.astype(np.float32)
    data[:4, :] = np.nan                                  # a dead strip
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRVAL1"], hdr["CRVAL2"] = ra0, dec0
    hdr["CRPIX1"], hdr["CRPIX2"] = nx / 2 + 0.5, ny / 2 + 0.5
    hdr["CD1_1"], hdr["CD2_2"] = -scale_arcsec / 3600, scale_arcsec / 3600
    hdr["CD1_2"] = hdr["CD2_1"] = 0.0
    hdr["EQUINOX"] = 2000.0
    fits.PrimaryHDU(data, hdr).writeto(path)
    if fwhm_arcsec is not None:
        from file_io.sidecar import set_section
        set_section(Path(path), "analysis", {"noise_px": float(sigma), "psf": {"fwhm_arcsec": float(fwhm_arcsec)}})


def _finest_tiles(out, order):
    return sorted(int(p.stem[4:]) for p in (out / "mean" / f"Norder{order}").rglob("Npix*.fits"))


def test_two_overlapping_masters_reduce_to_a_hips_tree(tmp_path):
    src = tmp_path / "masters"
    src.mkdir()
    # Two 1.3° × 1° fields, 30″/px, overlapping by half a field in RA.
    _write_master(src / "a.fits", 30.0, 10.0, 100.0, seed=1)
    _write_master(src / "b.fits", 30.7, 10.0, 200.0, seed=2)
    out = tmp_path / "sky"
    width = 64
    info = rd.reduce_folder(src, out, order=None, width=width, workers=2, level=False, psf_fwhm=PSF, log=lambda s: None)
    # 30″ pixels → cells of ~26″ (pixel order 13) → tile order 13 − 6 = 7 with 64-px tiles.
    assert info["order"] == rd.order_for(30.0, width) == 7
    assert info["masters"] == 2 and 12 <= info["tiles"] <= 24     # ~9 tiles a field, shared overlap
    props = read_properties((out / "mean" / "properties").read_text())
    assert (props["hips_order"], props["hips_tile_width"], props["hips_frame"], props["hips_tile_format"]) == ("7", "64", "equatorial", "fits")
    # The noise tree is half width; the count tree is 16-bit integers.
    assert read_properties((out / "noise" / "properties").read_text())["hips_tile_width"] == "32"
    assert read_properties((out / "count" / "properties").read_text())["hips_pixel_bitpix"] == "16"
    area = (30.0 ** 2)                                    # file units are per pixel; served per arcsec²
    mean, noise, count = {}, {}, {}
    for npix in _finest_tiles(out, 7):
        mean[npix] = rd.read_tile(out / "mean", 7, npix)
        noise[npix] = rd.read_tile(out / "noise", 7, npix)
        count[npix] = rd.read_tile(out / "count", 7, npix)
        assert fits.getheader(out / "count" / rd.tile_path(7, npix))["BITPIX"] == 16
        assert noise[npix].shape == (width // 2, width // 2)
        assert fits.getheader(out / "mean" / rd.tile_path(7, npix))["PSFFWHM"] > PSF    # the pixel widens it
    counts = np.concatenate([c.ravel() for c in count.values()])
    assert set(np.unique(counts)) == {0.0, 1.0, 2.0}
    both = np.concatenate([m[c == 2] for m, c in zip(mean.values(), count.values())])
    only = np.concatenate([m[c == 1] for m, c in zip(mean.values(), count.values())])
    assert both.size > 500 and only.size > 500
    # Equal PSF and noise → the overlap is the plain mean of 100 and 200 per pixel; one master → its own level.
    assert np.nanmean(both) == pytest.approx(150.0 / area, rel=0.02)
    lo, hi = only[only < 150 / area], only[only >= 150 / area]
    assert np.nanmean(lo) == pytest.approx(100.0 / area, rel=0.03)
    assert np.nanmean(hi) == pytest.approx(200.0 / area, rel=0.03)
    # Noise = 1/√Σw: σ per arcsec² alone, σ/√2 where both contribute (its
    # half-width cells are judged by the count of their four finest cells).
    n2 = np.concatenate([n[rd._degrade(c) == 2] for n, c in zip(noise.values(), count.values())])
    n1 = np.concatenate([n[rd._degrade(c) == 1] for n, c in zip(noise.values(), count.values())])
    assert np.nanmedian(n1) == pytest.approx(2.0 / area, rel=0.15)
    assert np.nanmedian(n2) == pytest.approx(2.0 / area / np.sqrt(2), rel=0.15)
    # The pyramid reaches order 0 and a parent is the mean of its children.
    for k in range(7):
        assert any((out / "mean" / f"Norder{k}").rglob("Npix*.fits")), k
    child = max(mean, key=lambda n: np.isfinite(mean[n]).sum())
    parent = rd.read_tile(out / "mean", 6, child >> 2)
    i = child & 3
    half = width // 2
    block = parent[((i >> 1) & 1) * half:((i >> 1) & 1) * half + half, (i & 1) * half:(i & 1) * half + half]
    assert np.allclose(np.nan_to_num(block), np.nan_to_num(rd._degrade(mean[child])), atol=1e-6)


def test_sky_offsets_are_solved_from_overlaps_and_the_map_is_seamless(tmp_path):
    src = tmp_path / "masters"
    src.mkdir()
    # Four fields of the same 100-unit sky: a row a-b-c overlapping in RA with
    # foregrounds +20 and −10 on b and c, and a lone field d with +50.
    _write_master(src / "a.fits", 30.0, 10.0, 100.0, seed=1)
    _write_master(src / "b.fits", 30.7, 10.0, 100.0, seed=2, sky=20.0)
    _write_master(src / "c.fits", 31.4, 10.0, 100.0, seed=3, sky=-10.0)
    _write_master(src / "d.fits", 80.0, 70.0, 100.0, seed=4, sky=50.0)      # in the polar cap
    out = tmp_path / "sky"
    width = 64
    area = 30.0 ** 2
    info = rd.reduce_folder(src, out, width=width, workers=2, psf_fwhm=PSF, log=lambda s: None)
    # Every cell under a footprint is reduced, corners and polar-cap rhombi
    # included: the lone field's covered area is its own area.
    cell_deg2 = (rd.CELL_ARCSEC_NSIDE1 / (width << info["order"]) / 3600.0) ** 2
    covered = 0
    for p in (out / "count" / f"Norder{info['order']}").rglob("Npix*.fits"):
        c = rd.read_tile(out / "count", info["order"], int(p.stem[4:]))
        m = rd.read_tile(out / "mean", info["order"], int(p.stem[4:]))
        covered += int(((c > 0) & (m * area > 125)).sum())             # d's cells carry its +50
    # (116 live rows: the dead strip is four of 120; bilinear sampling loses half a pixel at the rim.)
    assert covered * cell_deg2 == pytest.approx(160 * 116 * (30.0 / 3600) ** 2, rel=0.03)
    off = {k.split(":")[0]: v * area for k, v in info["offsets"].items()}     # back to file units
    assert off["a.fits"] == 0.0                                 # the reference of its group
    assert off["b.fits"] == pytest.approx(20.0, abs=0.5)
    assert off["c.fits"] == pytest.approx(-10.0, abs=0.5)
    assert off["d.fits"] == 0.0                                 # no neighbour: its own level
    # Every covered cell of the row now sits at 100 (per arcsec²), seams included.
    vals = []
    for npix in _finest_tiles(out, 7):
        m, c = rd.read_tile(out / "mean", 7, npix), rd.read_tile(out / "count", 7, npix)
        vals.append(m[c > 0] * area)
    vals = np.concatenate(vals)
    row = vals[vals < 125]                                      # d's cells carry its +50
    assert row.size > 2000 and np.percentile(row, 5) > 90 and np.percentile(row, 95) < 110
    assert np.median(row) == pytest.approx(100.0, rel=0.02)
    # The state carries the pairs and the offsets, and a rerun has nothing to do.
    state = json.loads((out / "state.json").read_text())
    assert len(state["pairs"]) >= 2 and len(state["offsets"]) == 4
    again = rd.reduce_folder(src, out, width=width, workers=2, psf_fwhm=PSF, log=lambda s: None)
    assert again["changed"] == 0 and again["offsets"] == info["offsets"]


def test_one_master_comes_back_as_itself(tmp_path):
    """Proper coaddition of a single image is that image: the matched filter
    and the whitening cancel exactly. A smooth scene, compared cell by cell
    with the master sampled directly, away from its footprint edge."""
    src = tmp_path / "masters"
    src.mkdir()
    _write_master(src / "a.fits", 30.0, 10.0, 100.0, sigma=0.01, seed=1, fwhm_arcsec=PSF, blob=(90, 60, 50.0, 20.0))
    out = tmp_path / "sky"
    width = 64
    info = rd.reduce_folder(src, out, width=width, workers=2, level=False, log=lambda s: None)
    m = rd.open_master(src / "a.fits")
    from scipy.ndimage import binary_erosion
    checked = 0
    for npix in _finest_tiles(out, info["order"]):
        coadd = rd.read_tile(out / "mean", info["order"], npix)
        direct = rd.sample_master(m, rd.tile_cell_vectors(info["order"], npix, width, 1)).reshape(width, width)
        interior = binary_erosion(np.isfinite(direct), iterations=8)
        if interior.sum() < 100:
            continue
        checked += interior.sum()
        assert np.abs(coadd[interior] - direct[interior]).max() < 0.01 * 50.0 / 30.0 ** 2     # 1% of the blob
    assert checked > 5000


def _star_profile(mean, sky, cell_arcsec):
    """FWHM (arcsec) of the brightest star in a tile from its peak cell's
    row and column, and its flux (sum above the sky × cell area)."""
    iy, ix = np.unravel_index(np.nanargmax(mean), mean.shape)
    widths = []
    for line, c in ((mean[iy, :] - sky, ix), (mean[:, ix] - sky, iy)):
        peak = line[c]
        for direction in (1, -1):
            j = c
            while 0 <= j + direction < line.size and line[j + direction] > peak / 2:
                j += direction
            k = j + direction
            frac = (line[j] - peak / 2) / (line[j] - line[k])
            widths.append(abs(j + direction * frac - c))
    r = 12
    patch = mean[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] - sky
    return 2 * np.mean(widths) * cell_arcsec, float(np.nansum(patch)) * cell_arcsec ** 2, (iy, ix)


def test_two_resolutions_give_the_predicted_psf_and_conserve_flux(tmp_path):
    """A fine, sharp master and a coarse, blurrier one of the same star:
    the coadd's PSF is the quadrature-weighted mean written to the tile
    (not the plain weighted mean of widths), the star keeps its flux, and
    adding the second master later lands the same tile as a fresh build."""
    width = 64
    sky, flux = 100.0, 4.0e6                                 # file units × pixel; the star is flux / (2πσ²) at its peak
    centre = (30.0, 10.0)
    # stars at the field centre and off it: at least one falls well inside a tile
    stars = [(0.0, 0.0), (6.0, 4.0), (-7.0, -3.0), (5.0, -6.0), (-4.0, 6.0)]   # offsets in arcmin
    fine = dict(scale_arcsec=10.0, nx=240, ny=180, fwhm_arcsec=20.0, sigma=0.5)
    coarse = dict(scale_arcsec=30.0, nx=80, ny=60, fwhm_arcsec=60.0, sigma=0.5)
    def positions(cfg):
        # the same sky positions on each grid (RA grows leftwards: CD1_1 < 0) and the same flux
        return [(cfg["nx"] / 2 - 0.5 - dx * 60 / cfg["scale_arcsec"], cfg["ny"] / 2 - 0.5 + dy * 60 / cfg["scale_arcsec"], flux)
                for dx, dy in stars]
    def build(folder, which):
        folder.mkdir(exist_ok=True)
        if "fine" in which:
            _write_master(folder / "fine.fits", *centre, sky, seed=1, stars=positions(fine), **fine)
        if "coarse" in which:
            _write_master(folder / "coarse.fits", *centre, sky, seed=2, stars=positions(coarse), **coarse)
    both = tmp_path / "both"
    build(both, ("fine", "coarse"))
    out = tmp_path / "sky"
    info = rd.reduce_folder(both, out, width=width, workers=2, level=False, log=lambda s: None)
    order = info["order"]
    cell = rd.CELL_ARCSEC_NSIDE1 / (width << order)
    a, b = rd.open_master(both / "fine.fits"), rd.open_master(both / "coarse.fits")
    # the coarse master has the larger weight per arcsec² (bigger pixels, same per-pixel noise)
    assert b.weight() > a.weight()
    predicted = rd.coadd_fwhm_cells([(a.weight(), a.psf_sigma_cells(cell, 1)), (b.weight(), b.psf_sigma_cells(cell, 1))]) * cell
    assert 20.0 < predicted < 60.0
    best = None
    for npix in _finest_tiles(out, order):
        mean = rd.read_tile(out / "mean", order, npix)
        if not np.isfinite(mean).any() or np.nanmax(mean) < (sky + 2.0) / 100.0:
            continue
        fwhm, got_flux, (iy, ix) = _star_profile(mean, sky / 100.0, cell)
        if 14 <= iy < width - 14 and 14 <= ix < width - 14:
            assert fits.getheader(out / "mean" / rd.tile_path(order, npix))["PSFFWHM"] == pytest.approx(predicted, rel=0.02)
            best = (fwhm, got_flux)
            break
    assert best is not None, "no star fell inside a tile"
    fwhm, got_flux = best
    assert fwhm == pytest.approx(predicted, rel=0.12)
    assert got_flux == pytest.approx(flux, rel=0.05)
    # Incremental: the fine master alone, then the coarse one arrives.
    one = tmp_path / "one"
    build(one, ("fine",))
    inc = tmp_path / "sky_inc"
    rd.reduce_folder(one, inc, width=width, workers=2, level=False, log=lambda s: None)
    shutil.copy(both / "coarse.fits", one / "coarse.fits")
    shutil.copy(both / "coarse.fits.json", one / "coarse.fits.json")
    again = rd.reduce_folder(one, inc, width=width, workers=2, level=False, log=lambda s: None)
    assert again["changed"] > 0
    assert _finest_tiles(inc, order) == _finest_tiles(out, order)
    for npix in _finest_tiles(out, order):
        assert np.allclose(np.nan_to_num(rd.read_tile(inc / "mean", order, npix)),
                           np.nan_to_num(rd.read_tile(out / "mean", order, npix)), rtol=1e-5, atol=1e-7)
