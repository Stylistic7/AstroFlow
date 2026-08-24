"""Tests for the gnomonic (TAN) projection and the Target/mosaic branch.

A projection bug here does not raise. The two properties that would otherwise
fail silently:

  * `project` and `unproject` are exact inverses. This is the one that governs
    CORRECTNESS: `place_tile` inverse-maps every output pixel through
    `unproject` and then through the source WCS, so a broken inverse smears the
    data. (A wrong ROTATION, by contrast, still places every pixel correctly —
    it only costs framing, since any bijection round-trips.)
  * a field at position angle theta lands AXIS-ALIGNED, measured as polygon
    area / bounding-box area == 1. This is what removes the wedge corners and
    the oversampled canvas, and it is what settled the rotation's sign.
"""
import numpy as np
import pytest
from astropy.wcs import WCS

from geometry.projections.gnomonic import GnomonicProjection
from geometry.projections.mercator import MercatorProjection
from plate_solving import get_rotation_angle


def _tan_wcs(ra, dec, scale_arcsec, nx, ny, pa_deg):
    """A TAN WCS with the CD matrix this library's solver actually produces.

    Every one of the 1,589 plate solutions on disk has

        CD = s * [[ cos t, sin t],
                  [-sin t, cos t]]        (determinant +s^2, i.e. POSITIVE)

    so that `get_rotation_angle` == arctan2(CD1_2, CD1_1) == t. Building the
    opposite parity here instead would flip which sign of `rotation` frames a
    field tightly, and the tests would then happily certify the wrong one —
    which is exactly what happened before this was checked against real data.
    """
    s = scale_arcsec / 3600.0
    t = np.radians(pa_deg)
    header = {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": ra, "CRVAL2": dec,
        "CRPIX1": nx / 2, "CRPIX2": ny / 2,
        "CD1_1": s * np.cos(t), "CD1_2": s * np.sin(t),
        "CD2_1": -s * np.sin(t), "CD2_2": s * np.cos(t),
    }
    return WCS(header), header


def _tightness(xs, ys):
    """Polygon area / bounding-box area. 1.0 means perfectly axis-aligned."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    return area / ((xs.max() - xs.min()) * (ys.max() - ys.min()))


class TestRoundTrip:
    @pytest.mark.parametrize("lat0", [0.0, 45.0, 61.5, -30.0, 85.0])
    @pytest.mark.parametrize("rotation", [0.0, 30.0, 88.8, 150.0, -60.0])
    def test_round_trip_over_a_field(self, lat0, rotation):
        """Every point in a 3-degree field returns to where it started."""
        proj = GnomonicProjection(
            np.radians(150.0), np.radians(lat0), np.radians(rotation)
        )
        # A grid spanning +-1.5 deg, spaced in true angle (RA compressed by
        # sec(lat) so the field is square on the sky, not just in coordinates).
        d = np.radians(1.5)
        dlat = np.linspace(-d, d, 11)
        dlon = np.linspace(-d, d, 11) / max(np.cos(np.radians(lat0)), 1e-6)
        lon = np.radians(150.0) + dlon[None, :]
        lat = np.radians(lat0) + dlat[:, None]
        lon, lat = np.broadcast_arrays(lon, lat)

        lon2, lat2 = proj.unproject(*proj.project(lon, lat))
        assert np.max(np.abs(lat2 - lat)) < 1e-12
        # Compare RA as an angular difference so 359.99 vs -0.01 doesn't count.
        dra = np.abs(np.arctan2(np.sin(lon2 - lon), np.cos(lon2 - lon)))
        assert np.max(dra) < 1e-12

    def test_tangent_point_is_the_origin(self):
        """rho == 0 is the one place the inverse divides by zero."""
        proj = GnomonicProjection(np.radians(150.0), np.radians(61.5), 0.9)
        x, y = proj.project(np.radians(150.0), np.radians(61.5))
        assert abs(float(x)) < 1e-15 and abs(float(y)) < 1e-15
        lon, lat = proj.unproject(0.0, 0.0)
        assert not np.isnan(lon) and not np.isnan(lat)
        assert abs(float(lon) - np.radians(150.0)) < 1e-15
        assert abs(float(lat) - np.radians(61.5)) < 1e-15

    def test_scalars_round_trip(self):
        proj = GnomonicProjection(np.radians(10.0), np.radians(-20.0), 0.3)
        lon, lat = proj.unproject(*proj.project(np.radians(11.0), np.radians(-19.0)))
        assert abs(float(lon) - np.radians(11.0)) < 1e-12
        assert abs(float(lat) - np.radians(-19.0)) < 1e-12


class TestPositionAngle:
    """The rotation sign — the one thing here that measurement had to settle."""

    def test_cd_matrix_parity_matches_the_library(self):
        """If real solutions ever stop being positive-determinant, the sign
        `_projection_for` uses is no longer the right one, and these tests
        would go on passing against a fiction. Pin the assumption itself."""
        _, header = _tan_wcs(150.0, 61.5, 1.0, 4000, 3000, 88.8)
        det = (header["CD1_1"] * header["CD2_2"]
               - header["CD1_2"] * header["CD2_1"])
        assert det > 0
        assert abs(np.degrees(get_rotation_angle(header)) - 88.8) < 1e-9

    @pytest.mark.parametrize("pa_deg", [0.0, 30.0, 88.8, 150.0])
    @pytest.mark.parametrize("dec", [0.0, 61.5, -45.0])
    def test_rotated_field_lands_axis_aligned(self, pa_deg, dec):
        wcs, header = _tan_wcs(150.0, dec, 1.0, 4000, 3000, pa_deg)
        fp = wcs.calc_footprint()
        lon, lat = np.radians(fp[:, 0]), np.radians(fp[:, 1])

        # Exactly how mosaics/viewbox.py builds it: the header angle, NEGATED.
        proj = GnomonicProjection(
            np.radians(150.0), np.radians(dec), -get_rotation_angle(header)
        )
        assert _tightness(*proj.project(lon, lat)) > 0.9999

    @pytest.mark.parametrize("pa_deg", [30.0, 150.0])
    def test_unnegated_rotation_would_be_caught(self, pa_deg):
        """Guards the guard: the tightness assertion has to actually fail on
        the wrong sign, or it certifies nothing. (PA near 0 or 90 can't tell
        the signs apart — they differ by ~180 deg, which preserves alignment —
        which is why the real fields, all near PA 90, only degraded to 0.48 at
        their worst rather than obviously breaking.)"""
        wcs, header = _tan_wcs(150.0, 61.5, 1.0, 4000, 3000, pa_deg)
        fp = wcs.calc_footprint()
        lon, lat = np.radians(fp[:, 0]), np.radians(fp[:, 1])
        proj = GnomonicProjection(
            np.radians(150.0), np.radians(61.5), get_rotation_angle(header)
        )
        assert _tightness(*proj.project(lon, lat)) < 0.6

    def test_beats_mercator_on_a_rotated_field(self):
        """Mercator has no rotation term, so a PA-30 field wastes ~47% of canvas."""
        wcs, _ = _tan_wcs(150.0, 61.5, 1.0, 4000, 3000, 30.0)
        fp = wcs.calc_footprint()
        lon, lat = np.radians(fp[:, 0]), np.radians(fp[:, 1])
        assert _tightness(*MercatorProjection().project(lon, lat)) < 0.6


class TestUniformScale:
    def test_no_taper_across_a_high_declination_field(self):
        """Mercator tapers the Heart's field by 8.6%; TAN doesn't.

        The field has to be a REAL rectangle — one the camera could have shot,
        i.e. a rectangle in some tangent plane — not a constant-delta-lon
        trapezoid, which is genuinely narrower at the top on the sky itself and
        so makes every projection look like it tapers.

        Take the top and bottom edges of that rectangle and compare their
        projected lengths. On the sensor they are equal by construction.
        """
        dec = 61.5
        # 4000 x 3000 at 3"/px == 3.3 x 2.5 degrees, the Heart's extent.
        wcs, _ = _tan_wcs(150.0, dec, 3.0, 4000, 3000, pa_deg=0.0)
        fp = wcs.calc_footprint()
        lon, lat = np.radians(fp[:, 0]), np.radians(fp[:, 1])

        def edge_length_ratio(proj):
            xs, ys = (np.asarray(v) for v in proj.project(lon, lat))
            # Bottom edge = the two lowest corners, top = the two highest.
            order = np.argsort(ys)
            bottom = abs(xs[order[0]] - xs[order[1]])
            top = abs(xs[order[2]] - xs[order[3]])
            return max(top, bottom) / min(top, bottom)

        gnomonic = GnomonicProjection(np.radians(150.0), np.radians(dec), 0.0)
        assert edge_length_ratio(gnomonic) < 1.001
        assert edge_length_ratio(MercatorProjection()) > 1.08


class TestCanvasSize:
    def test_high_declination_canvas_stops_being_oversampled(self):
        """The reason this change is not merely cosmetic.

        `compute_viewbox` divides the PROJECTED span by the field's true
        angular resolution. Mercator's local scale is sec(delta), so at the
        Heart's +61.5 that inflates the canvas by sec^2 = 4.4x in area — the
        measured figure on the real masters is 4.66-4.77x. Every downstream
        product is that much bigger, carrying no extra detail: it is the same
        image upsampled 2.1x per axis.
        """
        dec = 61.5
        wcs, header = _tan_wcs(150.0, dec, 3.0, 4000, 3000, pa_deg=90.0)
        fp = wcs.calc_footprint()
        lon, lat = np.radians(fp[:, 0]), np.radians(fp[:, 1])
        res = np.radians(3.0 / 3600)
        native = 4000 * 3000

        def canvas_pixels(proj):
            xs, ys = (np.asarray(v) for v in proj.project(lon, lat))
            return ((xs.max() - xs.min()) / res) * ((ys.max() - ys.min()) / res)

        gnomonic = GnomonicProjection(
            np.radians(150.0), np.radians(dec), -get_rotation_angle(header))
        assert canvas_pixels(gnomonic) / native < 1.02
        assert canvas_pixels(MercatorProjection()) / native > 4.0


class TestEndToEndPlacement:
    """Push a real image through the real `place_tile` into a real viewbox.

    Everything above tests geometry in the abstract. This tests that the
    pipeline actually reproduces the frame the camera shot — right side up,
    right way round, filling the canvas.
    """

    RA, DEC, PA, SCALE = 150.0, 61.5, 88.8, 3.0
    NX, NY, TILE = 512, 384, 128

    def _placed(self, projection):
        from scipy.interpolate import RegularGridInterpolator
        from spherical_geometry.polygon import SingleSphericalPolygon
        from geometry.polygons import project_polygon
        from mosaics.types import Viewbox
        from mosaics.placement import place_tile

        nx, ny, tile = self.NX, self.NY, self.TILE
        wcs, _ = _tan_wcs(self.RA, self.DEC, self.SCALE, nx, ny, self.PA)

        # Structure in both axes, plus a bright stripe on the TOP edge and a
        # different one on the LEFT edge — so a flip, mirror or transpose shows.
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
        image = np.sin(xx / 21.0) * np.cos(yy / 17.0) + 2.0
        image[:12, :] = 5.0
        image[:, :12] = 4.0
        interp = RegularGridInterpolator(
            (np.arange(ny, dtype=float), np.arange(nx, dtype=float)), image,
            method="linear", bounds_error=False, fill_value=0)

        fp = SingleSphericalPolygon.from_wcs(wcs, steps=16)
        xs, ys = project_polygon(fp, projection)
        res = np.radians(self.SCALE / 3600)
        x_min, y_min = min(xs), min(ys)
        x_span, y_span = max(xs) - x_min, max(ys) - y_min
        npx, npy = int(np.ceil(x_span / res)), int(np.ceil(y_span / res))
        ntx, nty = int(np.ceil(npx / tile)), int(np.ceil(npy / tile))
        vb = Viewbox(projection=projection, x_min=x_min, x_span=x_span,
                     y_min=y_min, y_span=y_span, resolution=res,
                     num_pixels_x=npx, num_pixels_y=npy)

        # Chunk size is a placement detail, not viewbox geometry. Shrink it so
        # a 512x384 frame spans several chunks; patched rather than passed
        # because the fixtures here are class-scoped and monkeypatch is not.
        from unittest.mock import patch
        import mosaics.placement.astroflow as _placement

        canvas = np.zeros((nty * tile, ntx * tile), np.float32)
        with patch.object(_placement, "_TILE_SIZE", tile):
            for ty in range(nty):
                for tx in range(ntx):
                    _, _, data = place_tile(vb, ty, tx, wcs, interp, False)
                    canvas[ty*tile:(ty+1)*tile, tx*tile:(tx+1)*tile] = data
        return canvas[:npy, :npx], image

    @pytest.fixture(scope="class")
    def gnomonic_canvas(self):
        _, header = _tan_wcs(self.RA, self.DEC, self.SCALE,
                             self.NX, self.NY, self.PA)
        return self._placed(GnomonicProjection(
            np.radians(self.RA), np.radians(self.DEC),
            -get_rotation_angle(header)))

    def test_canvas_matches_the_sensor_and_is_nearly_all_data(
            self, gnomonic_canvas):
        canvas, _ = gnomonic_canvas
        h, w = canvas.shape
        assert abs(w - self.NX) <= 2 and abs(h - self.NY) <= 2
        assert (canvas > 0).mean() > 0.98, "canvas should be data, not wedges"

    def test_pixel_values_survive_the_round_trip(self, gnomonic_canvas):
        canvas, image = gnomonic_canvas
        interior = canvas[8:-8, 8:-8]
        lit = interior[interior > 0]
        assert abs(lit.max() - image.max()) < 0.01
        assert abs(lit.mean() - image.mean()) < 0.15

    def test_orientation_is_preserved(self, gnomonic_canvas):
        """The sensor's top edge is the canvas's top edge, and its left edge
        the canvas's left — no flip, mirror or transpose."""
        canvas, _ = gnomonic_canvas
        h, w = canvas.shape

        rows, cols = np.where(np.isclose(canvas, 5.0, atol=0.05))  # top stripe
        assert rows.max() < h * 0.1, "top-edge stripe did not land at the top"
        assert cols.max() > w * 0.9, "top-edge stripe did not span the width"

        rows, cols = np.where(np.isclose(canvas, 4.0, atol=0.05))  # left stripe
        assert cols.max() < w * 0.1, "left-edge stripe did not land at the left"
        assert rows.max() > h * 0.9, "left-edge stripe did not span the height"

    def test_mercator_would_be_oversampled_and_rotated(self):
        """The before picture, and what is actually wrong with it.

        At this field's PA of 88.8 the canvas is NOT wedge-cornered: rotating a
        rectangle by ~90 degrees gives a transposed rectangle, which bounds
        tightly, so ~95% of the Mercator canvas still carries data. The costs
        here are that it holds 4.6x the pixels for the same sky (sec^2 of
        delta +61.5), and that a landscape sensor comes out portrait.

        Wedge corners are real but they are an intermediate-PA effect — see
        `test_beats_mercator_on_a_rotated_field`, where PA 30 drops Mercator's
        bounding-box fill to 0.53.
        """
        canvas, _ = self._placed(MercatorProjection())
        h, w = canvas.shape
        assert h * w > 4.0 * self.NX * self.NY
        assert h > w, "landscape sensor came out portrait under north-up Mercator"
        assert (canvas > 0).mean() > 0.9, "at PA ~90 the waste is scale, not shape"


class TestViewboxSelection:
    """Targets get gnomonic; real mosaics stay on Mercator."""

    def test_mosaic_keeps_mercator(self):
        from mosaics.viewbox import _projection_for
        from attributes import AstrometricAttributes

        # A bare AstrometricAttributes stands in for anything that isn't a
        # Target — `_projection_for` branches on the type, nothing else.
        attrs = object.__new__(AstrometricAttributes)
        assert isinstance(_projection_for(attrs), MercatorProjection)


class TestHashability:
    def test_repr_carries_every_parameter(self):
        """The Viewbox cache key is built from this repr (hashing.hash_list):
        a parameter missing from it makes two different projections collide."""
        a = GnomonicProjection(1.0, 0.5, 0.25)
        for other in (
            GnomonicProjection(2.0, 0.5, 0.25),
            GnomonicProjection(1.0, 0.6, 0.25),
            GnomonicProjection(1.0, 0.5, 0.30),
        ):
            assert repr(a) != repr(other)
        assert " at 0x" not in repr(a), "address-based repr flaps the cache key"
