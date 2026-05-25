"""Unit tests for mosaic placement — tile projection and coordinate transforms."""
import numpy as np
import pytest
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

from geometry.projections.mercator import MercatorProjection
from geometry.viewbox import compute_viewbox as geometry_compute_viewbox
from geometry.polygons import project_polygon
from mosaics.types import Viewbox
from spherical_geometry.polygon import SingleSphericalPolygon


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_tan_wcs(
    ra_deg: float, dec_deg: float, scale_arcsec: float, nx: int, ny: int,
) -> WCS:
    """Create a simple TAN (gnomonic) WCS centered at (ra, dec)."""
    cdelt = scale_arcsec / 3600.0  # degrees per pixel
    header = {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": ra_deg,
        "CRVAL2": dec_deg,
        "CRPIX1": nx / 2,
        "CRPIX2": ny / 2,
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }
    return WCS(header)


def _make_viewbox_for_wcs(wcs: WCS, nx: int, ny: int, tile_size: int = 256) -> Viewbox:
    """Build a Viewbox that covers the same sky region as the WCS."""
    projection = MercatorProjection()

    # Get the WCS footprint corners and project them.
    footprint = wcs.calc_footprint()  # (4, 2) array of (ra, dec) in degrees
    ras_rad = np.radians(footprint[:, 0])
    decs_rad = np.radians(footprint[:, 1])
    xs, ys = projection.project(ras_rad, decs_rad)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Use same pixel scale as the source.
    resolution = x_span / nx

    num_pixels_x = nx
    num_pixels_y = int(np.ceil(y_span / resolution))
    num_tiles_x = int(np.ceil(num_pixels_x / tile_size))
    num_tiles_y = int(np.ceil(num_pixels_y / tile_size))

    return Viewbox(
        projection=projection,
        x_min=x_min,
        x_span=x_span,
        y_min=y_min,
        y_span=y_span,
        resolution=resolution,
        num_pixels_x=num_pixels_x,
        num_pixels_y=num_pixels_y,
        num_tiles_x=num_tiles_x,
        num_tiles_y=num_tiles_y,
        tile_width=tile_size,
        tile_height=tile_size,
    )


def _make_interpolator(image: np.ndarray) -> RegularGridInterpolator:
    """Create a RegularGridInterpolator for a 2D image."""
    ny, nx = image.shape
    rows = np.arange(ny, dtype=np.float64)
    cols = np.arange(nx, dtype=np.float64)
    return RegularGridInterpolator(
        (rows, cols), image.astype(np.float64),
        method="linear", bounds_error=False, fill_value=0,
    )


# ---------------------------------------------------------------------------
# place_tile tests
# ---------------------------------------------------------------------------

from mosaics.placement import place_tile


def _call_place_tile(viewbox, tile_y, tile_x, wcs, interp, skip_empty=False):
    return place_tile(viewbox, tile_y, tile_x, wcs, interp, skip_empty)


class TestPlaceTile:
    """Tests for the core tile projection function."""

    @pytest.fixture
    def setup(self):
        """Create a 100x100 source image with a gradient, matching WCS and viewbox."""
        nx, ny = 100, 100
        # Gradient image: pixel value = row index (0-99).
        image = np.tile(np.arange(ny, dtype=np.float64), (nx, 1)).T

        ra, dec = 100.0, 0.0
        scale = 10.0  # 10 arcsec/pixel — a wide-ish field for a small image

        wcs = _make_tan_wcs(ra, dec, scale, nx, ny)
        interp = _make_interpolator(image)
        viewbox = _make_viewbox_for_wcs(wcs, nx, ny, tile_size=128)

        return image, wcs, interp, viewbox

    def test_output_shape(self, setup):
        _, wcs, interp, viewbox = setup
        ty, tx, data = _call_place_tile(viewbox, 0, 0, wcs, interp)
        assert data is not None
        assert data.shape == (viewbox.tile_height, viewbox.tile_width)
        assert data.dtype == np.float32

    def test_has_nonzero_pixels(self, setup):
        _, wcs, interp, viewbox = setup
        _, _, data = _call_place_tile(viewbox, 0, 0, wcs, interp)
        assert np.any(data > 0), "Tile should contain non-zero pixels from source image"

    def test_center_pixel_has_expected_value(self, setup):
        image, wcs, interp, viewbox = setup
        _, _, data = _call_place_tile(viewbox, 0, 0, wcs, interp)
        # The center of the source image (row ~50) should map to somewhere in
        # the tile with a value near 50 (the gradient value at the center row).
        nonzero = data[data > 0]
        assert len(nonzero) > 0
        # The gradient ranges 0-99; the center region should produce values near 50.
        center_values = nonzero[(nonzero > 30) & (nonzero < 70)]
        assert len(center_values) > 0, "Expected some pixels near the center gradient value"

    def test_tile_outside_footprint_is_zeros(self, setup):
        _, wcs, interp, viewbox = setup
        # Tile far from the image — use a large tile index that's way off.
        # The viewbox might only have 1 tile, so create a viewbox with more tiles.
        far_viewbox = Viewbox(
            projection=viewbox.projection,
            x_min=viewbox.x_min + 1.0,  # shift 1 radian (~57 degrees) away
            x_span=viewbox.x_span,
            y_min=viewbox.y_min,
            y_span=viewbox.y_span,
            resolution=viewbox.resolution,
            num_pixels_x=viewbox.num_pixels_x,
            num_pixels_y=viewbox.num_pixels_y,
            num_tiles_x=viewbox.num_tiles_x,
            num_tiles_y=viewbox.num_tiles_y,
            tile_width=viewbox.tile_width,
            tile_height=viewbox.tile_height,
        )
        _, _, data = _call_place_tile(far_viewbox, 0, 0, wcs, interp)
        assert data is not None
        assert np.all(data == 0), "Tile far from source should be all zeros"

    def test_skip_empty_returns_none_for_distant_tile(self, setup):
        _, wcs, interp, viewbox = setup
        far_viewbox = Viewbox(
            projection=viewbox.projection,
            x_min=viewbox.x_min + 1.0,
            x_span=viewbox.x_span,
            y_min=viewbox.y_min,
            y_span=viewbox.y_span,
            resolution=viewbox.resolution,
            num_pixels_x=viewbox.num_pixels_x,
            num_pixels_y=viewbox.num_pixels_y,
            num_tiles_x=viewbox.num_tiles_x,
            num_tiles_y=viewbox.num_tiles_y,
            tile_width=viewbox.tile_width,
            tile_height=viewbox.tile_height,
        )
        _, _, data = _call_place_tile(far_viewbox, 0, 0, wcs, interp, skip_empty=True)
        assert data is None, "skip_empty should return None for off-footprint tile"

    def test_skip_empty_returns_data_when_corners_in_footprint(self):
        """Use a large source image so tile corners fall inside the WCS footprint."""
        nx, ny = 1000, 1000
        image = np.ones((ny, nx), dtype=np.float64)
        wcs = _make_tan_wcs(100.0, 0.0, 10.0, nx, ny)
        interp = _make_interpolator(image)
        # Small tile (128px) well inside a large source (1000px) — corners are in footprint.
        viewbox = _make_viewbox_for_wcs(wcs, nx, ny, tile_size=128)
        # Use a central tile.
        center_ty = viewbox.num_tiles_y // 2
        center_tx = viewbox.num_tiles_x // 2
        _, _, data = _call_place_tile(viewbox, center_ty, center_tx, wcs, interp, skip_empty=True)
        assert data is not None, "Central tile corners should be inside footprint"


class TestCoordinateRoundTrip:
    """Test the coordinate chain: pixel → WCS → Mercator → back."""

    def test_mercator_round_trip_equator(self):
        proj = MercatorProjection()
        lon, lat = np.radians(100.0), np.radians(0.0)
        x, y = proj.project(lon, lat)
        lon2, lat2 = proj.unproject(x, y)
        assert abs(lon - lon2) < 1e-12
        assert abs(lat - lat2) < 1e-12

    def test_mercator_round_trip_midlatitude(self):
        proj = MercatorProjection()
        lon, lat = np.radians(200.0), np.radians(45.0)
        x, y = proj.project(lon, lat)
        lon2, lat2 = proj.unproject(x, y)
        assert abs(lon - lon2) < 1e-12
        assert abs(lat - lat2) < 1e-10

    def test_wcs_center_maps_to_viewbox_center(self):
        """The center of the source image should map near the center of the viewbox."""
        nx, ny = 100, 100
        ra, dec = 100.0, 10.0
        wcs = _make_tan_wcs(ra, dec, 10.0, nx, ny)
        viewbox = _make_viewbox_for_wcs(wcs, nx, ny, tile_size=128)

        # WCS center pixel → RA/Dec
        center_ra, center_dec = wcs.all_pix2world(nx / 2, ny / 2, 0)

        # RA/Dec → Mercator
        proj = viewbox.projection
        cx, cy = proj.project(np.radians(float(center_ra)), np.radians(float(center_dec)))

        # Should be near the center of the viewbox.
        vb_center_x = viewbox.x_min + viewbox.x_span / 2
        vb_center_y = viewbox.y_min + viewbox.y_span / 2

        assert abs(cx - vb_center_x) < viewbox.resolution * 2
        assert abs(cy - vb_center_y) < viewbox.resolution * 2
