import os
import warnings
import ray
import cv2
import numpy as np
from beartype import beartype
from astropy.wcs import WCS, FITSFixedWarning  # type: ignore

warnings.filterwarnings("ignore", category=FITSFixedWarning)
from scipy.interpolate import RegularGridInterpolator  # type: ignore
from itertools import product
from pathlib import Path
from typing import Any
import structlog
from astropy.coordinates import SkyCoord  # type: ignore
from astropy import units as u  # type: ignore
from dataclasses import dataclass

from stacks.composite_stack import Stack
from targets.types import Target
from file_io.generic import read_image
from mosaics.placement.panel_io import write_placed_panel
from file_io.types import TiffFilePath, WithSidecar
from context import context
# MosaicLike bound as `Mosaic`: widens beartype to accept a Target acting as
# its own single-panel mosaic WITHOUT touching the `mosaic: Mosaic`
# annotations — those serialize into the op code hash, so editing them would
# re-key every op in this module (see mosaics/types.py).
from mosaics.types import MosaicLike as Mosaic, Viewbox
from decorators import node
from decisions.settings import ProcessingConfig
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution

class ProjectionFailedException(Exception):
    pass

_GB = 1024 ** 3

log = structlog.get_logger()

@dataclass(frozen=True)
class PlacedImage:
    pass

def round_to_nearest_multiple(number: float, multiple: int) -> int:
    return int(np.ceil(number / multiple) * multiple)


# The chunk size for projecting a canvas. This is a MEMORY-BOUNDING detail of
# how `place_image` walks the canvas — not a property of the viewbox, and not
# a storage format: the colour tail tiles by PANEL, and the display pyramids
# use their own 256 px tiling in `previews/create_tile_pyramid.py`.
#
# It lived on `Viewbox` as `tile_width`/`tile_height`/`num_tiles_*` until it
# was moved here. That was actively harmful twice over: `Viewbox.__hash__` is
# `hash_dataclass`, so four constants rode along in every cache key, and
# `compute_viewbox` rounded the CANVAS WIDTH up to a multiple of this while
# returning the un-rounded width — which clipped 24.6% off the right edge of
# every master. Canvas geometry has nothing to do with how it gets chunked.
_TILE_SIZE = 2048


def _num_tiles(pixels: int) -> int:
    """Chunks needed to cover `pixels`. The last one overhangs, which is
    fine — `place_image` crops the assembled buffer to the canvas."""
    return int(np.ceil(pixels / _TILE_SIZE))

@beartype
def place_tile(
    viewbox: Viewbox,
    tile_y: int,
    tile_x: int,
    wcs: Any,
    interp: RegularGridInterpolator,
    skip_empty: bool = True,
) -> tuple[int, int, np.ndarray | None]:
    tile_y_min = (
        viewbox.y_min + tile_y * _TILE_SIZE * viewbox.resolution
    )
    tile_y_max = tile_y_min + _TILE_SIZE * viewbox.resolution

    tile_x_min = (
        viewbox.x_min + tile_x * _TILE_SIZE * viewbox.resolution
    )
    tile_x_max = tile_x_min + _TILE_SIZE * viewbox.resolution

    if skip_empty:
        # Compute the four corner points in RA/Dec.
        corners_x = [tile_x_min, tile_x_max, tile_x_min, tile_x_max]
        corners_y = [tile_y_min, tile_y_min, tile_y_max, tile_y_max]
        ras, decs = viewbox.projection.unproject(
            np.array(corners_x), np.array(corners_y)
        )
        ras, decs = np.degrees(ras), np.degrees(decs)

        # If none of the corners of this tile fall inside the source image's
        # footprint, this tile can be skipped.
        inside_footprint = [
            wcs.footprint_contains(
                SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            )
            for ra, dec in zip(ras, decs)
        ]
        if not any(inside_footprint):
            return tile_y, tile_x, None

    # Compute the RA, Dec of all the tile's pixels.
    ys = np.linspace(
        tile_y_min, tile_y_max, _TILE_SIZE, endpoint=False
    )
    xs = np.linspace(tile_x_min, tile_x_max, _TILE_SIZE, endpoint=False)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    ras, decs = viewbox.projection.unproject(xx, yy)
    ras, decs = np.degrees(ras), np.degrees(decs)

    source_xs, source_ys = wcs.all_world2pix(ras, decs, 0, quiet=True)
    tile_data = interp((source_ys, source_xs)).astype(np.float32)
    return tile_y, tile_x, tile_data

def _make_interpolator(
    filename: Path,
) -> tuple[RegularGridInterpolator, tuple[int, int]]:
    # TODO: The subframes need to be calibrated before being placed.
    image = read_image(filename)

    # TODO: This seems to take up a huge amount of memory with method 'cubic.'
    # Maybe tiling would reduce the intermediate memory usage.
    source_height, source_width = image.shape
    interp = RegularGridInterpolator(
        (range(source_height), range(source_width)),
        image,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    return interp, (source_height, source_width)


# Threads for the per-tile projection pool. Capped WELL below the default
# (~cpu+4 = 28): each concurrent tile holds ~700 MB of float64 projection
# temporaries + a scipy RGI query, so the default spiked place_image to ~20 GB
# (measured phys_footprint_peak) and its honest memory reservation serialized it
# to ~2 concurrent Ray tasks. `all_world2pix` is GIL-bound single-core anyway, so
# the pool only parallelizes the interp step — capping to 4 cuts the peak ~7x for
# little wall-time cost, and lets Ray run many more place_image TASKS at once
# (real cross-process parallelism for the single-core transform). Lives in
# _project_in_tiles, not place_image's source, so no cache invalidation.
_PROJECTION_THREADS = 4


def _project_in_tiles(
    viewbox: Viewbox,
    interp: RegularGridInterpolator,
    wcs: WCS,
    tile_range: tuple[int, int, int, int] | None = None,
    skip_empty: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Project the source image into `viewbox`, restricted to the tile
    sub-rectangle `tile_range = (ty0, ty1, tx0, tx1)` (default: the whole
    canvas). Returns `(image, y0_px, x0_px)`: the projected, canvas-clipped
    buffer and its top-left position in full-canvas pixel coordinates.

    Restricting to the footprint's tile bbox is what keeps `place_image`'s RSS
    at bbox size instead of allocating the full ~9 GB canvas: tiles outside the
    range are `place_image`-zero anyway, so skipping them is bit-identical.
    Every surviving `place_tile` call uses the GLOBAL tile indices, so its
    pixels are byte-for-byte the full-canvas result — only the destination
    slice is shifted into the smaller buffer.

    `interp` and `wcs` stay in this worker's Python heap; per-tile work runs on
    a thread pool sharing both references. The interpolation call is numpy-heavy
    and releases the GIL, so threads get real parallelism without copying."""
    from concurrent.futures import ThreadPoolExecutor

    ty0, ty1, tx0, tx1 = tile_range or (
        0, _num_tiles(viewbox.num_pixels_y), 0, _num_tiles(viewbox.num_pixels_x),
    )
    buf_h = (ty1 - ty0) * _TILE_SIZE
    buf_w = (tx1 - tx0) * _TILE_SIZE
    projected_image = np.zeros((buf_h, buf_w), dtype=np.float32)

    tiles = list(product(range(ty0, ty1), range(tx0, tx1)))
    with ThreadPoolExecutor(max_workers=_PROJECTION_THREADS) as ex:
        for tile_y, tile_x, tile_data in ex.map(
            lambda tx: place_tile(
                viewbox, tx[0], tx[1], wcs, interp, skip_empty,
            ),
            tiles,
        ):
            if tile_data is None:
                continue
            ly = (tile_y - ty0) * _TILE_SIZE
            lx = (tile_x - tx0) * _TILE_SIZE
            projected_image[
                ly : ly + _TILE_SIZE,
                lx : lx + _TILE_SIZE,
            ] = tile_data

    # Drop tile-grid padding that spills past the canvas edge (num_pixels is
    # <= num_tiles * tile_size). Matching the old full-canvas slice keeps the
    # eroded canvas-edge behaviour unchanged.
    y0_px, x0_px = ty0 * _TILE_SIZE, tx0 * _TILE_SIZE
    clip_h = min(buf_h, viewbox.num_pixels_y - y0_px)
    clip_w = min(buf_w, viewbox.num_pixels_x - x0_px)
    return projected_image[:clip_h, :clip_w], y0_px, x0_px

# One whole tile of guaranteed-zero slack snapped around the projected
# footprint. The tile-snap already rounds the pixel bbox OUT to whole tiles;
# this extra tile absorbs the sub-pixel disagreement between the polygon
# projection here and place_image's own per-pixel unproject path when a
# footprint edge lands right on a tile boundary. Far larger than the erosion
# halo, so the eroded boundary always sees the same zeros the full canvas had.
_BBOX_TILE_MARGIN = 1
# Per-edge WCS samples for the footprint polygon — follows the image edges
# (incl. SIP distortion) closely; project_polygon then densifies each
# great-circle edge further, so the projected bounds bound the true footprint.
_FOOTPRINT_WCS_STEPS = 16


def _footprint_tile_bbox(
    wcs: WCS, viewbox: Viewbox, source_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Tile sub-rectangle `(ty0, ty1, tx0, tx1)` of `viewbox` covering where the
    source image projects — or None if it lands entirely off-canvas.

    Reuses the mosaic footprint machinery: `from_wcs` samples the WCS boundary
    into a spherical polygon, `project_to_viewbox` maps it to canvas pixels via
    the same `(x - x_min)/resolution` that `place_tile`'s unproject is the
    inverse of. We take the pixel bounds, snap OUT to whole tiles plus a
    one-tile margin, and clip to the canvas. Whole tiles keep each `place_tile`
    call byte-identical to the full-canvas run; the margin makes the cover a
    superset, so streaming just the bbox is bit-identical to the full canvas."""
    from spherical_geometry.polygon import SingleSphericalPolygon
    from mosaics.trimming import project_to_viewbox

    # `from_wcs` reads the image size off the WCS; set it explicitly from the
    # source we actually read, so the footprint is right even if the plate-
    # solution header carries no NAXIS.
    src_h, src_w = source_shape
    wcs.pixel_shape = (src_w, src_h)
    footprint = SingleSphericalPolygon.from_wcs(wcs, steps=_FOOTPRINT_WCS_STEPS)
    minx, miny, maxx, maxy = project_to_viewbox(footprint, viewbox).bounds
    if not all(np.isfinite([minx, miny, maxx, maxy])):
        return None

    m = _BBOX_TILE_MARGIN
    ty0 = max(0, int(np.floor(miny / _TILE_SIZE)) - m)
    ty1 = min(_num_tiles(viewbox.num_pixels_y), int(np.ceil(maxy / _TILE_SIZE)) + m)
    tx0 = max(0, int(np.floor(minx / _TILE_SIZE)) - m)
    tx1 = min(_num_tiles(viewbox.num_pixels_x), int(np.ceil(maxx / _TILE_SIZE)) + m)
    if ty0 >= ty1 or tx0 >= tx1:
        return None
    return (ty0, ty1, tx0, tx1)


def _assert_footprint_within_bbox(
    image: np.ndarray, y0_px: int, x0_px: int, viewbox: Viewbox,
) -> None:
    """Fail-fast guard that the projected footprint really fits inside the bbox
    buffer: no non-zero pixel may touch a buffer border that is interior to the
    canvas (a border flush against the canvas edge is fine — the footprint
    legitimately reaches the mosaic edge there). If this fires, the bbox
    under-covered and streaming just the bbox would drop footprint the full
    canvas had, so we refuse rather than write a wrong panel."""
    h, w = image.shape
    rows = np.any(image != 0, axis=1)
    if not rows.any():
        return  # blank panel; place_image's own np.any check raises
    cols = np.any(image != 0, axis=0)
    top = int(np.argmax(rows)) == 0
    bottom = int(np.argmax(rows[::-1])) == 0
    left = int(np.argmax(cols)) == 0
    right = int(np.argmax(cols[::-1])) == 0
    # A buffer border is on the canvas edge (so touching it is legitimate) when
    # its canvas coordinate is 0 or num_pixels.
    assert not (top and y0_px != 0), "footprint hits interior top of bbox"
    assert not (bottom and y0_px + h != viewbox.num_pixels_y), \
        "footprint hits interior bottom of bbox"
    assert not (left and x0_px != 0), "footprint hits interior left of bbox"
    assert not (right and x0_px + w != viewbox.num_pixels_x), \
        "footprint hits interior right of bbox"


# Erosion is done in `_EROSION_TILE`-sized blocks across a thread pool: the
# global distance transform was the single-threaded, full-canvas-memory tail
# of `place_image`. Each block distance-transforms its own footprint plus a
# `px` halo — provably (and empirically, bit-for-bit) identical to the global
# transform for the only thing we use it for, the `< px` threshold, since a
# pixel within `px` of the boundary has that boundary pixel inside an
# axis-aligned `px` box. `_EROSION_HALO_EXTRA` is pure insurance (px alone
# verified sufficient across shapes/tile sizes).
_EROSION_TILE = 2048
_EROSION_HALO_EXTRA = 4


def _erode_edges(image: np.ndarray, px: int, tile: int = _EROSION_TILE) -> np.ndarray:
    """Shape-following inward erosion of the valid-data (non-zero) region
    by `px` pixels. Removes the registration interpolation/partial-coverage
    rolloff at frame borders before the mosaic merge — PhotometricMosaic
    samples the overlap, and un-eroded edge pixels cause seam artifacts.

    `image != 0` is the footprint (the interpolator zero-fills outside it);
    the distance transform gives an exact Euclidean inward shave following
    the footprint's actual shape (not a rectangle crop), zeroing every pixel
    within `px` of the boundary. `px <= 0` is a no-op.

    No hole-filling: `place_image` writes every output pixel by projecting
    the (gap-free) source rectangle, so the footprint is a solid, simply-
    connected region — the projection cannot manufacture interior holes, and
    the pedestal-normalized masters carry a ~0.005 floor so there are no
    exact-zero *interior* values to erode around either. Distance-to-nearest-
    zero therefore already measures distance to the single outer boundary,
    which is the whole point of the shave. Dropping the global
    `binary_fill_holes` (May-2026 addition) is bit-identical on real data.

    **Erodes IN PLACE and in parallel.** The footprint `mask` is snapshotted
    from the pre-erosion image so every block reads a stable footprint (a
    block's halo must not observe a neighbour that's already been eroded);
    the float pixels are then zeroed in place, block by block, across a thread
    pool. `cv2.distanceTransform` releases the GIL and blocks write disjoint
    slices, so this turns the erosion from a single-threaded full-canvas pass
    (its own `dist` + `where` + `astype` copies, ~2–3× the canvas in RAM) into
    a many-core pass whose only extra allocation is the uint8 `mask` plus a
    handful of small per-block temporaries."""
    if px <= 0:
        return image
    from concurrent.futures import ThreadPoolExecutor

    H, W = image.shape
    mask = (image != 0).astype(np.uint8)
    halo = px + _EROSION_HALO_EXTRA

    def erode_block(y0: int, x0: int) -> None:
        y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
        hy0, hx0 = max(0, y0 - halo), max(0, x0 - halo)
        hy1, hx1 = min(H, y1 + halo), min(W, x1 + halo)
        sub = mask[hy0:hy1, hx0:hx1]
        if not sub.any():
            image[y0:y1, x0:x1] = 0.0        # block wholly outside footprint
            return
        if sub.all():
            return                            # block wholly interior -> keep
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        iy, ix = y0 - hy0, x0 - hx0
        di = dist[iy:iy + (y1 - y0), ix:ix + (x1 - x0)]
        image[y0:y1, x0:x1][di < px] = 0.0

    coords = [(y0, x0) for y0 in range(0, H, tile) for x0 in range(0, W, tile)]
    with ThreadPoolExecutor() as ex:
        list(ex.map(lambda c: erode_block(*c), coords))
    return image


def _place_image_narrow(args, kwargs):
    """Cache-key hook: project `config` to its only dependency here,
    `edge_erosion_px`, in its arg slot. `config` is a literal that sits
    BEFORE the trailing `image_metadata` data-dependency (the graph
    appends dep results last), so it's the second-to-last positional."""
    mosaic, target, filter, viewbox, config, image_metadata = args
    return (mosaic, target, filter, viewbox,
            config.edge_erosion_px, image_metadata), kwargs


# ephemeral: a multi-GB reprojection whose ONLY consumer is the merge that
# immediately follows it; CPU-cheap to re-derive.
@node(
    ephemeral=True,
    cache_key_narrow=_place_image_narrow,
    auto_memory=True,
    # Reprojected into the mosaic's Mercator viewbox: non-square pixels, can't
    # solve. (A Target's viewbox is now gnomonic-with-rotation, which IS a valid
    # TAN frame with square pixels, so single-field placements could carry a
    # real WCS — worth revisiting, but flipping this re-keys every placement.)
    plate_solve=False,
)
def place_image(
    mosaic: Mosaic, target: Target, filter: str, viewbox: Viewbox,
    config: ProcessingConfig, image_metadata: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Project one panel's master into the mosaic's viewbox, then erode
    its valid-data edge.

    Pipeline-agnostic: the downscaling — if any — is fully captured
    by the `viewbox` size and by the upstream master (already
    downscaled for the PixInsight Downscaled pipeline). place_image
    just projects whatever it's given. The on-disk filename includes
    a viewbox content-hash prefix so PI Master and PI Downscaled outputs land in
    distinct files.

    Takes the whole `config`; the cache key is narrowed to just
    `edge_erosion_px` (the only field this op depends on)."""
    assert image_metadata.plate_solution is not None, (
        f"place_image requires a plate solution on {image_metadata.filename}, "
        "but it has none. Its producer must plate-solve (the default); only "
        "ops with no solvable star field set plate_solve=False."
    )
    log.info("Placing image", mosaic=str(mosaic), target=str(target),
             filter=filter, filename=image_metadata.filename, viewbox=str(viewbox))

    wcs = WCS(image_metadata.plate_solution.header)
    interp, source_shape = _make_interpolator(image_metadata.filename)

    # Project only the tiles the source footprint actually covers, into a
    # bbox-sized buffer, then stream a full-canvas TIFF by lazily embedding that
    # bbox on a black canvas — so RSS stays at bbox size, not the ~9 GB gigapixel
    # canvas that most panels barely touch. The file on disk is byte-identical
    # to the old full-canvas write; only the peak memory changes.
    bbox = _footprint_tile_bbox(wcs, viewbox, source_shape)
    if bbox is None:
        raise ProjectionFailedException(
            f"Source for target {target.stable_hash} in mosaic "
            f"{mosaic.stable_hash} ({filter}) projects entirely off the "
            f"viewbox — likely a WCS or viewbox mismatch."
        )
    image, y0_px, x0_px = _project_in_tiles(viewbox, interp, wcs, tile_range=bbox)
    _assert_footprint_within_bbox(image, y0_px, x0_px, viewbox)
    image = _erode_edges(image, config.edge_erosion_px)

    # `image_metadata` MUST flow into the call_hash. Three
    # pipelines (Subframe, PI Master, PI Downscaled) place the
    # same target panel into the same viewbox at the same filter,
    # differing only in their upstream master. Omitting
    # `image_metadata` here collapses all three onto a single
    # output filename — multiple writers race, downstream readers
    # see half-written zlib streams (corrupt TIFF).
    output_filename = place_image.filename_fn(
        mosaic, target, filter, viewbox, config, image_metadata,
    )

    if not np.any(image):
        raise ProjectionFailedException(
            f"Placed image for target {target.stable_hash} in mosaic "
            f"{mosaic.stable_hash} ({filter}) is entirely blank. This "
            f"likely indicates a WCS or viewbox mismatch."
        )

    # bbox-on-disk: persist only the projected bbox pixels + the bbox's (x, y)
    # offset and the canvas size (in the header), NOT a full-canvas embed. The
    # OSS/Subframe mergers read the small array + offset directly (no full-canvas
    # decompress); the PI PhotometricMosaic path re-hydrates to full-canvas
    # deflate on demand (see panel_io). zstd is fine — place_image no longer feeds
    # PixInsight directly.
    write_placed_panel(
        output_filename, image,
        offset_xy=(x0_px, y0_px),
        canvas_size=(viewbox.num_pixels_x, viewbox.num_pixels_y),
        metadata={"FILTER": filter, "STBLHASH": target.stable_hash},
    )
    # A placed image is one panel by definition. merge_images sums
    # input counts to track "most complete merge so far." Stored in the
    # context cache (not the file sidecar) because @makes_image wipes
    # the sidecar after every producer run.
    from mosaics.merging import _set_panel_count
    _set_panel_count(output_filename, 1)
    log.info("Place image complete", output_filename=output_filename)

    return WithSidecar(output_filename)
