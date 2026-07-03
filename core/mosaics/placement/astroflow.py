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
from file_io.tiff_file_io import write_tiff_image
from file_io.types import TiffFilePath, WithSidecar
from context import context
from mosaics.types import Mosaic, Viewbox
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
        viewbox.y_min + tile_y * viewbox.tile_height * viewbox.resolution
    )
    tile_y_max = tile_y_min + viewbox.tile_height * viewbox.resolution

    tile_x_min = (
        viewbox.x_min + tile_x * viewbox.tile_width * viewbox.resolution
    )
    tile_x_max = tile_x_min + viewbox.tile_width * viewbox.resolution

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
        tile_y_min, tile_y_max, viewbox.tile_height, endpoint=False
    )
    xs = np.linspace(tile_x_min, tile_x_max, viewbox.tile_width, endpoint=False)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    ras, decs = viewbox.projection.unproject(xx, yy)
    ras, decs = np.degrees(ras), np.degrees(decs)

    source_xs, source_ys = wcs.all_world2pix(ras, decs, 0, quiet=True)
    tile_data = interp((source_ys, source_xs)).astype(np.float32)
    return tile_y, tile_x, tile_data

def _make_interpolator(filename: Path):
    # TODO: The subframes need to be calibrated before being placed.
    image = read_image(filename)

    # TODO: This seems to take up a huge amount of memory with method 'cubic.'
    # Maybe tiling would reduce the intermediate memory usage.
    source_height, source_width = image.shape
    return RegularGridInterpolator(
        (range(source_height), range(source_width)),
        image,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

def _project_in_tiles(
    viewbox: Viewbox,
    interp: RegularGridInterpolator,
    wcs: WCS,
    skip_empty: bool = True,
):
    """Project every tile of the source image into `viewbox`.
    `interp` and `wcs` stay in this worker's Python heap; per-tile
    work runs on a thread pool sharing both references. The
    interpolation call (`interp((source_ys, source_xs))`) is
    numpy-heavy and releases the GIL, so threads get real
    parallelism without copying anything."""
    from concurrent.futures import ThreadPoolExecutor

    projected_image = np.zeros(
        (
            viewbox.num_tiles_y * viewbox.tile_height,
            viewbox.num_tiles_x * viewbox.tile_width,
        ),
        dtype=np.float32,
    )

    tiles = list(product(
        range(viewbox.num_tiles_y), range(viewbox.num_tiles_x),
    ))
    with ThreadPoolExecutor() as ex:
        for tile_y, tile_x, tile_data in ex.map(
            lambda tx: place_tile(
                viewbox, tx[0], tx[1], wcs, interp, skip_empty,
            ),
            tiles,
        ):
            if tile_data is None:
                continue
            tile_y_idx = tile_y * viewbox.tile_height
            tile_x_idx = tile_x * viewbox.tile_width
            projected_image[
                tile_y_idx : tile_y_idx + viewbox.tile_height,
                tile_x_idx : tile_x_idx + viewbox.tile_width,
            ] = tile_data

    return projected_image[0 : viewbox.num_pixels_y, 0 : viewbox.num_pixels_x]

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


@node(
    cache_key_narrow=_place_image_narrow,
    auto_memory=True,
    plate_solve=False,  # reprojected into the mosaic viewbox (Mercator-like) — non-square pixels, can't solve
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
    interp = _make_interpolator(image_metadata.filename)
    image = _project_in_tiles(viewbox, interp, wcs, skip_empty=False)
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

    write_tiff_image(output_filename, image, {
        "FILTER": filter,
        "STBLHASH": target.stable_hash,
    })
    # A placed image is one panel by definition. merge_images sums
    # input counts to track "most complete merge so far." Stored in the
    # context cache (not the file sidecar) because @makes_image wipes
    # the sidecar after every producer run.
    from mosaics.merging import _set_panel_count
    _set_panel_count(output_filename, 1)
    log.info("Place image complete", output_filename=output_filename)

    return WithSidecar(output_filename)
