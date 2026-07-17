"""Graph-level tiling idiom — run any single-image `@node` op over big
overlapping tiles as distributable graph nodes, then feather them back.

`process_in_tiles(op, *op_args, dims=..., blend=...)` is the user-facing pipeline
helper. `op` is an ordinary `@node` shaped `op(..., image) -> image` (image LAST).
Call it directly to run whole; wrap it in `process_in_tiles` to tile — SAME node,
same args, the caller decides per use. process_in_tiles crops the source (the
last of `op_args`) into overlapping tiles via `crop_tile` nodes, runs `op` on
each tile (op is unchanged — it just sees a smaller image), and reassembles with
a streaming feather `blend`. The pure grid/crop/blend math + the real-data
equivalence proof (whole ≈ tiled, PSNR ~76 dB, no seams) live in
`file_io/tiling.py`.

Why tiling is safe here: rc-astro (BXT/NXT/SXT) already tile internally (512 px,
semi-independent for non-stationary PSF) — they're local, not global-statistics
tools — so a coarser external tiling with a feathered overlap is the same idea.
"""
import os
from pathlib import Path

from astroflow.pipeline import ephemeral_intermediates
from decisions.settings import ProcessingConfig
from decorators import node
from file_io.tiff_file_io import write_tiff_image
from file_io.tiling import tile_grid, crop_source_region, feather_blend_to_bigtiff
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic


def _crop_narrow(args, kwargs):
    # A crop's identity is (rect, source content); `mosaic` only routes the path.
    mosaic, rect, source = args
    return (mosaic, rect, source), kwargs


@node(cache_key_narrow=_crop_narrow, auto_memory=True, plate_solve=False)
def crop_tile(mosaic: Mosaic, rect: tuple, source: ImageMetadata) -> WithSidecar[TiffFilePath]:
    """One overlapping tile cropped out of the (tiled BigTIFF) source — streaming,
    so only the tile's pixels are read. Written STRIP-based so rc-astro ops can
    read it directly (rc-astro rejects tiled TIFF). The unit `process_in_tiles`
    feeds to `op`; ephemeral — reclaimed after `op` consumes it."""
    out = crop_tile.filename_fn(mosaic, rect, source)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_tiff_image(Path(out), crop_source_region(source.filename, rect),
                     {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _blend_narrow(args, kwargs):
    mosaic, config, rects, *tiles = args
    return (mosaic, config.tiling, rects, *tiles), kwargs


@node(cache_key_narrow=_blend_narrow, auto_memory=True, plate_solve=False)
def blend_tiles(mosaic: Mosaic, config: ProcessingConfig, rects: tuple,
                *tiles: ImageMetadata) -> WithSidecar[TiffFilePath]:
    """Feather the processed tiles back into one BigTIFF — streaming (never
    materializes the canvas). The generic INTERMEDIATE blend (no `display=`); a
    deliverable passes its own display-carrying blend to `process_in_tiles`."""
    out = blend_tiles.filename_fn(mosaic, config, rects, *tiles)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects], [t.filename for t in tiles], out,
        overlap=config.tiling.overlap, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def process_in_tiles(op, *op_args, dims, blend=blend_tiles):
    """Run `op` over big overlapping tiles of the source (the LAST of `op_args`),
    fanned out as distributable graph nodes, then feather them in `blend`.

    `op_args` are exactly what you'd pass `op` directly (source last); `dims` =
    (width, height) of the source, from the mosaic viewbox — the grid is built at
    TRACE time, before any node runs. `op` is unchanged: it just receives a tile
    image instead of the whole one. With `config.tiling.enabled` False it's a
    single full-image tile (i.e. `op` whole). The `ProcessingConfig` is found in
    `op_args`; tiles are ephemeral (reclaimed after `blend`); the deliverable
    identity lives on `blend`."""
    mosaic, source, lead = op_args[0], op_args[-1], op_args[:-1]
    config = next(a for a in op_args if isinstance(a, ProcessingConfig))
    w, h = dims
    if config.tiling.enabled:
        rects = tuple(tile_grid(w, h, config.tiling.tile_px, config.tiling.overlap))
    else:
        rects = ((0, 0, w, h),)   # single full-image tile == run `op` whole
    with ephemeral_intermediates():
        tiles = [op(*lead, crop_tile(mosaic, r, source)) for r in rects]
    return blend(mosaic, config, rects, *tiles)
