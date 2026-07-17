"""Generic "tile idiom" — apply a memory-heavy external image operation to a
large image by splitting it into overlapping tiles, processing each tile, and
reassembling with a streaming feathered blend.

Motivation: rc-astro (NXT/BXT/SXT) and many third-party tools load the WHOLE
image into RAM, so a gigapixel master OOM-kills them on a normal machine
(`rc-astro exited -9`). rc-astro tiles *internally* for GPU memory, but still
holds the whole image; this higher-level tiling bounds RAM/compute and lets the
per-tile work distribute across a Ray cluster.

This module is deliberately tool- and domain-agnostic (only pyvips/numpy +
`file_io`). The compute-graph wiring — per-tile `@node`s, a reassembly node, the
`config.tiling` toggle — lives with the ops in `mosaics/color/post_process.py`.
`process_in_tiles` (the traced graph-building helper) is defined there too,
because it records domain `@node`s; the three primitives here are what it and
the tile/blend node bodies call:

  * `tile_grid`               — the overlapping grid (pure, trace-time).
  * `crop_source_region`      — stream ONE tile out of a big tiled BigTIFF.
  * `feather_blend_to_bigtiff`— stream N overlapping tiles back into one BigTIFF.

Nothing here ever materializes the full canvas: `crop_source_region` reads only
one tile's pixels, and `feather_blend_to_bigtiff` builds a lazy pyvips
expression that `tiffsave` pulls tile-by-tile (each output region touches only
the few tiles overlapping it).
"""
import json
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import pyvips

from file_io.tiff_file_io import _DESC, _atomic_tiffsave, COMPATIBLE_COMPRESSION

# A tile's placement in the source/canvas: (x0, y0, width, height), all in px.
TileRect = tuple


def tile_size_for(width: int, height: int, target: int = 512):
    """A libvips tiled-TIFF tile size valid for a `width`×`height` image: a
    multiple of 16, ≤ `target`, and ≤ the image. libvips rejects a tile LARGER
    than the image ("tiff2vips: tile size out of range"), which bites small
    images (e.g. mock-test masters). Returns None — meaning write a STRIP TIFF —
    when the image is too small to tile at all."""
    def clamp(dim: int) -> int:
        return min(target, dim) // 16 * 16
    tw, th = clamp(width), clamp(height)
    return (tw, th) if tw >= 16 and th >= 16 else None


def tile_grid(width: int, height: int, tile_px: int, overlap: float) -> list[TileRect]:
    """Overlapping tile grid covering a `width`×`height` image.

    Pure and trace-time (no image read): callers pass dimensions known from the
    mosaic viewbox. `step = round(tile_px·(1-overlap))`; starts march by `step`
    with the last start clamped so the final tile ends exactly at the edge (its
    overlap with the previous tile just grows). A dimension ≤ `tile_px` yields a
    single full-length tile on that axis. Because `tile_px` is fixed, only the
    tile COUNT grows with image size — the per-tile working set never does.
    """
    if tile_px <= 0:
        raise ValueError(f"tile_px must be positive; got {tile_px}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}")
    step = max(1, round(tile_px * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile_px:
            return [0]
        s = list(range(0, length - tile_px + 1, step))
        if s[-1] != length - tile_px:
            s.append(length - tile_px)
        return s

    tw, th = min(tile_px, width), min(tile_px, height)
    return [(x, y, tw, th) for y in starts(height) for x in starts(width)]


def crop_source_region(source_path: Path | str, rect: TileRect) -> np.ndarray:
    """Stream one tile out of a (tiled) BigTIFF without loading the whole image.

    Reuses the tile-server idiom (`web_server/backend/tiles.py:_serve_tile_
    response`): open lazily with `unlimited=True` (our wide gigapixel strips
    exceed libvips' anti-bomb cap) and `crop` the region — libvips decodes only
    the touched tiles. Returns `(h, w)` for mono or `(h, w, bands)` for colour.
    """
    x0, y0, w, h = rect
    img = pyvips.Image.new_from_file(str(source_path), unlimited=True)
    return img.crop(x0, y0, w, h).numpy()


def _cosine_ramp_1d(n: int, taper: int, flat_lo: bool, flat_hi: bool) -> np.ndarray:
    """1-D raised-cosine (Hann) feather weight of length `n`: ramps 0→1 over the
    first/last `taper` px, flat 1.0 in between. A border-facing side (`flat_*`)
    stays flat 1.0 so the image edge isn't faded to black. Samples are
    half-integer centred; `np.minimum` keeps tiny tiles (overlapping ramps) ≤ 1.
    """
    w = np.ones(n, dtype=np.float32)
    t = int(min(taper, n))
    if t > 0:
        i = np.arange(t, dtype=np.float32)
        ramp = (0.5 - 0.5 * np.cos(np.pi * (i + 0.5) / taper)).astype(np.float32)
        if not flat_lo:
            w[:t] = np.minimum(w[:t], ramp)
        if not flat_hi:
            w[n - t:] = np.minimum(w[n - t:], ramp[::-1])
    return np.ascontiguousarray(w)


def feather_blend_to_bigtiff(
    rects: list[TileRect],
    tile_paths: list[Path | str],
    dest: Path | str,
    *,
    overlap: float,
    blend: str = "hann",
    metadata: dict | None = None,
    tile: tuple[int, int] | None = None,
) -> None:
    """Reassemble `tile_paths` (placed at `rects`) into one BigTIFF at `dest`,
    feathered so tile seams are invisible — fully streaming.

    Builds the lazy pyvips expression `Σ(embed(tileᵢ·wᵢ)) / Σ(embed(wᵢ))` over
    the canvas implied by `rects`, then `tiffsave`. It is demand-driven: for each
    output region libvips only pulls the 1–4 tiles that actually overlap it (the
    rest `embed` to black without reading their files), so peak RAM is a few tile
    regions — independent of total image size. The weight is a separable
    raised-cosine window built from two tiny 1-D vectors (kept alive for the lazy
    pipeline), so nothing tile-sized is held in memory either. `blend="edt"` is
    reserved for `mosaics/merging/oss._edt_weight` (not yet wired).
    """
    if blend != "hann":
        raise NotImplementedError(f"blend={blend!r} not supported yet")
    if len(rects) != len(tile_paths):
        raise ValueError(f"rects/tile_paths length mismatch: {len(rects)} vs {len(tile_paths)}")
    if not rects:
        raise ValueError("no tiles to blend")

    canvas_w = max(x + w for (x, y, w, h) in rects)
    canvas_h = max(y + h for (x, y, w, h) in rects)

    keepalive: list[np.ndarray] = []   # 1-D window buffers referenced by pyvips
    numer_terms: list[pyvips.Image] = []
    denom_terms: list[pyvips.Image] = []
    for (x0, y0, w, h), path in zip(rects, tile_paths):
        wx = _cosine_ramp_1d(w, max(1, round(w * overlap)),
                             flat_lo=(x0 == 0), flat_hi=(x0 + w >= canvas_w))
        wy = _cosine_ramp_1d(h, max(1, round(h * overlap)),
                             flat_lo=(y0 == 0), flat_hi=(y0 + h >= canvas_h))
        keepalive.extend((wx, wy))
        # Separable window as a lazy w×h image: row (w×1) tiled down, col (1×h)
        # tiled across, multiplied. The 1-D buffers are tiny (~KB) so holding
        # them alive costs nothing; a full w×h weight array is never allocated.
        row = pyvips.Image.new_from_memory(wx.data, w, 1, 1, "float").replicate(1, h)
        col = pyvips.Image.new_from_memory(wy.data, 1, h, 1, "float").replicate(w, 1)
        weight = row * col
        tile_img = pyvips.Image.new_from_file(str(path), unlimited=True)
        numer_terms.append((tile_img * weight).embed(x0, y0, canvas_w, canvas_h, extend="black"))
        denom_terms.append(weight.embed(x0, y0, canvas_w, canvas_h, extend="black"))

    numer = reduce(operator.add, numer_terms)
    denom = reduce(operator.add, denom_terms)
    # Guard the rare uncovered pixel (denom 0 → output 0 there) without changing
    # covered pixels; with a covering grid every output pixel has denom > 0.
    out = numer / (denom > 0).ifthenelse(denom, 1.0)

    out = out.copy()
    out.set_type(pyvips.GValue.gstr_type, _DESC,
                 json.dumps(metadata or {}, ensure_ascii=True))
    # Clamp the tile to the canvas (libvips rejects a tile larger than the image).
    target = min(tile) if tile else 512
    _atomic_tiffsave(out, Path(dest), tile=tile_size_for(canvas_w, canvas_h, target),
                     compression=COMPATIBLE_COMPRESSION)
    del keepalive  # explicit: window buffers no longer referenced after the save
