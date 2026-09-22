"""Pixels for an LLM: previews sized for a context window, regions and
statistics sized for computation. Every read is bounded; nothing here
decodes a gigapixel master in full."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from mcp_server.server import NotAvailableYet, context, to_json, tool
from mcp_server.entities import ObjectType

MAX_PNG_PX = 4_000_000
MAX_VALUES = 65_536
MAX_NPY_ELEMENTS = 16_000_000
MAX_SAMPLE_PX = 16_000_000
MAX_MIN_PX = 2048
MAX_CELLS = 16
_SUFFIXES = {".fit", ".fits", ".xisf", ".tif", ".tiff"}


def checked_path(path: str) -> Path:
    """An existing image file under the library's input or output roots."""
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() not in _SUFFIXES:
        raise ValueError(f"{p}: not an image file ({sorted(_SUFFIXES)})")
    if not p.is_file():
        raise ValueError(f"{p} does not exist")
    roots = [Path(r).resolve() for r in (context.input_dirs or [])]
    if context.output_dir:
        roots.append(Path(context.output_dir).resolve())
    if not any(p == r or r in p.parents for r in roots):
        raise ValueError(f"{p} is outside the library roots {[str(r) for r in roots]}")
    return p


def _is_fits(p: Path) -> bool:
    return p.suffix.lower() in (".fit", ".fits")


def _level_image(p: Path, level: int):
    """(pyvips image of `level`, keepalive) for a TIFF pyramid page or a
    memory-mapped XISF shrunk by 2**level. FITS has no levels."""
    import pyvips
    if p.suffix.lower() == ".xisf":
        from file_io.xisf_mmap import open_vips
        img, keepalive = open_vips(p)
        if level:
            img = img.shrink(2 ** level, 2 ** level)
        return img, keepalive
    base = pyvips.Image.tiffload(str(p), page=0, unlimited=True)
    try:
        n_pages = base.get("n-pages")
    except pyvips.Error:
        n_pages = 1
    if level >= n_pages:
        raise ValueError(f"{p} has {n_pages} level(s); level {level} does not exist")
    return pyvips.Image.tiffload(str(p), page=level, unlimited=True), None


def _stride_to(arr: np.ndarray, max_px: int) -> tuple[np.ndarray, int]:
    k = max(1, int(np.ceil(np.sqrt(arr.shape[0] * arr.shape[1] / max_px))))
    return arr[::k, ::k], k


def _sample(p: Path, max_px: int) -> tuple[np.ndarray, int]:
    """A float32 (H, W[, bands]) sample of at most `max_px` pixels and the
    stride it was taken at, without decoding more than necessary."""
    from previews.create_tile_pyramid import _exceeds_thumbnail_budget
    if _is_fits(p):
        from file_io.generic import read_image, read_image_geometry
        g = read_image_geometry(p)
        if _exceeds_thumbnail_budget(g.width_pixels, g.height_pixels):
            raise NotAvailableYet(f"{p} is {g.width_pixels}x{g.height_pixels}: "
                                  "too large to decode in full and it has no pyramid")
        arr = np.asarray(read_image(p), dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        return _stride_to(arr, max_px)
    if p.suffix.lower() == ".xisf":
        from file_io.xisf_mmap import open_vips
        img, keepalive = open_vips(p)
        if isinstance(keepalive, np.ndarray):
            sub, k = _stride_to(keepalive, max_px)
            return np.array(sub, dtype=np.float32), k
        k = max(1, int(np.ceil(np.sqrt(img.width * img.height / max_px))))
        return img.subsample(k, k).numpy().astype(np.float32), k
    import pyvips
    base = pyvips.Image.tiffload(str(p), page=0, unlimited=True)
    try:
        n_pages = base.get("n-pages")
    except pyvips.Error:
        n_pages = 1
    for page in range(n_pages):
        img = pyvips.Image.tiffload(str(p), page=page, unlimited=True)
        if img.width * img.height <= max_px:
            return img.numpy().astype(np.float32), 2 ** page
    if _exceeds_thumbnail_budget(base.width, base.height):
        raise NotAvailableYet(f"{p} has no pyramid level under {max_px} px and "
                              "is too large to decode in full")
    return _stride_to(base.numpy().astype(np.float32), max_px)


def _cell_medians(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(bands, rows, cols) medians of a grid of equal cells over COVERED
    pixels (exact zero is the uncovered-canvas sentinel of a placed mosaic);
    NaN for a cell with no covered pixel."""
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, nb = arr.shape
    out = np.full((nb, rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            block = arr[r * h // rows:(r + 1) * h // rows,
                        c * w // cols:(c + 1) * w // cols].reshape(-1, nb)
            for b in range(nb):
                covered = block[:, b][block[:, b] > 0]
                if covered.size:
                    out[b, r, c] = np.median(covered)
    return out


@tool()
def image_stats(path: str, cells_rows: int = 4, cells_cols: int = 4,
                max_px: int = MAX_SAMPLE_PX) -> dict:
    """Numeric summary of an image from a bounded sample (<= max_px pixels
    at a uniform stride): per band median, MAD, sigma, min/max, percentiles,
    sky median/sigma; plus a rows x cols grid of cell medians (absolute and
    relative to the band median) with its range — the large-scale
    uniformity metric: clean frames sit near 1.0-1.2, a moon ghost or
    reflection stands out at several times the sky.
    """
    from previews.linear_stats import band_stats
    p = checked_path(path)
    rows, cols = max(1, min(cells_rows, MAX_CELLS)), max(1, min(cells_cols, MAX_CELLS))
    arr, stride = _sample(p, max(1, min(max_px, MAX_SAMPLE_PX)))
    nb = 1 if arr.ndim == 2 else arr.shape[2]
    bands = band_stats(arr.reshape(-1, nb))
    cells = _cell_medians(arr, rows, cols)
    medians = np.array([b["median"] or 1.0 for b in bands])[:, None, None]
    rel = cells / medians
    return to_json({
        "path": str(p), "sample_shape": list(arr.shape), "sample_stride": stride,
        "bands": bands,
        "cells": {"rows": rows, "cols": cols, "medians": cells,
                  "relative": rel,
                  "rel_range": np.nanmax(rel, axis=(1, 2)) / np.nanmin(rel, axis=(1, 2)),
                  "max_rel_cell": np.nanmax(rel, axis=(1, 2)),
                  "uncovered_cells": int(np.isnan(rel).sum())},
    })


def _png_meta(path: Path, status: str | None, min_px: int) -> dict:
    return {"source": str(path), "status": status, "min_px": min_px}


def _thumbnail_png(p: Path, min_px: int) -> bytes:
    from previews.create_tile_pyramid import (NoPreviewAvailable,
                                              png_from_linear, pyramid_thumbnail)
    if _is_fits(p):
        arr, _ = _sample(p, max(min_px * min_px * 4, 1_000_000))
        return png_from_linear(arr, min_px)
    try:
        return pyramid_thumbnail(p, min_px)
    except NoPreviewAvailable as e:
        raise NotAvailableYet(str(e)) from e


@tool()
def preview(path: str | None = None, object_type: ObjectType | None = None,
            object_id: str | None = None, pipeline: str | None = None,
            display: str | None = None, filter: str | None = None,
            min_px: int = 512) -> list:
    """A PNG preview (STF autostretch, `min_px` on the short side, <= 2048)
    of an image file or a product. Give `path`, or an object: with
    `display` (a tap name from products_for_object) the product resolves
    through the catalog; without one, the object's canonical colour render.
    Returns the image followed by {source, status, min_px}.
    """
    from mcp.server.mcpserver import Image
    min_px = max(64, min(min_px, MAX_MIN_PX))
    if path is not None:
        p = checked_path(path)
        return [Image(data=_thumbnail_png(p, min_px), format="png"),
                _png_meta(p, None, min_px)]
    if object_type is None or object_id is None:
        raise ValueError("give either path, or object_type and object_id")
    if display is None:
        from finishes.preview import render_canonical_preview
        rendered = render_canonical_preview(object_type, object_id, pipeline)
        if rendered is None:
            raise NotAvailableYet(f"no scene format of {object_type} {object_id} "
                                  "is servable yet")
        png, status = rendered
        return [Image(data=png, format="png"),
                {"source": f"{object_type}/{object_id}", "status": status,
                 "pipeline": pipeline}]
    from catalog import queries as cat
    product = cat.Product(object_type, object_id,
                          pipeline or cat.default_pipeline_for(object_type, object_id),
                          display, filter)
    viewable, status = cat.resolve_product(product)
    if viewable is None:
        raise NotAvailableYet(f"{product.as_view()} of {object_type} {object_id} "
                              f"is {status}")
    return [Image(data=_thumbnail_png(Path(viewable), min_px), format="png"),
            _png_meta(Path(viewable), status, min_px)]


@tool()
def read_region(path: str, x0: int, y0: int, width: int, height: int,
                level: int = 0, output: Literal["png", "values"] = "png",
                save_npy_to: str | None = None) -> list | dict:
    """Pixels of a rectangle (x0, y0, width, height) of an image at pyramid
    `level` (0 = full resolution; XISF and TIFF only; FITS is level 0).
    output="png": the region autostretched (<= 4 MP). output="values":
    the linear values as nested lists (<= 65,536 elements, for a quick
    look). save_npy_to=<path>: writes the region as .npy (<= 16 M elements)
    and returns its shape and stats — the route for real computation.
    """
    from mcp.server.mcpserver import Image
    from previews.create_tile_pyramid import png_from_linear
    from previews.linear_stats import band_stats
    p = checked_path(path)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if _is_fits(p):
        if level:
            raise ValueError("FITS has no pyramid levels; use level=0")
        from file_io.generic import read_image, read_image_geometry
        g = read_image_geometry(p)
        if x0 < 0 or y0 < 0 or x0 + width > g.width_pixels or y0 + height > g.height_pixels:
            raise ValueError(f"region exceeds the {g.width_pixels}x{g.height_pixels} image")
        full = np.asarray(read_image(p), dtype=np.float32)
        if full.ndim == 3 and full.shape[0] in (1, 3):
            full = np.moveaxis(full, 0, -1)
        arr = np.array(full[y0:y0 + height, x0:x0 + width])
    else:
        img, keepalive = _level_image(p, max(0, level))
        if x0 < 0 or y0 < 0 or x0 + width > img.width or y0 + height > img.height:
            raise ValueError(f"region exceeds level {level} ({img.width}x{img.height})")
        arr = np.asarray(img.crop(x0, y0, width, height).numpy(), dtype=np.float32)
        del keepalive
    arr = arr.squeeze()
    n = int(arr.size)
    nb = 1 if arr.ndim == 2 else arr.shape[2]
    stats = band_stats(arr.reshape(-1, nb))
    meta = {"path": str(p), "level": level, "rect": [x0, y0, width, height],
            "shape": list(arr.shape), "bands": stats}
    if save_npy_to is not None:
        if n > MAX_NPY_ELEMENTS:
            raise ValueError(f"{n} elements exceeds the .npy cap of {MAX_NPY_ELEMENTS}")
        dest = Path(save_npy_to).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, arr)
        return to_json({**meta, "npy": str(dest), "dtype": str(arr.dtype)})
    if output == "values":
        if n > MAX_VALUES:
            raise ValueError(f"{n} elements exceeds the inline cap of {MAX_VALUES}; "
                             "use save_npy_to or a coarser level")
        return to_json({**meta, "values": np.round(arr, 6)})
    if n > MAX_PNG_PX * nb:
        raise ValueError(f"{n // nb} pixels exceeds the PNG cap of {MAX_PNG_PX}; "
                         "use a coarser level or a smaller rectangle")
    return [Image(data=png_from_linear(arr, None), format="png"), to_json(meta)]

